from abc import abstractmethod

import math
import torch
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dist_util, logger
from fp16_util import convert_module_to_f16, convert_module_to_f32
import pywt
from nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):  # 
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock): 
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h  


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape

        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])

        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class Cross attention(nn.Module):
    

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels  
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.norm1 = normalization(channels)
        self.kv = conv_nd(1, channels, channels * 2, 1)
        self.q = conv_nd(1, 5, channels, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, pan):
        return checkpoint(self._forward, (x, pan), self.parameters(), self.use_checkpoint)

    def _forward(self, x, pan):
        b, c, *spatial = x.shape

        p_b, p_c, *spatial = pan.shape
        x = x.reshape(b, c, -1)
        pan = pan.reshape(p_b, p_c, -1)
        kv = self.kv(self.norm(x))

        q = self.norm1(self.q(pan))
        kv = kv.reshape(b * self.num_heads, -1, kv.shape[2])
        q = q.reshape(b * self.num_heads, -1, q.shape[2])
        qkv = th.cat([q, kv], dim=1)

        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class Start_layer(nn.Module):
    def __init__(self, channels, dims=2):
        super().__init__()
        self.channels = channels
        self.pan1 = conv_nd(dims, 1, 4, 3, stride=1, padding=1)
        self.pan2 = conv_nd(dims, 4, 4, 3, stride=1, padding=1)
        self.pan3 = conv_nd(dims, 4, 4, 3, stride=1, padding=1)
        self.ms1 = conv_nd(dims, 4, 4, 3, stride=1, padding=1)
        self.ms2 = conv_nd(dims, 4, 4, 3, stride=1, padding=1)
        self.ms3 = conv_nd(dims, 4, 4, 3, stride=1, padding=1)
        self.wfv1 = conv_nd(dims, 4, 4, 3, stride=1, padding=1)
        self.wfv2 = conv_nd(dims, 4, 4, 3, stride=1, padding=1)
        self.wfv3 = conv_nd(dims, 4, 4, 3, stride=1, padding=1)
        self.xt = conv_nd(dims, 4, 12, 3, stride=1, padding=1)
        self.all_conv = conv_nd(dims, 48, channels, 3, stride=1, padding=1)

    def forward(self, xt, low_res, pan_res, ms_res):
        pan = torch.cat([self.pan1(pan_res), self.pan2(self.pan1(pan_res)), self.pan3(self.pan2(self.pan1(pan_res)))],
                        dim=1)
        ms = torch.cat([self.ms1(ms_res), self.ms2(self.ms1(ms_res)), self.ms3(self.ms2(self.ms1(ms_res)))], dim=1)
        low_res = torch.cat(
            [self.wfv1(low_res), self.wfv2(self.wfv1(low_res)), self.wfv3(self.wfv2(self.wfv1(low_res)))], dim=1)
        xt1 = self.xt(xt)
        output = self.all_conv(torch.cat([xt1, pan, ms, low_res], dim=1))
        assert output.shape[1] == self.channels
        return output


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            num_heads=1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
     
        self.input_blocks = nn.ModuleList(
            [
                Start_layer(model_channels,dims)
                
                #conv_nd(dims, 13, model_channels, 3, padding=1)  

            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):  # 1,2,4,8

            for j in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                # self.input_blocks.append(TimestepEmbedSequential(*layers))
                if ((level > 0) and (level != len(channel_mult) - 1)):  
                    layers.append(
                        Cross attention(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))  
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2
        
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        
        self.output_blocks = nn.ModuleList([])
        for x, (level, mult) in enumerate(list(enumerate(channel_mult))[::-1]):

            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                # if ds in attention_resolutions:
                # if (x > 0) and (x < 5):
                    # layers.append(
                        # Cross attention(
                            # ch,
                            # use_checkpoint=use_checkpoint,
                            # num_heads=num_heads_upsample,
                        # )
                    # )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        
        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, low_res, pan_res, downpan_res, ms_res,downwfv_res,y=None):  
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))  
        
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        inpuy_att = [4, 5, 7, 8, 10, 11, 13, 14]
        input_num_pan = [0, 0, 1, 1, 2, 2, 3, 3]
        input_num = 0
        output_att = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        output_num_pan = [3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0]
        output_num = 0
        h = x.type(self.inner_dtype)

        for i, module in enumerate(self.input_blocks):
            if i == 0:
                #h = module(h)
                h = module(h,low_res,pan_res,ms_res)
                hs.append(h)
            # for name,params in module.named_modules():
            elif i in inpuy_att:
                h = module[0](h, emb)
                h = module[1](h, torch.cat([downpan_res[input_num_pan[input_num]],0.1*downwfv_res[input_num_pan[input_num]]],dim=1))
                hs.append(h)
                input_num = input_num + 1
            else:
                h = module(h, emb)
                hs.append(h)
        input_num = 0

        h = self.middle_block(h, emb)

        for k, module in enumerate(self.output_blocks):
            # if k in output_att:

                # cat_in = th.cat([h, hs.pop()], dim=1)
                # h = module[0](cat_in, emb)
                # h = module[1](h, downpan_res[output_num_pan[output_num]])
                # if len(module) == 3:
                    # h = module[2](h)
                # output_num = output_num + 1
            # else:
             cat_in = th.cat([h, hs.pop()], dim=1)
             h = module(cat_in, emb)
        output_num = 0
        h = h.type(x.dtype)  

        return self.out(h)

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2 + 1, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, pan_res=None, ms_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        # upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        #x = th.cat([x, low_res, pan_res, ms_res], dim=1).to(dist_util.dev())
        downpan_res = []
        downwfv_res = []
        _, new_height, new_width, _ = x.shape
        with th.no_grad():
            pan_res1 = pan_res.cpu()
            [cl, (cH5, cV5, cD5), (cH4, cV4, cD4), (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)] = pywt.wavedec2(
                pan_res1, 'db1', level=5)            
            downpan_res.append(torch.from_numpy(np.absolute(cH1) + np.absolute(cV1)+ np.absolute(cD1)).to(dist_util.dev()))
            downpan_res.append(torch.from_numpy(np.absolute(cH2) + np.absolute(cV2) + np.absolute(cD2)).to(dist_util.dev()))
            downpan_res.append(torch.from_numpy(np.absolute(cH3) + np.absolute(cV3) + np.absolute(cD3)).to(dist_util.dev()))
            downpan_res.append(torch.from_numpy(np.absolute(cH4) + np.absolute(cV4) + np.absolute(cD4)).to(dist_util.dev()))
            downpan_res.append(torch.from_numpy(np.absolute(cH5) + np.absolute(cV5) + np.absolute(cD5)).to(dist_util.dev()))
            down_size=[2,4,8,16,32]
            for i in  down_size:                  
                   downwfv_res.append(0.1*(F.interpolate(low_res, size=[128//i, 128//i], mode="bilinear",align_corners=True)).to(dist_util.dev()))

        return super().forward(x, timesteps, low_res, pan_res, downpan_res, ms_res,downwfv_res, **kwargs)

    def get_feature_vectors(self, x, timesteps, low_res=None, **kwargs):
        _, new_height, new_width, _ = x.shape
        # upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, low_res], dim=1)
        return super().get_feature_vectors(x, timesteps, **kwargs)

