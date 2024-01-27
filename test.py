"""
"""
import datetime
import time
import argparse
import os
import glob
import blobfile as bf
import numpy as np
import torch as th
import torch
import torch.distributed as dist
from image_datasets3 import load_data
import dist_util, logger
from osgeo import gdal, osr
import random
from script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("loading data...")

    data = load_data(
        data_dir=args.base_samples,
        batch_size=args.batch_size,
        class_cond=args.class_cond,
    )
    logger.log("creating samples...")
    all_images = []


    for j, super_data in enumerate(data):
        print(j)
        print(super_data[0][0].shape, len(super_data))
        res = dict(low_res=super_data[0][0], pan_res=super_data[0][1],ms_res=super_data[0][2])
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in res.items()}
        if j<540:
            if j>=0:       
                torch.manual_seed(1)
                noise=th.randn(1, 4, 128, 128).expand(args.batch_size, 4, 128, 128).to(dist_util.dev())
                sample = diffusion.p_sample_loop(
                    model,
                    (super_data[0][0].shape[0], 4, 128, 128),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,                
                )
                sample = th.div(sample, 0.0001)
                sample = sample.contiguous()
                datatype = gdal.GDT_Float32
                band_nums, height, width = sample[0].shape
                filepath = "sample_data1/output2_0.1"
                driver = gdal.GetDriverByName("GTiff")
                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
                if dist.get_rank() == 0:
                    for i in range(sample.shape[0]):
                        filename = os.path.join(filepath, "superfusion" + str(i + j * args.batch_size + 1) + '.tif')
                        print(filename)
                        dataset = driver.Create(filename, width, height, band_nums, datatype)
                        for num in range(band_nums):
                            dataset.GetRasterBand(num + 1).WriteArray(sample[i][num].cpu().numpy())
                        del dataset
                logger.log(f"created {len(all_images) * args.batch_size} samples")

    dist.barrier()
    logger.log("sampling complete")


def load_data_for_worker(base_samples, batch_size, class_cond):
    data = load_data(
        data_dir=base_samples,
        batch_size=batch_size,
        class_cond=class_cond,
    )
    batch, cond = next(data)
    res = dict(low_res=batch[0])
    yield res


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=12000,
        batch_size=120,
        last_batch_size=15,
        use_ddim=True,
        base_samples="sample_data",
        model_path="super_model2/ema0.1_0.9999_200000.pt",
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
