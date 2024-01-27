"""
test
"""

import argparse
import os
import torch as th
import torch
import torch.distributed as dist
from model.image_datasets3 import load_data
from model import dist_util, logger
from osgeo import gdal, osr
from model.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def create_argparser():
    defaults = dict(
        base_samples="test_data",#test data directory
        model_path="ema_0.9999_200000.pt",# path to test the model
        save_data_path="save_data",#  path to save the test data
        timestep_respacing="ddim250",#Accelerated sampling step size
        clip_denoised=True,
        batch_size=120,
        use_ddim=True,
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

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
        res = dict(low_res=super_data[0][0], pan_res=super_data[0][1],ms_res=super_data[0][2])
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in res.items()}
        torch.manual_seed(1)
        sample = diffusion.p_sample_loop(
                    model,
                    (super_data[0][0].shape[0], 4, 128, 128),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,)
        sample = th.div(sample, 0.0001)
        sample = sample.contiguous()
        datatype = gdal.GDT_Float32
        band_nums, height, width = sample[0].shape
        filepath = args.save_data_path
        driver = gdal.GetDriverByName("GTiff")
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if dist.get_rank() == 0:
             for i in range(sample.shape[0]):
                        filename = os.path.join(filepath, "fusion" + str(i + j * args.batch_size + 1) + '.tif')
                        print(filename)
                        dataset = driver.Create(filename, width, height, band_nums, datatype)
                        for num in range(band_nums):
                            dataset.GetRasterBand(num + 1).WriteArray(sample[i][num].cpu().numpy())
                        del dataset
             logger.log(f"created {len(all_images) * args.batch_size} samples")
    dist.barrier()
    logger.log("sampling complete")

if __name__ == "__main__":
    main()
