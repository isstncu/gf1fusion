"""
Train  model.
"""
import argparse
from model import dist_util, logger
from model.image_datasets2 import load_data
from model.resample import create_named_schedule_sampler
from model.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from model.train_util import TrainLoop
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

def create_argparser():
    defaults = dict(
        data_dir="sample_data",#the training data directory
        lr=1e-4,#the initial learning rate
        lr_anneal_steps=250000,#number of steps to train
        log_interval=500,#Interval for printing logs
        save_interval=50000,#Model save interval
        save_dir="save_dir",#The path to save the model
        resume_checkpoint="",#Breakpoint continuation
        diffusion_steps=1000,#Total step size of diffusion
        schedule_sampler="uniform",
        weight_decay=0.0,
        batch_size=12,
        microbatch=-1,
        ema_rate="0.9999",
        use_fp16=False,
        fp16_scale_growth=1e-3,
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
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    logger.log("creating data loader...")
    data = load_GF_data(
        args.data_dir,
        args.batch_size,
        large_size=args.large_size,
        small_size=args.small_size,
        class_cond=args.class_cond,
    )
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        save_dir=args.save_dir
    ).run_loop()

def load_GF_data(data_dir, batch_size, large_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=class_cond,
    )
    for large_batch, model_kwargs in data:
        model_kwargs["low_res"]=large_batch[1]
        model_kwargs["pan_res"]=large_batch[2]
        model_kwargs["ms_res"]=large_batch[3]
        yield large_batch, model_kwargs

if __name__ == "__main__":
    main()
