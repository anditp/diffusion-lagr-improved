"""
Train a diffusion model on Lagrangian trajectories in 3d turbulence.
"""

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.train_util import TrainLoop, train_distributed
from torch.cuda import is_available, device_count
import torch
import os
from torch.multiprocessing import spawn
from guided_diffusion.turb_datasets import dataset_from_file
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)


def _get_free_port():
  import socketserver
  with socketserver.TCPServer(("localhost", 0), None) as s:
    return s.server_address[1]


def main():
    args = create_argparser().parse_args()

    replica_count = device_count()
    logger.configure(dir = "logs")

    logger.log("creating model and diffusion...")
    logger.log(replica_count)
    
    if replica_count > 1:
        if args.batch_size % replica_count != 0:
          raise ValueError(f"Batch size {args.batch_size} is not evenly divisble by # GPUs {replica_count}.")
        args.batch_size = args.batch_size // replica_count
        port = _get_free_port()
        spawn(train_distributed, args=(replica_count, port, args), nprocs=replica_count, join=True)


def create_argparser():
    defaults = dict(
        dataset_path="/home/tau/apantea/data/velocities_3d.npy",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=16,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=1000,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        coordinate = None,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
