"""
Generate a large batch of Lagrangian trajectories from a model and save them as a large
numpy array. This can be used to produce samples for statistical evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn import DataParallel as DP

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir = "logs")
    
    logger.log(args.output)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    state_dict = th.load(args.model_path)
    state_dict_new = {}
    for key in state_dict.keys():
        new_key = key[7:]
        state_dict_new[new_key] = state_dict[key]
    del state_dict

    model.load_state_dict(state_dict_new)
    device = th.device("cuda")
    model.to(device)
    
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    #noise = th.zeros(
    # noise = th.ones(
    #     (args.batch_size, args.in_channels, args.image_size),
    #     dtype=th.float32,
    #     device=dist_util.dev()
    # ) * 2
    # noise = th.from_numpy(
    #     np.load('../velocity_module-IS64-NC128-NRB3-DS4000-NScosine-LR1e-4-BS256-sample/fixed_noise_64x1x64x64.npy')
    # ).to(dtype=th.float32, device=dist_util.dev())
    import os
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        # sample_fn = (
        #     diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        # )
        sample_fn = diffusion.p_sample_loop
        sample = sample_fn(
            model,
            (args.batch_size, args.in_channels, args.image_size),
            #noise=noise,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        #sample = sample.clamp(-1, 1)
        sample[:, -1] = sample[:, -1].clamp(-1, 1)
        #sample = sample.permute(0, 2, 1)
        #sample = sample.permute(0, 1, 3, 2)
        sample = sample.contiguous()

        all_images.append(sample.cpu().numpy())
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    with open(args.output, "wb") as f:
        np.save(f, arr)
    logger.log("sampling complete")

                                                                                                        
def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=16,
        use_ddim=False,
        model_path="/home/tau/apantea/diffusion-lagr-1/logs/model000000.pt",
        output = None
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
