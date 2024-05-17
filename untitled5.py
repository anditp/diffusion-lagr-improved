from .script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torch as th

class Args:
    def __init__(self):
        
        self.dims = 1
        self.image_size = 2000
        self.in_channels = 1
        self.num_channels = 32
        self.num_res_blocks = 1
        self.channel_mult = "1,2,4"
        self.diffusion_steps = 800
        self.noise_schedule = "tanh6,1"
        self.lr = 1e-5
        self.batch_size = 192
        

args = Args()

model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, model_and_diffusion_defaults().keys())
)
model.load_state_dict(th.load(args.model_path))
