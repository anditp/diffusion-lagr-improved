from guided_diffusion.script_util import create_model_and_diffusion, args_to_dict, model_and_diffusion_defaults
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

print(model)
