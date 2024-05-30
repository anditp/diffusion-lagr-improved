import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose
sys.path.append('../')
from guided_diffusion import logger


# interpolation methods
def interp1d(sample, dim):
    original_length = sample.shape[0]
    # time vector [1,..., 2000]
    T = np.linspace(1, original_length, num=original_length)
    xnew = np.linspace(1, original_length, num=dim)
    interpolated = np.interp(xnew, T, sample)
    return interpolated

class ParticleDataset(Dataset):
    def __init__(self, npy_filename, root_dir=None, transform=None):
        super().__init__()
        self.npy_filename = npy_filename
        self.root_dir = root_dir
        self.transform = transform
        self.data = np.load(self.npy_filename, encoding="ASCII", allow_pickle=True, mmap_mode='r')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        part_traj = self.data[idx]
        if self.transform:
            part_traj = self.transform(part_traj)

        return part_traj


class TakeOneCoord(object):
    """Take one coordinate from the trajectory. The object must be a Tensor that
    has shape (num_coords, length). Therefore, take note on this. If you're working
    with the npy file, you have to permute the dimensions fist with TensorChanFirst().
    
    Args:
        coord (int): coordinate to take from the trajectory.
        
    Returns:
        torch.Tensor: a tensor of shape (1, coord, length) with the trajectory of the chosen coordinate.
    """
    def __init__(self, coord):
        self.coord = coord

    def __call__(self, sample):
        traj = sample[self.coord, :]
        return  traj.unsqueeze(0)

class TensorChanFirst(object):
    def __init(self):
        pass
    def __call__(self, sample):
        return sample.permute(1, 0)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # return scalar type float32
        # which is the default type for model weights
        return torch.from_numpy(sample).float()


def dataset_from_file(npy_fname, 
                      batch_size, 
                      coordinate=None, 
                      is_distributed=False, 
                      **kwargs):
    """
    Function that returns a DataLoader for the Lagrangian trajectories dataset.

    Args:

        npy_fname (str): path to the .npy file containing the dataset
        batch_size (int): batch size.
        levels (int): number of levels to use for the multiscale interpolation.
        is_distributed (bool): whether to use a distributed sampler or not.
        **kwargs: additional arguments to pass to the DataLoader constructor.

    Returns:
        torch.utils.data.DataLoader: a DataLoader for the dataset.
    """
    # read 3D trajectories
    # usual transformations are ToTensor() and permute(1, 0)
    # to get channel-first tensors
    transforms = [ToTensor()]
    if coordinate is not None:
        transforms.append(TakeOneCoord(coord=coordinate))
    #else:
     #   transforms.append(TensorChanFirst())
    dataset = ParticleDataset(npy_fname, transform=Compose(transforms))
    
    
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not is_distributed,
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory = True,
        drop_last=True,
        **kwargs)
    
    while True:
        yield from loader
    
    

