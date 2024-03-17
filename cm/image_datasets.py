import os
from glob import glob
import numpy as np
from tqdm import tqdm
from torchvision import transforms as T
import torch
from scipy.io import loadmat
import cv2
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm


def min_max_scaler(x):
    x = x - x.min()
    x = x / x.max()

    return x


def load_data(
    *,
    dataset,
    batch_size,
    image_size,
    deterministic=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param dataset: choose the dataset.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if dataset == 'bone':
        path = '/data/dataset/PAM/Bone'
        all_files = list(glob(os.path.join(path, "train", "*.png")))

        dataset = PAMBoneDataset(
            image_size,
            all_files,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
        )
    elif dataset == 'brain':
        path = '/data/dataset/PAM/Brain'
        all_files = list(glob(os.path.join(path, "train", "*.tif")))
        dataset = PAMBrainDataset(
            image_size,
            all_files,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
        )
    else:
        raise ValueError(f'Dataset {dataset} unknown.')
    
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=not deterministic, num_workers=1, drop_last=True
    )
    while True:
        yield from loader


class PAMBrainDataset(torch.utils.data.Dataset):
    def __init__(self, image_size, all_paths, shard, num_shards):
        super().__init__()
        
        self.resolution = image_size
        self.num_channels = 1
        self.local_paths = all_paths[shard:][::num_shards]
        self.local_images = [
            np.array(Image.open(x)) for x in tqdm(self.local_paths)
        ]
        self.gt = []

        self.augment_fn = T.Compose([
            T.RandomCrop(self.resolution),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
        ])

    def __len__(self):
        return len(self.local_paths)
    
    def __getitem__(self, idx):
        img = torch.from_numpy(self.local_images[idx]).float()[None, ...]
        img = self.augment_fn(img)

        img = min_max_scaler(img)
        
        return img
    

class PAMBoneDataset(torch.utils.data.Dataset):
    def __init__(self, image_size, all_paths, shard, num_shards):
        super().__init__()
        
        self.resolution = image_size
        self.num_channels = 1
        self.local_paths = all_paths[shard:][::num_shards]
        self.gt = []

        self.augment_fn = T.Compose([
            T.RandomCrop(self.resolution),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
        ])

    def __len__(self):
        return len(self.local_paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.local_paths[idx], cv2.IMREAD_GRAYSCALE)[None, ...]
        img = torch.from_numpy(img).float()
        img = self.augment_fn(img)

        img = min_max_scaler(img)
        
        return img