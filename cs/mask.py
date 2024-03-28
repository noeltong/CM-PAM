import torch
import numpy as np
import math
import random


def get_mask(
    img,
    size,
    batch_size,
    type="gaussian2d",
    acc_factor=8,
    center_fraction=0,
    fix=False,
):
    mux_in = size**2
    if type.endswith("2d"):
        Nsamp = mux_in // acc_factor
    elif type.endswith("1d"):
        Nsamp = size // acc_factor
    if type == "gaussian2d":
        mask = torch.zeros_like(img)
        cov_factor = size * (5. / 128)
        mean = [size // 2, size // 2]
        cov = [[size * cov_factor, 0], [0, size * cov_factor]]
        if fix:
            samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
            int_samples = samples.astype(int)
            int_samples = np.clip(int_samples, 0, size - 1)
            mask[..., int_samples[:, 0], int_samples[:, 1]] = 1
        else:
            for i in range(batch_size):
                # sample different masks for batch
                samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
                int_samples = samples.astype(int)
                int_samples = np.clip(int_samples, 0, size - 1)
                mask[i, :, int_samples[:, 0], int_samples[:, 1]] = 1
    elif type == "uniformrandom2d":
        mask = torch.zeros_like(img)
        if fix:
            mask_vec = torch.zeros([1, size * size])
            samples = np.random.choice(size * size, int(Nsamp))
            mask_vec[:, samples] = 1
            mask_b = mask_vec.view(size, size)
            mask[:, ...] = mask_b
        else:
            for i in range(batch_size):
                # sample different masks for batch
                mask_vec = torch.zeros([1, size * size])
                samples = np.random.choice(size * size, int(Nsamp))
                mask_vec[:, samples] = 1
                mask_b = mask_vec.view(size, size)
                mask[i, ...] = mask_b
    elif type == "gaussian1d":
        mask = torch.zeros_like(img)
        mean = size // 2
        std = size * (60 / 128)
        Nsamp_center = int(size * center_fraction)
        if fix:
            samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp * 1.2))
            int_samples = samples.astype(int)
            int_samples = np.clip(int_samples, 0, size - 1)
            mask[..., int_samples] = 1
            c_from = size // 2 - Nsamp_center // 2
            mask[..., c_from : c_from + Nsamp_center] = 1
        else:
            for i in range(batch_size):
                samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp * 1.2))
                int_samples = samples.astype(int)
                int_samples = np.clip(int_samples, 0, size - 1)
                mask[i, :, :, int_samples] = 1
                c_from = size // 2 - Nsamp_center // 2
                mask[i, :, :, c_from : c_from + Nsamp_center] = 1
    elif type == "uniform1d":
        mask = torch.zeros_like(img)
        if fix:
            Nsamp_center = int(size * center_fraction)
            samples = np.random.choice(size, int(Nsamp - Nsamp_center))
            mask[..., samples] = 1
            # ACS region
            c_from = size // 2 - Nsamp_center // 2
            mask[..., c_from : c_from + Nsamp_center] = 1
        else:
            for i in range(batch_size):
                Nsamp_center = int(size * center_fraction)
                samples = np.random.choice(size, int(Nsamp - Nsamp_center))
                mask[i, :, :, samples] = 1
                # ACS region
                c_from = size // 2 - Nsamp_center // 2
                mask[i, :, :, c_from : c_from + Nsamp_center] = 1
    elif type == 'sparse2d':
        mask = torch.zeros_like(img)
        sep_axis = int(math.sqrt(acc_factor))
        start_idx = int(size - int(size // sep_axis) * sep_axis) // 2
        mask[..., start_idx::sep_axis, start_idx::sep_axis] = 1

    elif type == 'sparse1d':
        mask = torch.zeros_like(img)
        start_idx = int(size - int(size // acc_factor) * acc_factor) // 2
        mask[..., start_idx::acc_factor] = 1

    # elif type == 'drop1d':
    #     mask = torch.ones_like(img)
    #     start_idx = random.randint(0, size - Nsamp)
    #     mask[..., start_idx:(start_idx + Nsamp)] = 0



    else:
        NotImplementedError(f"Mask type {type} is currently not supported.")

    return mask
