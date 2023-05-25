import os
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

import logging
LOG = logging.getLogger(__name__)

def set_max_size():
    """
    This function sets the upper limit on sequence based on how much memory is available
    on the CUDA device
    """
    if torch.cuda.is_available():
        LOG.info("CUDA device is available.")
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        LOG.info(f"Device name: {device_name}")
        # Get GPU memory information
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = gpu_memory / (1024 ** 3)
        LOG.info(f"Total GPU memory: {gpu_memory_gb:.2f} GB")
        if gpu_memory_gb < 13:
            return 670
        elif gpu_memory_gb < 17:
            return 750
        elif gpu_memory_gb < 30:
            return 950
        elif gpu_memory_gb < 40:
            return 1024
        else:
            return 1400
    else:
        return 1400
try:
    MAX_PAD = set_max_size() # for memory constraints
except Exception as exc:
    LOG.warning('Failed to set MAX_PAD based on CUDA device: Exception:', exc)
    MAX_PAD = 670
    LOG.warning('Falling back to MAX_PAD =', MAX_PAD)

def set_seeds(seed):
    if seed is not None:
        # https://pytorch.org/docs/stable/notes/randomness.html
        # also see pytorch lightning seed everything
        LOG.info(f"seeding {seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)


def get_ids(feature_dir='features/2d_features/', label_dir='features/pairwise/'):
    features = set(os.listdir(feature_dir))
    labels = set(os.listdir(label_dir))
    intersection  = labels.intersection(features)
    intersection -= {'3j80A.npz', '3j80a.npz'}
    return np.array(list(intersection))


def get_partition(ids, test_prop):
    test_idx = np.random.choice(len(ids), replace=False, size=int(test_prop*len(ids)))
    mask = np.zeros(ids.shape, dtype=bool)
    mask[test_idx] = True
    partition = {
        'train': ids[~mask],
        'test': ids[mask],
    }
    return partition


def get_torch_device(force_cpu=False):
    try:
        if torch.cuda.is_available() and not force_cpu:
            device_string = "cuda"
        # elif torch.backends.mps.is_available() and not force_cpu:
        #     device_string = "mps"
        else:
            device_string = "cpu"
    except Exception as exc:
        LOG.error(f'Exception: {exc}')
        device_string = "cpu"
    LOG.info(f'Using device: {device_string}')
    device = torch.device(device_string)
    return device


def vari_pad_collate(batch):
    """
    batch: List of tuples length=batchsize, where each tuple is a pair of torch.tensors: (features, labels)
    Alternate collate_fn for pytorch DataLoader class
    dynamically adjusts the padding size to match largest chain in batch
    See:
    torch/utils/data/_utils/collate.py
    """
    crop_size = min(max([e[1].shape[-1] for e in batch]), MAX_PAD)
    resize_x_shape = np.array(batch[0][0].shape)
    resize_y_shape = np.array(batch[0][1].shape)
    resize_x_shape[-2:] = crop_size # change size of the last two dimensions
    resize_y_shape[-2:] = crop_size
    resized_batch = []
    for x, y in batch:
        resize_x = torch.zeros(tuple(resize_x_shape))
        resize_y = torch.zeros(tuple(resize_y_shape))
        # make the padding for the flags equal 1
        fill_idx = min(crop_size, x.shape[-1])
        resize_x[..., :fill_idx, :fill_idx] = x[..., :fill_idx, :fill_idx]
        resize_y[..., :fill_idx, :fill_idx] = y[..., :fill_idx, :fill_idx]
        x_y_tuple = (resize_x, resize_y)
        resized_batch.append(x_y_tuple)
    return default_collate(resized_batch)
