import numpy as np
import torch


def transpose_and_flip(data):
    if isinstance(data, np.ndarray):
        result = np.flip(data.T, axis=1).copy()
    elif isinstance(data, torch.Tensor):
        result = torch.flip(data.transpose(0, 1), dims=[1]).clone()
    else:
        raise TypeError("Input must be either a NumPy array or PyTorch tensor.")
    return result


def undo_transpose_and_flip(data):
    if isinstance(data, np.ndarray):
        result = np.flip(data, axis=1).T.copy()
    elif isinstance(data, torch.Tensor):
        result = torch.flip(data, dims=[1]).transpose(0, 1).clone()
    else:
        raise TypeError("Input must be either a NumPy array or PyTorch tensor.")
    return result
