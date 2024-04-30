import torch
import numpy as np


def check_valid_JM(arr):
    pass


def check_for_inf_or_nan(arr):
    if torch.is_tensor(arr):
        if torch.isinf(arr).any() or torch.isnan(arr).any():
            raise ValueError("Array contains an inf of nan.")
        pass
    elif np.is_array(arr):
        pass
    else:
        raise TypeError("Array is not a torch tensor or numpy array")


def check_for_nans(tensor):
    """
    Check if there are any NaN values in a PyTorch tensor.
    Args:
        tensor (torch.Tensor): The tensor to check for NaN values.
    Returns:
        bool: True if there are NaNs, False otherwise.
    """
    if torch.isnan(tensor).any():
        print("Warning: NaN values detected!")
        return True
    else:
        print("No NaN values detected.")
        return False
