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