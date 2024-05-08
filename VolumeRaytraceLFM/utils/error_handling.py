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


def check_for_negative_values(my_list):
    if any(value < 0 for value in my_list):
        negative_value = next(value for value in my_list if value < 0)
        raise ValueError(f"Invalid value: {negative_value}. The list should not contain negative integers.")


def check_for_negative_values_dict(my_dict):
    # Iterate over each sublist in the lists of lists stored as values in the dictionary
    if any(value < 0 for nested_list in my_dict.values() for sublist in nested_list for value in sublist):
        raise ValueError("The dictionary contains negative values.")
    else:
        print("All entries are nonnegative.")
