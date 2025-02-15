import torch
import numpy as np
import math


def check_valid_JM(arr):
    pass


def check_for_inf_or_nan(arr):
    if torch.is_tensor(arr):
        if torch.isinf(arr).any() or torch.isnan(arr).any():
            raise ValueError("Array contains an inf of nan.")
        pass
    elif isinstance(arr, np.ndarray):
        if np.isinf(arr).any() or np.isnan(arr).any():
            raise ValueError("Array contains an inf or nan.")
    elif isinstance(arr, (float, int)):
        if math.isinf(arr) or math.isnan(arr):
            raise ValueError("Input is an inf or nan.")
    else:
        raise TypeError("Input is not a torch tensor, numpy array, float, or int.")


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
        raise ValueError(
            f"Invalid value: {negative_value}. The list should not contain negative integers."
        )


def check_for_negative_values_dict(my_dict):
    # Iterate over each sublist in the lists of lists stored as values in the dictionary
    if any(
        value < 0
        for nested_list in my_dict.values()
        for sublist in nested_list
        for value in sublist
    ):
        raise ValueError("The dictionary contains negative values.")
    else:
        print("All entries are nonnegative.")


def check_for_negative_values_list_of_lists(my_list):
    if any(value < 0 for sublist in my_list for value in sublist):
        raise ValueError("The list of lists contains negative values.")
    else:
        print("All entries are nonnegative.")


def replace_nans_in_optic_axis(volume):
    """Used in response to an error message."""
    with torch.no_grad():
        num_nan_vecs = torch.sum(torch.isnan(volume.optic_axis[0, :]))
        if num_nan_vecs > 0:
            replacement_vecs = torch.nn.functional.normalize(
                torch.rand(3, int(num_nan_vecs)), p=2, dim=0
            )
            volume.optic_axis[:, torch.isnan(volume.optic_axis[0, :])] = (
                replacement_vecs
            )
            print(
                f"Replaced {num_nan_vecs} NaN optic axis vectors with random unit vectors."
            )
