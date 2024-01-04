import torch
import numpy as np
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume
from VolumeRaytraceLFM.setup_parameters import (
    setup_optical_parameters
)
from VolumeRaytraceLFM.volumes import volume_args


def zero_near_zero(tensor, threshold):
    """
    Set elements of the tensor to zero if they are within a specified
    threshold of zero.

    Args:
        tensor (torch.Tensor): The input tensor.
        threshold (float): The threshold for determining "close to zero".

    Returns:
        torch.Tensor: The modified tensor with elements close to zero
                      set to zero.
    """
    # Identify elements within the threshold range
    close_to_zero = torch.logical_and(tensor > -threshold, tensor < threshold)

    # Set these elements to zero
    tensor[close_to_zero] = 0

    return tensor


def identify_close_to_zero(tensor, threshold):
    """
    Identify elements of the tensor that are within a specified
    threshold of zero.

    Args:
        tensor (torch.Tensor): The input tensor.
        threshold (float): The threshold for determining "close to zero".

    Returns:
        close_to_zero (torch.Tensor): A boolean tensor with the same
                                        shape as the input tensor.
    """
    # Identify elements within the threshold range
    close_to_zero = torch.logical_and(tensor > -threshold, tensor < threshold)

    return close_to_zero


def zero_near_zero_voxels(volume : BirefringentVolume, threshold):
    """Set voxels of the volume to zero if they are within a specified
    threshold of zero."""
    volume.Delta_n.requires_grad_(False)
    volume.optic_axis.requires_grad_(False)
    close_to_zero = identify_close_to_zero(volume.Delta_n, threshold)
    volume.Delta_n[close_to_zero] = 0
    # volume.optic_axis[:, close_to_zero] = 0

    return volume


def threshold_and_save_volume(input_vol_path, output_vol_path, optical_info, threshold):
    """Load a volume, set voxels close to zero to zero, and save the volume.
    Args:
        input_vol_path (str): Path to the input volume.
        output_vol_path (str): Path to save the output volume.
        optical_info (dict): Dictionary of optical parameters.
        threshold (float): Threshold for determining "close to zero".
    Returns:
        None
    """
    input_volume = BirefringentVolume.init_from_file(
        input_vol_path, BackEnds.PYTORCH, optical_info)
    output_volume = zero_near_zero_voxels(input_volume, threshold)
    my_description = f"Volume {input_vol_path} thresholded at {threshold}"
    output_volume.save_as_file(output_vol_path, my_description)
    print(f"Saved volume to {output_vol_path}.")


def find_3d_bounding_box(array):
    """Find the bounding box of a 3D array."""
    # Find the indices of non-zero elements
    rows, cols, depths = np.nonzero(array)

    # Determine the bounding box
    if len(rows) == 0 or len(cols) == 0 or len(depths) == 0:
        return None  # No non-zero elements in the array

    top_row = np.min(rows)
    bottom_row = np.max(rows)
    left_col = np.min(cols)
    right_col = np.max(cols)
    shallow_depth = np.min(depths)
    deep_depth = np.max(depths)

    bounding_box = (top_row, bottom_row, left_col, right_col, shallow_depth, deep_depth)
    return bounding_box

def zero_outside_bounding_box(array, bounding_box):
    """Zero out elements outside of a given bounding box.
    Args:
        array (np.ndarray): The input array.
        bounding_box (tuple): The bounding box coordinates.
    Returns:
        result (np.ndarray): The modified array.
    """
    # Make a copy of the array to avoid modifying the original
    result = array.copy()

    # Extract the bounding box coordinates
    top_row, bottom_row, left_col, right_col, shallow_depth, deep_depth = bounding_box

    # Zero out elements outside of the bounding box
    # Zero out elements above the top row
    result[:top_row, :, :] = 0
    # Zero out elements below the bottom row
    result[bottom_row + 1:, :, :] = 0
    # Zero out elements left of the left column
    result[:, :left_col, :] = 0
    # Zero out elements right of the right column
    result[:, right_col + 1:, :] = 0
    # Zero out elements shallower than the shallow depth
    result[:, :, :shallow_depth] = 0
    # Zero out elements deeper than the deep depth
    result[:, :, deep_depth + 1:] = 0

    return result

def box_out_and_save_volume(optical_config_path, random_vol_args,
                            object_args, output_file_path):
    """Load a volume, zero out elements outside the bounding box of a
    given volume, and save the modified volume.

    Args:
        optical_config_path (str): Path to the optical configuration file.
        random_vol_args (dict): Arguments for creating the random volume.
        object_args (dict): Arguments for creating the object volume.
        output_file_path (str): Path to save the output volume.
    Returns:
        None
    """
    # Set up optical parameters
    optical_info = setup_optical_parameters(optical_config_path)

    # Create a random volume
    random_volume = BirefringentVolume(
        backend=BackEnds.NUMPY,
        optical_info=optical_info,
        volume_creation_args=random_vol_args
    )
    delta_n = random_volume.get_delta_n()

    # Create a sphere volume
    sphere_volume = BirefringentVolume(
        backend=BackEnds.NUMPY,
        optical_info=optical_info,
        volume_creation_args=object_args
    )
    delta_n_sphere = sphere_volume.get_delta_n()

    # Find the bounding box of the sphere volume
    bounding_box = find_3d_bounding_box(delta_n_sphere)

    # Zero out elements outside the bounding box in the random volume
    delta_n_new = zero_outside_bounding_box(delta_n, bounding_box)
    random_volume.Delta_n = delta_n_new

    # Save the modified volume to a file
    my_description = f"Volume {random_vol_args} for area within the bounding_box of {object_args}"
    random_volume.save_as_file(output_file_path, my_description)
    print(f"Saved volume to {output_file_path}.")



if __name__ == '__main__':
    optical_config_path = "config_settings/optical_config_sphere_large_vol.json"
    random_volume_args = volume_args.random_args
    sphere_volume_args = volume_args.sphere_args6_thick
    output_file_path = "volumes/random_volume_box_sphere6_thick.h5"
    box_out_and_save_volume(optical_config_path, random_volume_args,
                            sphere_volume_args, output_file_path)
