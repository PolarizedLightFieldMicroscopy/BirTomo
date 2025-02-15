"""Functions for modifying the shape and contents of a
birefringent volume."""

import numpy as np
from VolumeRaytraceLFM.utils.orientation_utils import undo_transpose_and_flip
from VolumeRaytraceLFM.utils.lightfield_utils import average_intensity_per_lenslet
from VolumeRaytraceLFM.utils.dimensions_utils import extend_image_with_borders


def pad_to_region_shape(delta_n, optic_axis, volume_shape, region_shape):
    """
    Args:
        delta_n (np.array): 3D array with dimension volume_shape
        optic_axis (np.array): 4D array with dimension (3, *volume_shape)
        volume_shape (np.array): dimensions of object volume
        region_shape (np.array): dimensions of the region fitting the object,
                                    values must be less than volume_shape
    Returns:
        padded_delta_n (np.array): 3D array with dimension region_shape
        padded_optic_axis (np.array): 4D array with dimension (3, *region_shape)
    """
    assert (
        volume_shape <= region_shape
    ).all(), "Error: volume_shape must be less than region_shape"
    z_, y_, x_ = region_shape
    z, y, x = volume_shape
    z_pad = abs(z_ - z)
    y_pad = abs(y_ - y)
    x_pad = abs(x_ - x)
    padded_delta_n = np.pad(
        delta_n,
        (
            (z_pad // 2, z_pad // 2 + z_pad % 2),
            (y_pad // 2, y_pad // 2 + y_pad % 2),
            (x_pad // 2, x_pad // 2 + x_pad % 2),
        ),
        mode="constant",
    ).astype(np.float64)
    padded_optic_axis = np.pad(
        optic_axis,
        (
            (0, 0),
            (z_pad // 2, z_pad // 2 + z_pad % 2),
            (y_pad // 2, y_pad // 2 + y_pad % 2),
            (x_pad // 2, x_pad // 2 + x_pad % 2),
        ),
        mode="constant",
        constant_values=np.sqrt(3),
    ).astype(np.float64)
    return padded_delta_n, padded_optic_axis


def crop_to_region_shape(delta_n, optic_axis, volume_shape, region_shape):
    """
    Args:
        delta_n (np.array): 3D array with dimension volume_shape
        optic_axis (np.array): 4D array with dimension (3, *volume_shape)
        volume_shape (np.array): dimensions of object volume
        region_shape (np.array): dimensions of the region fitting the object,
                                    values must be greater than volume_shape
    Returns:
        cropped_delta_n (np.array): 3D array with dimension region_shape
        cropped_optic_axis (np.array): 4D array with dimension (3, *region_shape)
    """
    assert (
        volume_shape >= region_shape
    ).all(), "Error: volume_shape must be greater than region_shape"
    crop_start = (volume_shape - region_shape) // 2
    crop_end = crop_start + region_shape
    cropped_delta_n = delta_n[
        crop_start[0] : crop_end[0],
        crop_start[1] : crop_end[1],
        crop_start[2] : crop_end[2],
    ]
    cropped_optic_axis = optic_axis[
        :,
        crop_start[0] : crop_end[0],
        crop_start[1] : crop_end[1],
        crop_start[2] : crop_end[2],
    ]
    return cropped_delta_n, cropped_optic_axis


def scale_birefringence_z_projection_center(birefringence, ret_avg):
    """Scale the birefringence array by projecting the ret_avg along the z axis.
    If the y and x dimensions of ret_avg do not match the birefringence array, 
    scale only the centered portion of the birefringence array.
    Args:
        birefringence (numpy.array): The 3D birefringence array (z, y, x).
        ret_avg (numpy.array): The 2D array of average intensity values (y, x).
    Returns:
        numpy.array: The birefringence array scaled by ret_avg projected along z.
    """
    bir_copy = birefringence.copy()
    z_dim, y_dim, x_dim = bir_copy.shape
    ret_avg = ret_avg.T
    ret_y, ret_x = ret_avg.shape
    ret_avg = ret_avg / np.max(ret_avg)
    
    # If the dimensions match, scale the whole array
    if ret_y == y_dim and ret_x == x_dim:
        ret_avg_expanded = np.repeat(ret_avg[np.newaxis, :, :], z_dim, axis=0)
        bir_copy = bir_copy * ret_avg_expanded
    else:
        # Otherwise, scale only the center portion
        
        # Calculate the start and end indices for the center region
        start_y = (y_dim - ret_y) // 2
        end_y = start_y + ret_y
        start_x = (x_dim - ret_x) // 2
        end_x = start_x + ret_x
        
        # Expand ret_avg along the z axis to match the center region of birefringence
        ret_avg_expanded = np.repeat(ret_avg[np.newaxis, :, :], z_dim, axis=0)
        
        # Scale only the center region of the birefringence array
        bir_copy[:, start_y:end_y, start_x:end_x] *= ret_avg_expanded
    return bir_copy


def adjust_birefringence_distribution_from_retardance(initial_birefringence, ret_image_meas, optical_info):
    # Adjust the initial volume to match the retardance image
    ret_image_meas_oriented = undo_transpose_and_flip(ret_image_meas)
    ret_avg = average_intensity_per_lenslet(ret_image_meas_oriented, optical_info["pixels_per_ml"])#.T
    vol_shape = optical_info["volume_shape"]
    ret_avg = extend_image_with_borders(ret_avg, (vol_shape[1], vol_shape[2]))
    scaled_birefringence = scale_birefringence_z_projection_center(initial_birefringence, ret_avg)
    return scaled_birefringence
