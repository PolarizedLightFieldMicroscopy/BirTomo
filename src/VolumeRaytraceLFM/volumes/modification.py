"""Functions for modifying the shape of a birefringent volume."""

import numpy as np


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
    Parameters:
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
