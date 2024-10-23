import numpy as np


def ret_and_azim_from_intensity(image_list, swing):
    """Calculate the retardance and azimuth from a list of 5 intensity
    using the PolScope 5-frame algorithm."""
    if len(image_list) != 5:
        raise ValueError(f"Expected 5 images, got {len(image_list)}.")
    a = image_list[4] - image_list[1]
    b = image_list[2] - image_list[3]
    den = (image_list[1] + image_list[2] + image_list[3] + image_list[4] - 4 * image_list[0]) / 2
    prefactor = np.tan(np.pi * swing)

    tmp = np.arctan(prefactor * np.sqrt(a**2 + b**2) / (np.abs(den) + np.finfo(float).eps))
    ret = np.where(den == 0, np.pi / 2, tmp)
    ret = np.where(den < 0, np.pi - tmp, ret)

    azim = np.zeros_like(a)
    azim = np.where((a == 0) & (b == 0), 0, (0.5 * (np.arctan2(-a / 2, b) + np.pi)) % np.pi)

    return [ret, azim]
