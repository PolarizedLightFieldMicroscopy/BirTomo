'''Functions to adjust PolScope images'''

import numpy as np


def normalize_retardance(ret_image_polscope, ret_ceiling, wavelength=550):
    '''Normalize the retardance image to be between 0 and pi
    Args:
        ret_image_polscope (np.array): the retardance image from the PolScope
        ret_ceiling (scalar): maximum retardance value in units of nanometers
        wavelength (scalar): wavelength of the light in units of nanometers
    Returns:
        ret_image (np.array): the normalized retardance image in units of radians
    '''
    # intensity to nanometers
    max_pixel_value = 65535  # for 16-bit images
    ret_image_nm = (ret_image_polscope / max_pixel_value) * ret_ceiling
    # nanometers to radians
    ret_image = (ret_image_nm / wavelength) * (2 * np.pi)
    assert np.min(ret_image) >= 0, f"Minimun retardance value is below zero: Min = {np.min(ret_image):.3f}"
    assert np.max(ret_image) <= np.pi, f"Maximum retardance value exceeded π: Max = {np.max(ret_image):.3f}"
    return ret_image


def normalize_azimuth(azim_image_polscope):
    '''Normalize the azimtuh image to be between 0 and pi
    Args:
        ret_image_polscope (np.array): the azimuth image from the PolScope
    Returns:
        azim_image (np.array): the normalized azimuth image in units of radians
    '''
    # intensity to degrees
    max_pixel_value = 18000  # for 16-bit images
    azim_image_degrees = azim_image_polscope / (max_pixel_value / 180)
    # degrees to radians
    azim_image = azim_image_degrees * np.pi / 180
    assert np.min(azim_image) >= 0, f"Minimun azimuth value is below zero: Min = {np.min(azim_image):.3f}"
    assert np.max(azim_image) <= np.pi, f"Maximum azimuth value exceeded π: Max = {np.max(azim_image):.3f}"
    return azim_image
