'''Functions to adjust PolScope images'''
import skimage.io as io
import numpy as np
import time


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
    if ret_image_polscope.dtype == 'uint8':
        max_pixel_value = 255  # for 8-bit images
    elif ret_image_polscope.dtype == 'uint16':
        max_pixel_value = 65535  # for 16-bit images
    else:
        raise ValueError("Retardance data type must be uint8 or uint16")
    ret_image_nm = (ret_image_polscope / max_pixel_value) * ret_ceiling
    # nanometers to radians
    ret_image = (ret_image_nm / wavelength) * (2 * np.pi)
    assert np.min(ret_image) >= 0, f"Minimun retardance value is below zero: Min = {np.min(ret_image):.3f}"
    assert np.max(ret_image) <= np.pi, f"Maximum retardance value exceeded π: Max = {np.max(ret_image):.3f}"
    return ret_image.astype('float32')


def normalize_azimuth(azim_image_polscope):
    '''Normalize the azimtuh image to be between 0 and pi
    Args:
        ret_image_polscope (np.array): the azimuth image from the PolScope
    Returns:
        azim_image (np.array): the normalized azimuth image in units of radians
    '''
    # intensity to degrees
    if azim_image_polscope.dtype == 'uint8':
        max_pixel_value = 180  # for 8-bit images
    elif azim_image_polscope.dtype == 'uint16':
        max_pixel_value = 18000  # for 16-bit images
    else:
        raise ValueError("Azimuth image data type must be uint8 or uint16")
    azim_image_degrees = azim_image_polscope / (max_pixel_value / 180)
    # degrees to radians
    azim_image = azim_image_degrees * np.pi / 180
    assert np.min(azim_image) >= 0, f"Minimun azimuth value is below zero: Min = {np.min(azim_image):.3f}"
    assert np.max(azim_image) <= np.pi, f"Maximum azimuth value exceeded π: Max = {np.max(azim_image):.3f}"
    return azim_image.astype('float32')


def prepare_ret_azim_images(retardance_path, azimuth_path, ret_ceiling, wavelength_um):
    '''Prepare the retardance and azimuth images for reconstruction that
    were collected from the LC-PolScope.
    Args:
        retardance_path (str): path to the retardance image
        azimuth_path (str): path to the azimuth image
        ret_ceiling (scalar): maximum retardance value in nanometers
        wavelength_um (scalar): wavelength of the light in micrometers
    Returns:
        ret_image (np.array): normalized retardance image in radians
        azim_image (np.array): normalized azimuth image in radians
    '''
    start_time = time.perf_counter()
    wavelength_nm = wavelength_um * 1000
    ret_polscope = io.imread(retardance_path)
    azim_polscope = io.imread(azimuth_path)
    ret_image_meas = normalize_retardance(ret_polscope, ret_ceiling, wavelength=wavelength_nm)
    azim_image_meas = normalize_azimuth(azim_polscope)
    end_time = time.perf_counter()
    print(f"Prepared measured images in {end_time - start_time:.3f} seconds")
    return ret_image_meas, azim_image_meas
