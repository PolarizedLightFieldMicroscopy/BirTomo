"""Sets of optical info for testing."""
import numpy as np
import pytest
from VolumeRaytraceLFM.abstract_classes import OpticalElement

@pytest.fixture(scope="module")
def optical_info_vol11():
    # Fetch default optical info
    optical_info = OpticalElement.get_optical_info_template()

    optical_info['volume_shape'] = [11, 11, 11]
    optical_info['axial_voxel_size_um'] = 1.0
    optical_info['pixels_per_ml'] = 5
    optical_info['na_obj'] = 1.2
    optical_info['n_medium'] = 1.52
    optical_info['wavelength'] = 0.550
    optical_info['n_micro_lenses'] = 1
    optical_info['n_voxels_per_ml'] = 1
    optical_info['polarizer'] = np.array([[1, 0], [0, 1]])
    optical_info['analyzer'] = np.array([[1, 0], [0, 1]])

    return optical_info

def set_optical_info(vol_shape, pixels_per_ml, num_lenslets):
    # Fetch default optical info
    optical_info = OpticalElement.get_optical_info_template()

    optical_info['volume_shape'] = vol_shape
    optical_info['axial_voxel_size_um'] = 1.0
    optical_info['pixels_per_ml'] = pixels_per_ml
    optical_info['na_obj'] = 1.2
    optical_info['n_medium'] = 1.35
    optical_info['wavelength'] = 0.550
    optical_info['n_micro_lenses'] = num_lenslets
    optical_info['n_voxels_per_ml'] = 1
    optical_info['M_obj'] = 60
    optical_info['cube_voxels'] = True
    optical_info['camera_pix_pitch'] = 6.5
    optical_info['polarizer'] = np.array([[1, 0], [0, 1]])
    optical_info['analyzer'] = np.array([[1, 0], [0, 1]])
    optical_info['polarizer_swing'] = 0.03

    return optical_info
