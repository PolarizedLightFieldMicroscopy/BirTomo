'''Tests for RayTraceLFM class'''
import numpy as np
import torch
import pytest
import h5py
from plotly.graph_objs import Figure
from VolumeRaytraceLFM.abstract_classes import *
# from VolumeRaytraceLFM.birefringence_implementations import *

@pytest.fixture(scope="module")
def optical_info():
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

@pytest.fixture
def ray_trace_lfm_instance(request, optical_info):
    if request.param == 'numpy':
        return RayTraceLFM(backend=BackEnds.NUMPY, torch_args={}, optical_info=optical_info)
    elif request.param == 'pytorch':
        return RayTraceLFM(backend=BackEnds.PYTORCH, torch_args={}, optical_info=optical_info)

def test_rays_through_vol(pixels_per_ml=5, naObj=60, nMedium=1.52, volume_ctr_um=np.array([0.5, 0.5, 0.5])):
    ray_enter, ray_exit, ray_diff = RayTraceLFM.rays_through_vol(pixels_per_ml, naObj, nMedium, volume_ctr_um)
    
    rays_shape = ray_enter.shape
    assert rays_shape == ray_exit.shape == ray_diff.shape
    assert pixels_per_ml == rays_shape[1] == rays_shape[2]

@pytest.mark.parametrize("ray_trace_lfm_instance", ['numpy', 'pytorch'], indirect=True)
def test_compute_rays_geometry_no_file(ray_trace_lfm_instance):
    filename = None
    
    rays = ray_trace_lfm_instance
    rays.compute_rays_geometry(filename)

    assert rays.vox_ctr_idx is not None
    assert rays.volume_ctr_um is not None
    assert rays.ray_valid_indices is not None
    assert rays.ray_vol_colli_indices is not None
    assert rays.ray_vol_colli_lengths is not None
    assert rays.ray_valid_direction is not None

    assert len(rays.ray_valid_indices[0]) == len(rays.ray_vol_colli_indices)

    # Testing rays_through_vol
    rays_shape = rays.ray_entry.shape
    assert rays_shape == rays.ray_exit.shape == rays.ray_direction.shape
    assert rays.optical_info['pixels_per_ml'] == rays_shape[1] == rays_shape[2]