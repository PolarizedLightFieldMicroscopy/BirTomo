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
    ray_enter, ray_exit, ray_diff = RayTraceLFM.rays_through_vol(
        pixels_per_ml, naObj, nMedium, volume_ctr_um
        )
    
    rays_shape = ray_enter.shape
    assert rays_shape == ray_exit.shape == ray_diff.shape
    assert pixels_per_ml == rays_shape[1] == rays_shape[2]

@pytest.mark.parametrize("ray_trace_lfm_instance", ['numpy', 'pytorch'], indirect=True)
def test_compute_lateral_ray_length_and_voxel_span(ray_trace_lfm_instance):
    '''
    Test that the voxel span is computed correctly.
    The sample ray_diff is created with pixels_per_ml = 5.
    This function is called by compute_rays_geometry, where the
        axial_volume_dim is rays.optical_info['volume_shape'][0].
    There no dependence on optical_info.
    '''
    rays = ray_trace_lfm_instance

    test_ray_diff = np.array([[[ 0.9547,  0.9719,  0.9776,  0.9719,  0.9547],
        [ 0.9719,  0.9889,  0.9944,  0.9889,  0.9719],
        [ 0.9776,  0.9944,  1.    ,  0.9944,  0.9776],
        [ 0.9719,  0.9889,  0.9944,  0.9889,  0.9719],
        [ 0.9547,  0.9719,  0.9776,  0.9719,  0.9547]],

       [[ 0.2105,  0.1053, -0.    , -0.1053, -0.2105],
        [ 0.2105,  0.1053, -0.    , -0.1053, -0.2105],
        [ 0.2105,  0.1053,  0.    , -0.1053, -0.2105],
        [ 0.2105,  0.1053,  0.    , -0.1053, -0.2105],
        [ 0.2105,  0.1053,  0.    , -0.1053, -0.2105]],

       [[ 0.2105,  0.2105,  0.2105,  0.2105,  0.2105],
        [ 0.1053,  0.1053,  0.1053,  0.1053,  0.1053],
        [-0.    ,  0.    ,  0.    ,  0.    , -0.    ],
        [-0.1053, -0.1053, -0.1053, -0.1053, -0.1053],
        [-0.2105, -0.2105, -0.2105, -0.2105, -0.2105]]])

    rays.optical_info['volume_shape'][0] = 11
    rays.optical_info["pixels_per_ml"] = 5

    # axial_volume_dim is rays.optical_info['volume_shape'][0]
    axial_volume_dim = 11
    lat_length, voxel_span = RayTraceLFM.compute_lateral_ray_length_and_voxel_span(
        test_ray_diff, axial_volume_dim
        )
    expected_voxel_span = 2.0

    assert voxel_span == expected_voxel_span, \
        f"Expected {expected_voxel_span}, got {voxel_span}"

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
    assert rays.lateral_ray_length_from_center is not None
    assert rays.voxel_span_per_ml is not None
    assert rays.ray_valid_direction is not None
    assert rays.ray_direction_basis is not None

    assert len(rays.ray_valid_indices[0]) == len(rays.ray_vol_colli_indices)

    # Testing rays_through_vol
    rays_shape = rays.ray_entry.shape
    assert rays_shape == rays.ray_exit.shape == rays.ray_direction.shape
    assert rays.optical_info['pixels_per_ml'] == rays_shape[1] == rays_shape[2]

def create_array_with_set_nonzero(size, num_nonzero):
    """
    Creates a square numpy array of the given size with num_nonzero 5 non-zero elements.

    Args:
        size (int): The size of the square array.

    Returns:
        numpy.ndarray: The resulting array.
    """
    if size * size < num_nonzero:
        raise ValueError(f"Size must be large enough to accommodate {num_nonzero} non-zero elements")

    # Create an array of zeros
    array = np.zeros((size, size), dtype=np.float32)

    # Randomly select num_nonzero unique positions to set as non-zero
    non_zero_positions = np.random.choice(size * size, num_nonzero, replace=False)
    
    # Set these positions to non-zero values
    rows = non_zero_positions // size
    cols = non_zero_positions % size
    array[rows, cols] = np.random.uniform(0.01, 1.5, size=num_nonzero)

    return array

# @pytest.mark.parametrize("ray_trace_lfm_instance", ['numpy', 'pytorch'], indirect=True)
# def test_compute_rays_geometry_no_file(ray_trace_lfm_instance):
def test_filter_nonzero_rays_single_lenslet():
    optical_info = OpticalElement.get_optical_info_template()
    optical_info['volume_shape'] = [3, 5, 5]
    optical_info['axial_voxel_size_um'] = 1.0
    optical_info['pixels_per_ml'] = 7
    optical_info['na_obj'] = 1.2
    optical_info['n_medium'] = 1.52
    optical_info['wavelength'] = 0.550
    optical_info['n_micro_lenses'] = 1
    optical_info['n_voxels_per_ml'] = 1
    rays = RayTraceLFM(backend=BackEnds.PYTORCH, torch_args={}, optical_info=optical_info)

    filename = None
    n_lenslets = rays.optical_info['n_micro_lenses']
    n_pixels_per_ml = rays.optical_info['pixels_per_ml']
    num_nonzero = 5
    ret_image = create_array_with_set_nonzero(n_lenslets * n_pixels_per_ml, num_nonzero)

    rays.compute_rays_geometry(filename=filename, image=ret_image)

    for var in [len(rays.ray_valid_indices[1]),
                len(rays.ray_vol_colli_indices),
                len(rays.ray_vol_colli_lengths),
                len(rays.ray_valid_direction)
                ]:
        assert var == num_nonzero
