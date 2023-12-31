import pytest
from unittest.mock import patch, call
from tests.fixtures_backend import backend_fixture
from VolumeRaytraceLFM.simulations import (
    ForwardModel, BackEnds, BirefringentRaytraceLFM, BirefringentVolume
)


@pytest.fixture
def forward_model(backend_fixture):
    optical_system = {
        "optical_info": {
            "volume_shape": [3, 5, 5],
            "axial_voxel_size_um": 1.0,
            "cube_voxels": True,
            "pixels_per_ml": 17,
            "n_micro_lenses": 1,
            "n_voxels_per_ml": 1,
            "M_obj": 60,
            "na_obj": 1.2,
            "n_medium": 1.35,
            "wavelength": 0.550,
            "camera_pix_pitch": 6.5,
            "polarizer": [[1, 0], [0, 1]],
            "analyzer": [[1, 0], [0, 1]],
            "polarizer_swing": 0.03
        }
    }
    return ForwardModel(optical_system, backend_fixture)


@pytest.mark.parametrize("backend_fixture", ['numpy', 'pytorch'], indirect=True)
def test_mla_index(forward_model, backend_fixture):
    assert forward_model.backend == backend_fixture
    optical_info = {
        "volume_shape": [3, 7, 7],
        "axial_voxel_size_um": 1.0,
        "cube_voxels": True,
        "pixels_per_ml": 17,
        "n_micro_lenses": 1,
        "n_voxels_per_ml": 1,
        "M_obj": 60,
        "na_obj": 1.2,
        "wavelength": 0.550,
        "n_medium": 1.35,
        "camera_pix_pitch": 6.5,
    }
    volume = BirefringentVolume(
        backend=backend_fixture,
        optical_info=optical_info,
        volume_creation_args={
            'init_mode': 'single_voxel',
            'init_args': {
                'delta_n': -0.05,
                'offset': [0, 0, 0]
            }
        }
    )
    forward_model.forward_model(volume, backend_fixture)
    assert forward_model.ret_img is not None
    assert forward_model.azim_img is not None


@pytest.fixture
def birefringent_raytrace_lfm():
    # Setup for BirefringentRaytraceLFM instance
    optical_info = {
        "volume_shape": [3, 5, 5],
        "axial_voxel_size_um": 1.0,
        "cube_voxels": True,
        "pixels_per_ml": 17,
        "n_micro_lenses": 1,
        "n_voxels_per_ml": 1,
        "M_obj": 60,
        "na_obj": 1.2,
        "n_medium": 1.35,
        "wavelength": 0.550,
        "camera_pix_pitch": 6.5,
        "polarizer": [[1, 0], [0, 1]],
        "analyzer": [[1, 0], [0, 1]],
        "polarizer_swing": 0.03
    }
    BF_raytrace_torch = BirefringentRaytraceLFM(
        backend=BackEnds.PYTORCH, optical_info=optical_info
    )
    return BF_raytrace_torch


def test_form_mask_from_nonzero_pixels_dict_unique_index(birefringent_raytrace_lfm):
    with patch.object(BirefringentRaytraceLFM, '_form_mask_from_nonzero_pixels_dict') as mock_method:
        # Simulate calling the method with different indices
        # birefringent_raytrace_lfm._form_mask_from_nonzero_pixels_dict((0, 0))
        # birefringent_raytrace_lfm._form_mask_from_nonzero_pixels_dict((1, 0))
        # birefringent_raytrace_lfm._form_mask_from_nonzero_pixels_dict((0, 1))
        indices_to_test = [(0, 0), (1, 0), (0, 1)]
        for idx in indices_to_test:
            birefringent_raytrace_lfm._filter_ray_data(idx)

        # Verify that all calls to the method had unique indices
        call_args_list = mock_method.call_args_list
        indices = [call_args[0][0] for call_args in call_args_list]
        assert len(indices) == len(set(indices)), "Duplicate indices found in method calls"


def test_filter_ray_data_unique_index(birefringent_raytrace_lfm):
    rays = birefringent_raytrace_lfm
    volume = BirefringentVolume(
        backend=rays.backend,
        optical_info=rays.optical_info,
        volume_creation_args={
            'init_mode': 'single_voxel',
            'init_args': {
                'delta_n': -0.05,
                'offset': [0, 0, 0]
            }
        }
    )
    with patch.object(BirefringentRaytraceLFM, '_filter_ray_data') as mock_method:
        # Simulate calling the method with different indices

        indices_to_test = [(0, 0), (1, 0), (0, 1)]
        for idx in indices_to_test:
            if True:
                rays._filter_ray_data(idx)
            else:
                rays.calc_cummulative_JM_of_ray_torch(volume_in=volume, mla_index=idx)

        # Verify that all calls to the method had unique indices
        call_args_list = mock_method.call_args_list
        indices = [call_args[0][0] for call_args in call_args_list]
        assert len(indices) == len(set(indices)), "Duplicate indices found in method calls"
