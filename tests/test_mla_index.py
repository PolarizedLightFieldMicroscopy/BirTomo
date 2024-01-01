import pytest
from unittest.mock import patch, call
from tests.fixtures_backend import backend_fixture
from tests.fixtures_forward_model import forward_model_fixture
from tests.fixtures_optical_info import set_optical_info
from VolumeRaytraceLFM.simulations import (
    BackEnds, BirefringentRaytraceLFM, BirefringentVolume
)


@pytest.mark.parametrize("backend_fixture", ['numpy', 'pytorch'], indirect=True)
def test_mla_index(forward_model_fixture, backend_fixture):
    assert forward_model_fixture.backend == backend_fixture
    optical_info = set_optical_info([3, 7, 7], 17, 1)
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
    forward_model_fixture.forward_model(volume, backend_fixture)
    assert forward_model_fixture.ret_img is not None
    assert forward_model_fixture.azim_img is not None


@pytest.fixture
def birefringent_raytrace_lfm():
    optical_info = set_optical_info([3, 5, 5], 17, 1)
    BF_raytrace_torch = BirefringentRaytraceLFM(
        backend=BackEnds.PYTORCH, optical_info=optical_info
    )
    return BF_raytrace_torch


def test_form_mask_from_nonzero_pixels_dict_unique_index(birefringent_raytrace_lfm):
    rays = birefringent_raytrace_lfm
    rays.compute_rays_geometry()
    method_as_str = '_form_mask_from_nonzero_pixels_dict'
    with patch.object(BirefringentRaytraceLFM, method_as_str) as mock_method:
        # Simulate calling the method with different indices
        indices_to_test = [(0, 0), (1, 0), (0, 1)]
        for idx in indices_to_test:
            rays._filter_ray_data(idx)

        # Verify that all calls to the method had unique indices
        call_args_list = mock_method.call_args_list
        indices = [call_args[0][0] for call_args in call_args_list]
        err_message = "Duplicate indices found in method calls"
        assert len(indices) == len(set(indices)), err_message


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
