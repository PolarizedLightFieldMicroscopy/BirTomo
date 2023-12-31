"""Tests for forward model simulations."""
import pytest
import numpy as np
from tests.fixtures_backend import backend_fixture
from tests.fixtures_forward_model import forward_model_fixture
from tests.fixtures_optical_info import set_optical_info
from VolumeRaytraceLFM.simulations import (
    BackEnds, BirefringentRaytraceLFM, BirefringentVolume
)


@pytest.mark.parametrize("backend_fixture", ['numpy', 'pytorch'], indirect=True)
def test_backend(forward_model_fixture, backend_fixture):
    assert forward_model_fixture.backend == backend_fixture


@pytest.mark.parametrize("backend_fixture", ['numpy', 'pytorch'], indirect=True)
def test_is_pytorch_backend(forward_model_fixture, backend_fixture):
    assert forward_model_fixture.is_pytorch_backend() == (
        backend_fixture == BackEnds.PYTORCH)


@pytest.mark.parametrize("backend_fixture", ['numpy', 'pytorch'], indirect=True)
def test_is_numpy_backend(forward_model_fixture, backend_fixture):
    assert forward_model_fixture.is_numpy_backend() == (backend_fixture == BackEnds.NUMPY)


@pytest.mark.parametrize("backend_fixture", ['numpy', 'pytorch'], indirect=True)
def test_convert_to_numpy(forward_model_fixture, backend_fixture):
    # TODO: use backend_fixture to test both numpy and pytorch
    data = np.array([1, 2, 3])
    np.testing.assert_array_equal(forward_model_fixture.convert_to_numpy(data), data)


@pytest.mark.parametrize("backend_fixture", ['numpy', 'pytorch'], indirect=True)
def test_is_pytorch_tensor(forward_model_fixture, backend_fixture):
    # TODO: use backend_fixture to test both numpy and pytorch
    data = np.array([1, 2, 3])
    assert not forward_model_fixture.is_pytorch_tensor(data)


@pytest.mark.parametrize("backend_fixture", ['numpy', 'pytorch'], indirect=True)
def test_setup_raytracer(forward_model_fixture, backend_fixture):
    # TODO: use backend_fixture to test both numpy and pytorch
    rays = forward_model_fixture.setup_raytracer()
    assert isinstance(rays, BirefringentRaytraceLFM)


@pytest.mark.parametrize("backend_fixture", ['numpy', 'pytorch'], indirect=True)
def test_forward_model(forward_model_fixture, backend_fixture):
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
