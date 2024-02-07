import pytest
import torch
from VolumeRaytraceLFM.birefringence_implementations import (
    BirefringentVolume, BirefringentRaytraceLFM
)
from tests.fixtures_backend import backend_fixture
from tests.fixtures_optical_info import set_optical_info


@pytest.mark.parametrize("backend_fixture", ['numpy', 'pytorch'], indirect=True)
def test_forward_projection_exception_raising(backend_fixture):
    """Assures that forward projection works without indexing error."""
    torch.set_grad_enabled(False)

    optical_info = set_optical_info([7, 7, 7], 17, 3)   # [3, 7, 7], 11, 3
    optical_info['n_medium'] = 1.52

    BF_raytrace = BirefringentRaytraceLFM(
        backend=backend_fixture, optical_info=optical_info
        )
    BF_raytrace.compute_rays_geometry()

    voxel_volume = BirefringentVolume(
        backend=backend_fixture,
        optical_info=optical_info, 
        volume_creation_args={'init_mode': 'single_voxel'}
        )

    BF_raytrace.ray_trace_through_volume(voxel_volume)
