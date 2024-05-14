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


@pytest.mark.parametrize("backend_fixture", ['pytorch'], indirect=True)
def test_identify_voxels_repeated_zero_ret_empty_list(backend_fixture):
    """Test the identify_voxels_repeated_zero_ret method.
    We test with and with out a volume, and different numbers of microlenses.
    
    Note: The method is only developed with the pytorch backend.
    Note: With a supersampling of one, only a very small of voxels wil be raytraced.
    """
    torch.set_grad_enabled(False)
    ### Test using a 1x1 MLA
    vol_shape = [3, 7, 7]
    optical_info = set_optical_info(vol_shape, 17, 1)
    BF_raytrace = BirefringentRaytraceLFM(
        backend=backend_fixture, optical_info=optical_info
        )
    BF_raytrace.compute_rays_geometry()
    BF_raytrace.store_shifted_vox_indices()
    counts = BF_raytrace._count_vox_raytrace_occurrences(zero_ret_voxels=False)
    
    assert len(counts) == vol_shape[0], f"Expected {vol_shape[0]} voxels raytraced"

    voxel_volume = BirefringentVolume(
        backend=backend_fixture,
        optical_info=optical_info, 
        volume_creation_args={'init_mode': 'single_voxel'}
        )
    [ret_image, _] = BF_raytrace.ray_trace_through_volume(voxel_volume)
    ret_image_np = ret_image.cpu().numpy()
    BF_raytrace.compute_rays_geometry(image=ret_image_np)
    counts = BF_raytrace._count_vox_raytrace_occurrences(zero_ret_voxels=False)
    vox_list = BF_raytrace.identify_voxels_repeated_zero_ret()

    assert len(counts) == vol_shape[0], f"Expected {vol_shape[0]} voxels raytraced with volume"
    assert len(vox_list) == 0, "Expected zero voxels raytraced excluding repeated zero retardance voxels"

    ### Test using a 3x3 MLA
    test_3x3mla = False
    if test_3x3mla:
        vol_shape = [3, 9, 9]
        optical_info = set_optical_info(vol_shape, 17, 3)
        BF_raytrace = BirefringentRaytraceLFM(
            backend=backend_fixture, optical_info=optical_info
            )
        BF_raytrace.compute_rays_geometry()
        BF_raytrace.store_shifted_vox_indices()
        counts = BF_raytrace._count_vox_raytrace_occurrences(zero_ret_voxels=False)
        
        assert len(counts) == 75, "Expected 75 voxels raytraced"

        voxel_volume = BirefringentVolume(
            backend=backend_fixture,
            optical_info=optical_info, 
            volume_creation_args={'init_mode': 'single_voxel'}
            )
        [ret_image, _] = BF_raytrace.ray_trace_through_volume(voxel_volume)
        ret_image_np = ret_image.cpu().numpy()
        BF_raytrace.compute_rays_geometry(image=ret_image_np)
        counts = BF_raytrace._count_vox_raytrace_occurrences(zero_ret_voxels=False)
        vox_list = BF_raytrace.identify_voxels_repeated_zero_ret()

        assert len(counts) == 75, "Expected 75 voxels raytraced with volume"
        vox_list_err_msg = ("Expected 71 voxels raytraced excluding repeated " +
                            f"zero retardance voxels. Got: {vox_list}")
        assert len(vox_list) == 71, vox_list_err_msg

    ### Test using a 5x5 MLA
    vol_shape = [3, 9, 9]
    optical_info = set_optical_info(vol_shape, 17, 5)
    BF_raytrace = BirefringentRaytraceLFM(
        backend=backend_fixture, optical_info=optical_info
        )
    BF_raytrace.compute_rays_geometry()
    BF_raytrace.store_shifted_vox_indices()
    counts = BF_raytrace._count_vox_raytrace_occurrences(zero_ret_voxels=False)
    
    assert len(counts) == 205, "Expected 205 voxels raytraced"

    voxel_volume = BirefringentVolume(
        backend=backend_fixture,
        optical_info=optical_info, 
        volume_creation_args={'init_mode': 'single_voxel'}
        )
    [ret_image, _] = BF_raytrace.ray_trace_through_volume(voxel_volume)
    ret_image_np = ret_image.cpu().numpy()
    BF_raytrace.compute_rays_geometry(image=ret_image_np)
    counts = BF_raytrace._count_vox_raytrace_occurrences(zero_ret_voxels=False)
    vox_list = BF_raytrace.identify_voxels_repeated_zero_ret()

    assert len(counts) == 205, "Expected 205 voxels raytraced with volume"
    err_msg = ("Expected 198 voxels raytraced excluding repeated " +
               f"zero retardance voxels. Got: {vox_list}")
    assert len(vox_list) in {198, 199}, err_msg
    