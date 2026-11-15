"""The microlens array must be centered on the volume for both even and odd
numbers of microlenses."""

import numpy as np
import pytest
import torch
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import (
    BirefringentVolume,
    BirefringentRaytraceLFM,
)
from tests.fixtures_optical_info import set_optical_info

# Lateral margin, in voxels, around the microlens array. The rays of the
# 5-voxel-deep volume below reach about 5 voxels sideways from a lenslet.
MARGIN = 6


def _setup(n_micro_lenses, n_voxels_per_ml):
    lateral = n_micro_lenses * n_voxels_per_ml + 2 * MARGIN
    optical_info = set_optical_info([5, lateral, lateral], 16, n_micro_lenses)
    optical_info["n_voxels_per_ml"] = n_voxels_per_ml
    optical_info["aperture_radius_px"] = 8
    return optical_info


@pytest.mark.parametrize("n_micro_lenses", [2, 3, 4])
@pytest.mark.parametrize("n_voxels_per_ml", [1, 2])
def test_lenslet_blocks_are_centered(n_micro_lenses, n_voxels_per_ml):
    """Each lenslet converges on its own block of voxels, and the blocks
    tile the center of the volume."""
    optical_info = _setup(n_micro_lenses, n_voxels_per_ml)
    rays = BirefringentRaytraceLFM(backend=BackEnds.PYTORCH, optical_info=optical_info)
    rays.verbose = False
    rays.compute_rays_geometry()

    focal_z = optical_info["volume_shape"][0] // 2
    hits = np.array([v for ray in rays.ray_vol_colli_indices for v in ray])
    hits = hits[hits[:, 0] == focal_z]
    # Centroid of the focal-plane voxels of the reference lenslet, then shift.
    centroid = hits[:, 1:].mean(axis=0)
    lateral = optical_info["volume_shape"][1]
    start = (lateral - n_micro_lenses * n_voxels_per_ml) // 2
    for k in range(n_micro_lenses):
        ml = k - n_micro_lenses // 2
        offset = rays._calculate_current_offset(ml, ml, n_voxels_per_ml, n_micro_lenses)
        expected = start + n_voxels_per_ml * (k + 0.5) - 0.5
        np.testing.assert_allclose(centroid + offset, [expected, expected], atol=1e-9)


@pytest.mark.parametrize("n_micro_lenses", [2, 3, 4])
@pytest.mark.parametrize("n_voxels_per_ml", [1, 2])
def test_centered_object_gives_symmetric_light_field(n_micro_lenses, n_voxels_per_ml):
    """A shell centered in the volume has a retardance light field that is
    unchanged by a 180 degree rotation, and both forward-projection paths
    agree."""
    torch.set_grad_enabled(False)
    optical_info = _setup(n_micro_lenses, n_voxels_per_ml)
    rays = BirefringentRaytraceLFM(backend=BackEnds.PYTORCH, optical_info=optical_info)
    rays.verbose = False
    rays.compute_rays_geometry()
    radius = n_micro_lenses * n_voxels_per_ml / 2 + 1
    volume = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=optical_info,
        volume_creation_args={
            "init_mode": "ellipsoid",
            "init_args": {"radius": [2.5, radius, radius], "delta_n": 0.02},
        },
    )

    ret_per_lenslet, _ = rays.ray_trace_through_volume(volume)
    ret = ret_per_lenslet.numpy()
    assert ret.max() > 0.01
    np.testing.assert_allclose(ret, ret[::-1, ::-1], atol=1e-6)

    rays.prepare_for_all_rays_at_once()
    ret_all, _ = rays.ray_trace_through_volume(volume, all_rays_at_once=True)
    np.testing.assert_allclose(ret_all.numpy(), ret, atol=1e-6)


def test_rays_beyond_volume_edge_raise():
    """Rays that would leave the volume laterally must raise rather than be
    raveled into a neighboring row of the volume."""
    optical_info = set_optical_info([5, 6, 6], 16, 4)
    optical_info["aperture_radius_px"] = 8
    rays = BirefringentRaytraceLFM(backend=BackEnds.PYTORCH, optical_info=optical_info)
    rays.verbose = False
    rays.compute_rays_geometry()
    with pytest.raises(ValueError, match="increasing the non-axial volume"):
        rays.prepare_for_all_rays_at_once()
