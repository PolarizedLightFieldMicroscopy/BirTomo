"""Tests for the Siddon ray-voxel intersection algorithm."""

import numpy as np
import pytest
from VolumeRaytraceLFM.my_siddon import siddon


def brute_force_lengths(start, stop, voxel_size, volume_shape, n_samples=200000):
    """Reference: densely sample the segment and accumulate length per voxel."""
    t = (np.arange(n_samples) + 0.5) / n_samples
    pts = start[None, :] + t[:, None] * (stop - start)[None, :]
    step = np.linalg.norm(stop - start) / n_samples
    idx = np.floor(pts / np.asarray(voxel_size)).astype(int)
    inside = np.all((idx >= 0) & (idx < np.asarray(volume_shape)), axis=1)
    lengths = {}
    for key in map(tuple, idx[inside]):
        lengths[key] = lengths.get(key, 0.0) + step
    return lengths


def siddon_lengths_by_voxel(start, stop, voxel_size, volume_shape):
    voxels, lengths = siddon(start, stop, voxel_size, volume_shape)
    out = {}
    for vox, ell in zip(voxels, lengths):
        key = tuple(int(v) for v in vox)
        out[key] = out.get(key, 0.0) + ell
    return out


@pytest.mark.parametrize(
    "start, stop",
    [
        # Increasing y, decreasing z: last y-plane and first z-plane were dropped
        ([0.0, 10.3, 9.7], [5.0, 12.9, 6.2]),
        # Decreasing y, increasing z
        ([0.0, 12.6, 6.4], [5.0, 9.2, 9.9]),
        # Ray clipped by a lateral face (exits through y = 20)
        ([0.0, 17.5, 10.2], [5.0, 23.0, 11.7]),
        # Ray parallel to the optical axis
        ([0.0, 10.5, 10.5], [5.0, 10.5, 10.5]),
    ],
)
def test_siddon_matches_brute_force(start, stop):
    voxel_size = [1.0, 1.0, 1.0]
    volume_shape = [5, 20, 20]
    start, stop = np.array(start), np.array(stop)
    sid = siddon_lengths_by_voxel(start, stop, voxel_size, volume_shape)
    ref = brute_force_lengths(start, stop, voxel_size, volume_shape)
    assert set(sid) == set(ref), f"voxel sets differ: {set(sid) ^ set(ref)}"
    for key in ref:
        assert sid[key] == pytest.approx(ref[key], abs=1e-3), key


def test_siddon_non_cubic_voxels():
    voxel_size = [2.0, 1.5, 1.5]
    volume_shape = [4, 12, 12]
    start, stop = np.array([0.0, 8.3, 9.1]), np.array([8.0, 3.2, 12.4])
    sid = siddon_lengths_by_voxel(start, stop, voxel_size, volume_shape)
    ref = brute_force_lengths(start, stop, voxel_size, volume_shape)
    assert set(sid) == set(ref)
    for key in ref:
        assert sid[key] == pytest.approx(ref[key], abs=1e-3), key
