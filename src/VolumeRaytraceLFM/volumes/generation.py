"""
This module contains functions for generating numpy arrays that represent
the birefringence properties of a volume.
"""
import numpy as np


def generate_single_voxel_volume(
        volume_shape: list[int, int, int],
        delta_n: float = 0.01,
        optic_axis: list = [1, 0, 0],
        offset: list[int, int, int] = [0, 0, 0],
):
    # Identity the center of the volume after the shifts
    vox_idx = [s // 2 + o for s, o in zip(volume_shape, offset)]
    # Create a volume of all zeros.
    vol = np.zeros((4,) + tuple(volume_shape))
    # Set the birefringence and optic axis
    vol[0, vox_idx[0], vox_idx[1], vox_idx[2]] = delta_n
    vol[1:, vox_idx[0], vox_idx[1], vox_idx[2]] = np.array(optic_axis)
    return vol


def generate_random_volume(
    volume_shape: list[int, int, int],
    init_args: dict = {"Delta_n_range": [0, 1], "axes_range": [-1, 1]},
):
    """Generates a random volume."""
    np.random.seed(42)
    Delta_n = np.random.uniform(*init_args["Delta_n_range"], volume_shape)
    axes = [
        np.random.uniform(*init_args["axes_range"], volume_shape) for _ in range(3)
    ]
    norm_A = np.linalg.norm(axes, axis=0)
    return np.concatenate(
        [np.expand_dims(Delta_n, axis=0)]
        + [np.expand_dims(a / norm_A, axis=0) for a in axes],
        axis=0,
    )


def generate_planes_volume(
        volume_shape: list[int, int, int],
        n_planes: int = 1,
        z_offset: int = 0,
        delta_n: float = 0.01,
    ):
    vol = np.zeros((4, *volume_shape))
    z_size = volume_shape[0]
    z_ranges = np.linspace(0, z_size - 1, n_planes * 2).astype(int)
    optic_axis = np.random.uniform(-1, 1, (3, *volume_shape))
    vol[1:, ...] = optic_axis / np.linalg.norm(optic_axis, axis=0)

    if n_planes == 1:
        # Birefringence
        vol[0, z_size // 2 + z_offset, :, :] = delta_n
        # Axis
        vol[1, z_size // 2 + z_offset, :, :] = 1
        vol[2, z_size // 2 + z_offset, :, :] = 0
        vol[3, z_size // 2 + z_offset, :, :] = 0
        return vol

    random_data = generate_random_volume([n_planes])
    for z_ix in range(n_planes):
        slice_range = z_ranges[z_ix * 2 : z_ix * 2 + 1]
        expanded_data = np.expand_dims(random_data[:, z_ix], axis=(1, 2, 3))
        vol[:, slice_range] = expanded_data.repeat(volume_shape[1], axis=2).repeat(
            volume_shape[2], axis=3
        )
    return vol


def generate_ellipsoid_volume(
    volume_shape, center=[0.5, 0.5, 0.5], radius=[10, 10, 10], alpha=1, delta_n=0.01
):
    """Creates an ellipsoid with optical axis normal to the ellipsoid surface.
    Args:
        center [3]: [cz,cy,cx] from 0 to 1 where 0.5 is the center of the volume_shape.
        radius [3]: in voxels, the radius in z,y,x for this ellipsoid.
        alpha (float): Border thickness.
        delta_n (float): Delta_n value of birefringence in the volume
    Returns:
        vol (np.array): 4D array where the first dimension represents the
            birefringence and optic axis properties, and the last three
            dims represents the 3D spatial locations.
    """
    # Originally grabbed from https://math.stackexchange.com/questions/2931909/normal-of-a-point-on-the-surface-of-an-ellipsoid,
    #   then modified to do the subtraction of two ellipsoids instead.
    vol = np.zeros((4,) + tuple(volume_shape))
    kk, jj, ii = np.meshgrid(
        np.arange(volume_shape[0]),
        np.arange(volume_shape[1]),
        np.arange(volume_shape[2]),
        indexing="ij",
    )
    # shift to center
    kk = (center[0] * (volume_shape[0]-1)) - kk.astype(float)
    jj = (center[1] * (volume_shape[1]-1)) - jj.astype(float)
    ii = (center[2] * (volume_shape[2]-1)) - ii.astype(float)

    # DEBUG: checking the indices
    # np.argwhere(ellipsoid_border == np.min(ellipsoid_border))
    # plt.imshow(ellipsoid_border_mask[int(volume_shape[0] / 2),:,:])
    ellipsoid_border = (
        (kk**2) / (radius[0] ** 2)
        + (jj**2) / (radius[1] ** 2)
        + (ii**2) / (radius[2] ** 2)
    )
    hollow_inner = True
    if hollow_inner:
        ellipsoid_border_mask = np.abs(ellipsoid_border) <= 1
        # The inner radius could also be defined as a scaled version of the outer radius.
        # inner_radius = [0.9 * r for r in radius]
        inner_radius = [r - alpha for r in radius]
        inner_ellipsoid_border = (
            (kk**2) / (inner_radius[0] ** 2)
            + (jj**2) / (inner_radius[1] ** 2)
            + (ii**2) / (inner_radius[2] ** 2)
        )
        # using < so that inverse is >=
        inner_mask = np.abs(inner_ellipsoid_border) <= 1
    else:
        # This line feels wierd and maybe should not have the -alpha
        ellipsoid_border_mask = np.abs(ellipsoid_border - alpha) <= 1

    vol[0, ...] = ellipsoid_border_mask.astype(float)
    # Compute normals
    kk_normal = 2 * kk / radius[0]
    jj_normal = 2 * jj / radius[1]
    ii_normal = 2 * ii / radius[2]
    norm_factor = np.sqrt(kk_normal**2 + jj_normal**2 + ii_normal**2)
    # Avoid division by zero
    norm_factor[norm_factor == 0] = 1
    vol[1, ...] = (kk_normal / norm_factor) * vol[0, ...]
    vol[2, ...] = (jj_normal / norm_factor) * vol[0, ...]
    vol[3, ...] = (ii_normal / norm_factor) * vol[0, ...]
    vol[0, ...] *= delta_n
    # vol = vol.permute(0,2,1,3)
    if hollow_inner:
        # Hollowing out the ellipsoid
        combined_mask = np.logical_and(ellipsoid_border_mask, ~inner_mask)
        vol[0, ...] = vol[0, ...] * combined_mask.astype(float)
    return vol
