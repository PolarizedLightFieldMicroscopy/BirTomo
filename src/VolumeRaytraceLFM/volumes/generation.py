import numpy as np
from math import floor


def generate_single_voxel_volume(
    volume_shape, delta_n=0.01, optic_axis=[1, 0, 0], offset=[0, 0, 0]
):
    # Identity the center of the volume after the shifts
    vox_idx = [
        volume_shape[0] // 2 + offset[0],
        volume_shape[1] // 2 + offset[1],
        volume_shape[2] // 2 + offset[2],
    ]
    vol = np.zeros((4, *volume_shape))
    vol[0, vox_idx[0], vox_idx[1], vox_idx[2]] = delta_n
    vol[1:, vox_idx[0], vox_idx[1], vox_idx[2]] = np.array(optic_axis)
    return vol


def generate_random_volume(
    volume_shape, init_args={"Delta_n_range": [0, 1], "axes_range": [-1, 1]}
):
    np.random.seed(42)
    Delta_n = np.random.uniform(
        init_args["Delta_n_range"][0], init_args["Delta_n_range"][1], volume_shape
    )
    # Random axis
    min_axis = init_args["axes_range"][0]
    max_axis = init_args["axes_range"][1]
    a_0 = np.random.uniform(min_axis, max_axis, volume_shape)
    a_1 = np.random.uniform(min_axis, max_axis, volume_shape)
    a_2 = np.random.uniform(min_axis, max_axis, volume_shape)
    norm_A = np.sqrt(a_0**2 + a_1**2 + a_2**2)
    return np.concatenate(
        (
            np.expand_dims(Delta_n, axis=0),
            np.expand_dims(a_0 / norm_A, axis=0),
            np.expand_dims(a_1 / norm_A, axis=0),
            np.expand_dims(a_2 / norm_A, axis=0),
        ),
        0,
    )


def generate_planes_volume(volume_shape, n_planes=1, z_offset=0, delta_n=0.01):
    vol = np.zeros((4, *volume_shape))
    z_size = volume_shape[0]
    z_ranges = np.linspace(0, z_size - 1, n_planes * 2).astype(int)

    # Set random optic axis
    optic_axis = np.random.uniform(-1, 1, [3, *volume_shape])
    norms = np.linalg.norm(optic_axis, axis=0)
    vol[1:, ...] = optic_axis / norms

    if n_planes == 1:
        # Birefringence
        vol[0, z_size // 2 + z_offset, :, :] = delta_n
        # Axis
        vol[1, z_size // 2 + z_offset, :, :] = 1
        vol[2, z_size // 2 + z_offset, :, :] = 0
        vol[3, z_size // 2 + z_offset, :, :] = 0
        return vol
    random_data = generate_random_volume([n_planes])
    for z_ix in range(0, n_planes):
        vol[:, z_ranges[z_ix * 2] : z_ranges[z_ix * 2 + 1]] = (
            np.expand_dims(random_data[:, z_ix], [1, 2, 3])
            .repeat(1, 1)
            .repeat(volume_shape[1], 2)
            .repeat(volume_shape[2], 3)
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
    vol = np.zeros((4, *volume_shape))
    kk, jj, ii = np.meshgrid(
        np.arange(volume_shape[0]),
        np.arange(volume_shape[1]),
        np.arange(volume_shape[2]),
        indexing="ij",
    )
    # shift to center
    kk = floor(center[0] * volume_shape[0]) - kk.astype(float)
    jj = floor(center[1] * volume_shape[1]) - jj.astype(float)
    ii = floor(center[2] * volume_shape[2]) - ii.astype(float)

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
        inner_mask = np.abs(inner_ellipsoid_border) <= 1
    else:
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
