import torch
import numpy as np


def vox_ray_ret_azim_numpy(bir, optic_axis, rayDir, ell, wavelength):
    # Azimuth is the angle of the slow axis of retardance.
    # TODO: verify the order of these two components
    azim = np.arctan2(np.dot(optic_axis, rayDir[1]), np.dot(optic_axis, rayDir[2]))
    azim = 0 if bir == 0 else (azim + np.pi / 2 if bir < 0 else azim)
    # proj_along_ray = np.dot(optic_axis, rayDir[0])
    ret = (
        abs(bir)
        * (1 - np.dot(optic_axis, rayDir[0]) ** 2)
        * 2
        * ell
        * np.pi
        / wavelength
    )
    return ret, azim


def print_ret_azim_numpy(ret, azim):
    print(
        f"Azimuth angle of index ellipsoid is "
        + f"{np.around(np.rad2deg(azim), decimals=0)} degrees."
    )
    print(
        f"Accumulated retardance from index ellipsoid is "
        + f"{np.around(np.rad2deg(ret), decimals=0)} ~ {int(np.rad2deg(ret)) % 360} degrees."
    )


def vox_ray_ret_azim_torch(bir, optic_axis, rayDir, ell, wavelength):
    pi_tensor = torch.tensor(np.pi, device=bir.device, dtype=bir.dtype)
    # Dot product of optical axis and 3 ray-direction vectors
    OA_dot_rayDir = torch.linalg.vecdot(optic_axis, rayDir)
    # TODO: verify x2 should be mult by the azimuth angle
    azim = 2 * torch.arctan2(OA_dot_rayDir[1, :], OA_dot_rayDir[2, :])
    ret = abs(bir) * (1 - (OA_dot_rayDir[0, :]) ** 2) * ell * pi_tensor / wavelength
    neg_delta_mask = bir < 0
    # TODO: check how the gradients are affected--might be a discontinuity
    azim[neg_delta_mask] += pi_tensor
    return ret, azim


def normalized_projection_torch(optic_axis, rayDir):
    """Useful for the retardance calculuation if the
    optic axis is not normalized."""
    OA_dot_rayDir = torch.linalg.vecdot(optic_axis, rayDir)
    normAxis = torch.linalg.norm(optic_axis, axis=1)
    proj_along_ray = torch.full_like(OA_dot_rayDir[0, :], fill_value=1)
    proj_along_ray[normAxis != 0] = (
        OA_dot_rayDir[0, :][normAxis != 0] / normAxis[normAxis != 0]
    )
    return proj_along_ray


def calculate_vox_ray_ret_azim_torch(
    bir, optic_axis, rayDir, ell, wavelength, nonzeros_only=False
):
    if nonzeros_only:
        # Faster when the number of non-zero elements is large
        nonzero_indices = bir.nonzero()
        ret = torch.zeros_like(bir, device=bir.device, dtype=bir.dtype)
        azim = torch.zeros_like(bir, device=bir.device, dtype=bir.dtype)
        if nonzero_indices.numel() > 0:
            # Filter data for non-zero Delta_n
            nonzero_bir = bir[nonzero_indices]
            nonzero_optic_axis = optic_axis[nonzero_indices, :]
            nonzero_rayDir = rayDir[:, nonzero_indices, :]
            nonzero_ell = ell[nonzero_indices]

            ret_nonzeros, azim_nonzeros = vox_ray_ret_azim_torch(
                nonzero_bir, nonzero_optic_axis, nonzero_rayDir, nonzero_ell, wavelength
            )

            ret[nonzero_indices] = ret_nonzeros
            azim[nonzero_indices] = azim_nonzeros
        return ret, azim
    else:
        # Faster when the number of non-zero elements is small
        return vox_ray_ret_azim_torch(bir, optic_axis, rayDir, ell, wavelength)


def jones_torch(ret, azim):
    cos_ret = torch.cos(ret)
    sin_ret = torch.sin(ret)
    cos_azim = torch.cos(azim)
    sin_azim = torch.sin(azim)
    offdiag = 1j * sin_azim * sin_ret
    diag1 = cos_ret + 1j * cos_azim * sin_ret
    diag2 = torch.conj(diag1)

    jones = torch.empty([len(ret), 2, 2], dtype=torch.complex64, device=ret.device)
    jones[:, 0, 0] = diag1
    jones[:, 0, 1] = offdiag
    jones[:, 1, 0] = offdiag
    jones[:, 1, 1] = diag2
    return jones


def jones_torch_nonzeros(ret, azim):
    nonzero_indices = ret != 0
    cos_ret = torch.cos(ret[nonzero_indices].to(dtype=torch.float32))
    sin_ret = torch.sin(ret[nonzero_indices].to(dtype=torch.float32))
    cos_azim = torch.cos(azim[nonzero_indices].to(dtype=torch.float32))
    sin_azim = torch.sin(azim[nonzero_indices].to(dtype=torch.float32))
    offdiag = 1j * sin_azim * sin_ret
    diag1 = cos_ret + 1j * cos_azim * sin_ret
    diag2 = torch.conj(diag1)

    jones = torch.eye(2, dtype=diag1.dtype, device=ret.device).repeat(len(ret), 1, 1)
    jones[nonzero_indices, 0, 0] = diag1
    jones[nonzero_indices, 0, 1] = offdiag
    jones[nonzero_indices, 1, 0] = offdiag
    jones[nonzero_indices, 1, 1] = diag2
    return jones


def calculate_jones_torch(ret, azim, nonzeros_only=False):
    if nonzeros_only:
        # Faster when the number of non-zero elements is large
        return jones_torch_nonzeros(ret, azim)
    else:
        # Faster when the number of non-zero elements is small
        return jones_torch(ret, azim)
