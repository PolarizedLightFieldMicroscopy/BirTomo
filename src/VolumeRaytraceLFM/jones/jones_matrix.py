import torch
import numpy as np


def vox_ray_ret_azim_numpy(bir, optic_axis, rayDir, ell, wavelength):
    # Azimuth is the angle of the slow axis of retardance.
    # TODO: verify the order of these two components
    azim = np.atan2(np.dot(optic_axis, rayDir[1]), np.dot(optic_axis, rayDir[2]))
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
        "Azimuth angle of index ellipsoid is "
        + f"{np.around(np.rad2deg(azim), decimals=0)} degrees."
    )
    print(
        "Accumulated retardance from index ellipsoid is "
        + f"{np.around(np.rad2deg(ret), decimals=0)} ~ {int(np.rad2deg(ret)) % 360} degrees."
    )


def vox_ray_ret_azim_torch(bir, optic_axis, rayDir, ell, wavelength):
    """Calculate the retardance and azimuth angle for a given
    birefringence, optic axis, and ray direction.

    Args:
        bir (torch.Tensor): Birefringence values. Shape: [num_voxels, intersection_rows]
        optic_axis (torch.Tensor): Optic axis vectors. Shape: [num_voxels, 3, intersection_rows]
        rayDir (torch.Tensor): Ray direction vectors. Shape: [3, num_voxels, 3]
        ell (torch.Tensor): Path lengths. Shape: [num_voxels, intersection_rows]
        wavelength (float): Wavelength of light.

    Returns:
        ret (torch.Tensor): Retardance values. Shape: [num_voxels, intersection_rows]
        azim (torch.Tensor): Azimuth angles. Shape: [num_voxels, intersection_rows]
    """
    pi_tensor = torch.tensor(np.pi, dtype=bir.dtype, device=bir.device)
    # Dot product of optical axis and 3 ray-direction vectors
    OA_dot_rayDir = (rayDir.unsqueeze(2) @ optic_axis).squeeze(2)
    # There is the x2 here because it is not in the jones matrix function
    azim = 2 * torch.atan2(OA_dot_rayDir[1], OA_dot_rayDir[2])
    ret = abs(bir) * (1 - (OA_dot_rayDir[0]) ** 2) * ell * pi_tensor / wavelength
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
    # TODO: update the nonzero_only version now that bir is not 1D
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


def _get_diag_offdiag_jones(ret, azim, precision=torch.float64):
    """
    Args:
        ret (torch.Tensor): Retardance angles.
        azim (torch.Tensor): Azimuth angles.
        precision: torch.float32 or torch.float64
    """
    ret = ret.to(precision)
    azim = azim.to(precision)
    
    exp_ret = torch.polar(torch.tensor(1.0, dtype=precision, device=ret.device), ret)
    exp_azim = torch.polar(torch.tensor(1.0, dtype=precision, device=azim.device), azim)

    diag = exp_ret.real + 1j * exp_azim.real * exp_ret.imag
    offdiag = 1j * exp_azim.imag * exp_ret.imag

    return diag, offdiag


def jones_torch_from_diags(diag, offdiag):
    jones = torch.empty([*diag.shape, 2, 2], dtype=diag.dtype, device=diag.device)
    jones[:, :, 0, 0] = diag
    jones[:, :, 0, 1] = offdiag
    jones[:, :, 1, 0] = offdiag
    jones[:, :, 1, 1] = torch.conj(diag)
    return jones


def jones_torch(ret, azim, precision=torch.float64):
    """Computes the Jones matrix given the retardance and azimuth angles.
    Args:
        ret (torch.Tensor): Retardance angles.
        azim (torch.Tensor): Azimuth angles.
        precision (torch.dtype): Precision of the computation.
    Returns:
        torch.Tensor: The Jones matrices.
                      Shape: [*ret.shape, 2, 2]
    """
    diag, offdiag = _get_diag_offdiag_jones(ret, azim, precision=precision)
    jones = jones_torch_from_diags(diag, offdiag)
    return jones


def jones_torch_nonzeros(ret, azim, precision=torch.float32):
    nonzero_indices = ret != 0
    cos_ret = torch.cos(ret[nonzero_indices].to(dtype=precision))
    sin_ret = torch.sin(ret[nonzero_indices].to(dtype=precision))
    cos_azim = torch.cos(azim[nonzero_indices].to(dtype=precision))
    sin_azim = torch.sin(azim[nonzero_indices].to(dtype=precision))
    offdiag = 1j * sin_azim * sin_ret
    diag1 = cos_ret + 1j * cos_azim * sin_ret
    diag2 = torch.conj(diag1)

    jones = torch.eye(2, dtype=diag1.dtype, device=ret.device).repeat(len(ret), 1, 1)
    jones[nonzero_indices, 0, 0] = diag1
    jones[nonzero_indices, 0, 1] = offdiag
    jones[nonzero_indices, 1, 0] = offdiag
    jones[nonzero_indices, 1, 1] = diag2
    return jones


def calculate_jones_torch(ret, azim, nonzeros_only=False, precision=torch.float64):
    if nonzeros_only:
        # Faster when the number of non-zero elements is large
        return jones_torch_nonzeros(ret, azim)
    else:
        # Faster when the number of non-zero elements is small
        return jones_torch(ret, azim, precision=precision)
