import torch
import numpy as np


def calc_theta(jones):
    # jones is a batch of 2x2 matrices
    a = jones[:, 0, 0]
    # Clamp the real part to the valid range of acos to prevent NaNs
    # Note: bounds of -1 and 1 cause NaNs in backward pass
    upper_limit = torch.nextafter(torch.tensor(1.0), torch.tensor(-np.inf))
    lower_limit = torch.nextafter(torch.tensor(-1.0), torch.tensor(np.inf))
    # theta = torch.acos(torch.clamp(a.real, -0.999999, 0.999999))
    theta = torch.acos(torch.clamp(a.real, lower_limit, upper_limit))
    return theta


def calc_theta_single(jones):
    # jones is a single 2x2 matrix
    a = jones[0, 0]
    upper_limit = torch.nextafter(torch.tensor(1.0), torch.tensor(-np.inf))
    lower_limit = torch.nextafter(torch.tensor(-1.0), torch.tensor(np.inf))
    # theta = torch.acos(torch.clamp(a.real, -0.999999, 0.999999))
    theta = torch.acos(torch.clamp(a.real, lower_limit, upper_limit))
    return theta


def eigenvalues(jones):
    x = torch.linalg.eigvals(jones)
    return x


def eigenvalues_su2(jones):
    theta = calc_theta(jones)
    x = torch.stack([torch.exp(1j * theta), torch.exp(-1j * theta)], dim=-1)
    return x


def retardance_from_jones(jones, su2_method=False):
    if su2_method:
        x = eigenvalues_su2(jones)
    else:
        x = eigenvalues(jones)
    retardance = (torch.angle(x[:, 1]) - torch.angle(x[:, 0])).abs()
    return retardance


def retardance_from_jones_single(jones, su2_method=False):
    x = torch.linalg.eigvals(jones)
    retardance = (torch.angle(x[1]) - torch.angle(x[0])).abs()
    return retardance


def retardance_from_su2(jones):
    theta = calc_theta(jones)
    retardance = 2 * theta.abs()
    return retardance


def retardance_from_su2_single(jones):
    theta = calc_theta_single(jones)
    retardance = 2 * theta.abs()
    return retardance


def retardance_from_jones_numpy(jones, su2_method=False):
    e1, e2 = np.linalg.eigvals(jones)
    phase_diff = np.angle(e1) - np.angle(e2)
    retardance = np.abs(phase_diff)
    return retardance


def retardance_from_su2_numpy(jones):
    a = jones[0, 0]
    upper_limit = np.nextafter(1, -np.inf)
    lower_limit = np.nextafter(-1, np.inf)
    theta = np.arccos(np.clip(np.real(a), lower_limit, upper_limit))
    # theta = np.arccos(np.clip(np.real(a), -0.999999, 0.999999))
    retardance = 2 * np.abs(theta)
    return retardance


def azimuth_from_jones_numpy(jones):
    diag_sum = jones[0, 0] + jones[1, 1]
    diag_diff = jones[1, 1] - jones[0, 0]
    off_diag_sum = jones[0, 1] + jones[1, 0]
    a = np.imag(diag_diff / diag_sum)
    b = np.imag(off_diag_sum / diag_sum)
    if np.isclose(np.abs(a), 0.0) and np.isclose(np.abs(b), 0.0):
        azimuth = np.pi / 2
    else:
        azimuth = np.arctan2(a, b) / 2 + np.pi / 2
    # if np.isclose(azimuth,np.pi):
    #     azimuth = 0.0
    return azimuth


def azimuth_from_jones_torch(jones):
    if jones.ndim == 2:
        # jones is a single 2x2 matrix
        diag_sum = jones[0, 0] + jones[1, 1]
        diag_diff = jones[1, 1] - jones[0, 0]
        off_diag_sum = jones[0, 1] + jones[1, 0]     
    elif jones.ndim == 3:
        # jones is a batch of 2x2 matrices           
        diag_sum = (jones[:, 0, 0] + jones[:, 1, 1])
        diag_diff = (jones[:, 1, 1] - jones[: ,0, 0])
        off_diag_sum = jones[:, 0, 1] + jones[:, 1, 0]

    a = (diag_diff / diag_sum).imag
    b = (off_diag_sum / diag_sum).imag

    # atan2 with zero entries causes nan in backward, so let's filter them out

    # Intermediate variables for zero tensor
    zero_a = torch.tensor(0.0, dtype=a.dtype, device=a.device)
    zero_b = torch.tensor(0.0, dtype=b.dtype, device=b.device)
    zero_for_a = torch.zeros([1], dtype=a.dtype, device=a.device)
    zero_for_b = torch.zeros([1], dtype=b.dtype, device=b.device)

    # Check if a and b are scalar values (zero-dimensional)
    if a.ndim == 0 and b.ndim == 0:
        # Handle the scalar case
        azimuth = torch.pi / 2.0
        if not torch.isclose(a, zero_a) or not torch.isclose(b, zero_b):
            azimuth = torch.arctan2(a, b) / 2.0 + torch.pi / 2.0
    else:
        # Handle the non-scalar case
        azimuth = torch.zeros_like(a)
        close_to_zero_a = torch.isclose(a, zero_for_a)
        close_to_zero_b = torch.isclose(b, zero_for_b)
        zero_a_b = close_to_zero_a.bitwise_and(close_to_zero_b)
        azimuth[~zero_a_b] = torch.arctan2(a[~zero_a_b], b[~zero_a_b]) / 2.0 + torch.pi / 2.0
        azimuth[zero_a_b] = torch.pi / 2.0

    # TODO: if output azimuth is pi, make it 0 and vice-versa (arctan2 bug)
    # zero_index = torch.isclose(azimuth, torch.zeros([1]), atol=1e-5)
    # pi_index = torch.isclose(azimuth, torch.tensor(torch.pi), atol=1e-5)
    # azimuth[zero_index] = torch.pi
    # azimuth[pi_index] = 0
    return azimuth
