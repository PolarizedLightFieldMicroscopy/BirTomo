import torch
import numpy as np


def calc_theta(jones, clamp_eps=True):
    # jones is a batch of 2x2 matrices
    jones = jones.to(torch.complex128)
    a = jones[:, 0, 0]
    device = jones.device
    # Clamp the real part to the valid range of acos to prevent NaNs
    # Note: bounds of -1 and 1 cause NaNs in backward pass
    if clamp_eps:
        upper_limit = torch.nextafter(
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(-np.inf, dtype=torch.float64),
        ).to(device)
        lower_limit = torch.nextafter(
            torch.tensor(-1.0, dtype=torch.float64),
            torch.tensor(np.inf, dtype=torch.float64),
        ).to(device)
    else:
        upper_limit = torch.tensor(1.0, dtype=torch.float64).to(device)
        lower_limit = torch.tensor(-1.0, dtype=torch.float64).to(device)
    theta = torch.acos(torch.clamp(a.real, lower_limit, upper_limit))
    return theta


def calc_theta_single(jones):
    # jones is a single 2x2 matrix
    a = jones[0, 0].to(torch.complex128)
    upper_limit = torch.nextafter(
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(-np.inf, dtype=torch.float64),
    )
    lower_limit = torch.nextafter(
        torch.tensor(-1.0, dtype=torch.float64),
        torch.tensor(np.inf, dtype=torch.float64),
    )
    theta = torch.acos(torch.clamp(a.real, lower_limit, upper_limit))
    return theta


def eigenvalues(jones):
    x = torch.linalg.eigvals(jones)
    return x


def eigenvalues_su2(jones, clamp_eps=True):
    theta = calc_theta(jones, clamp_eps)
    x = torch.stack([torch.exp(1j * theta), torch.exp(-1j * theta)], dim=-1)
    return x


def retardance_from_jones(jones, su2_method=False):
    if su2_method:
        x = eigenvalues_su2(jones)
    else:
        x = eigenvalues(jones)
    retardance = (torch.angle(x[:, 1]) - torch.angle(x[:, 0])).abs().to(torch.float64)
    return retardance


def retardance_from_jones_single(jones, su2_method=False):
    x = torch.linalg.eigvals(jones)
    retardance = (torch.angle(x[1]) - torch.angle(x[0])).abs()
    return retardance


def retardance_from_su2(jones, clamp_eps=True):
    theta = calc_theta(jones, clamp_eps)
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
    upper_limit = np.nextafter(np.float64(1.0), np.float64(-np.inf))
    lower_limit = np.nextafter(np.float64(-1.0), np.float64(np.inf))
    theta = np.arccos(np.clip(np.real(a), lower_limit, upper_limit))
    retardance = 2 * np.abs(theta)
    return retardance


def azimuth_from_jones_numpy(jones, simple=True):
    j11 = jones[0, 0]
    j12 = jones[0, 1]
    imag_j11 = np.imag(j11)
    imag_j12 = np.imag(j12)
    azimuth = 0.5 * np.atan2(imag_j11, imag_j12) - np.pi / 4.0
    azimuth = np.remainder(azimuth, np.pi)
    if np.isclose(np.abs(imag_j11), 0.0) and np.isclose(np.abs(imag_j12), 0.0):
        azimuth = 0.0
    return azimuth


def azimuth_from_jones_torch(jones):
    """Compute the azimuth angle from a Jones matrix.
    Note: possible bug in atan2 when where pi and 0 are switched"""
    if jones.ndim == 3:
        # jones is a batch of 2x2 matrices
        j11 = jones[:, 0, 0]
        j12 = jones[:, 0, 1]
    elif jones.ndim == 2:
        # jones is a single 2x2 matrix
        j11 = jones[0, 0]
        j12 = jones[0, 1]
    else:
        raise ValueError("Invalid input shape")
    imag_j11 = torch.imag(j11)
    imag_j12 = torch.imag(j12)
    azimuth = torch.zeros_like(imag_j11, dtype=imag_j11.dtype)
    # Create a mask where both imaginary parts are not close to zero
    non_zero_mask = ~(
        torch.isclose(imag_j11, torch.zeros_like(imag_j11))
        & torch.isclose(imag_j12, torch.zeros_like(imag_j12))
    )
    # Compute azimuth for non-zero mask elements
    azimuth_non_zero = (
        0.5 * torch.atan2(imag_j11[non_zero_mask], imag_j12[non_zero_mask]) - torch.pi / 4.0
    )
    # Ensure the azimuth is within the range [0, pi)
    azimuth_non_zero = torch.remainder(azimuth_non_zero, torch.pi)
    azimuth[non_zero_mask] = azimuth_non_zero
    return azimuth
