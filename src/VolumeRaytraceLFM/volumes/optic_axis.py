import torch
import numpy as np


def stay_on_sphere(optic_axis):
    """Function to keep the optic axis on the unit sphere.
    Args:
        optic_axis (torch.Tensor): The optic axis tensor to be normalized.
    """
    with torch.no_grad():
        norms = torch.norm(optic_axis, dim=0)
        zero_norm_mask = norms == 0
        norms[zero_norm_mask] = 1
        optic_axis /= norms
    return optic_axis


def fill_vector_based_on_nonaxial(axis_full, axis_nonaxial):
    """Function to fill the axial component of the optic axis
    with the square root of the remaining components.
    Args:
        axis_full (torch.Tensor): The optic axis tensor to be updated.
        optic_axis_nonaxial (torch.Tensor): The nonaxial components of the optic axis.
    """
    with torch.no_grad():
        axis_full[1:, :] = axis_nonaxial
        square_sum = torch.sum(axis_full[1:, :] ** 2, dim=0)
        axis_full[0, :] = torch.sqrt(1 - square_sum)
        axis_full[0, torch.isnan(axis_full[0, :])] = 0
    return axis_full


def spherical_to_unit_vector(theta, phi):
    """Convert spherical angles to a unit vector.
    Args:
        theta (float): Azimuthal angle in radians (0 <= theta < 2*pi).
        phi (float): Polar angle in radians (0 <= phi <= pi/2).
    Returns:
        np.ndarray: Unit vector [z, y, x] where z >= 0.
    """
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.array([z, y, x])
