import numpy as np
import torch

def cartesian_to_spherical_numpy(arr):
    """
    Convert Cartesian coordinates to spherical coordinates.
    
    Args:
        arr (np.ndarray): array of shape (3, ...) where the
            first dimension contains Cartesian coordinates (x, y, z).
        
    Returns:
        np.ndarray: array of shape (3, ...) where the first dimension
            contains spherical coordinates (r, phi, theta).
    """
    x, y, z = arr[0, ...], arr[1, ...], arr[2, ...]
    r = np.sqrt(x**2 + y**2 + z**2)
    # Avoid division by zero and ensure the argument of arccos is within [-1, 1]
    eps = np.finfo(arr.dtype).eps
    r_safe = np.maximum(r, eps)  # Use r_safe to avoid division by zero
    z_r_safe = np.clip(z / r_safe, -1, 1)  # Clip values to avoid invalid input to arccos
    phi = np.arccos(z_r_safe)
    theta = np.arctan2(y, x)
    return np.array([r, phi, theta])


def spherical_to_cartesian_numpy(arr):
    """
    Convert spherical coordinates to Cartesian coordinates.
    
    Args:
        arr (np.ndarray): array of shape (3, ...) where the first
            dimension contains spherical coordinates (r, phi, theta).
        
    Returns:
        np.ndarray: array of shape (3, ...) where the first
            dimension contains Cartesian coordinates (x, y, z).
    """
    r, phi, theta = arr[0, ...], arr[1, ...], arr[2, ...]
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z])


def cartesian_to_spherical_torch(tensor):
    """
    Convert Cartesian coordinates to spherical coordinates using PyTorch.
    
    Args:
        tensor (torch.Tensor): A tensor of shape (3, ...) where the
            first dimension contains Cartesian coordinates (x, y, z).
        
    Returns:
        torch.Tensor: A tensor of shape (3, ...) where the first
            dimension contains spherical coordinates (r, phi, theta).
    """
    x, y, z = tensor[0, ...], tensor[1, ...], tensor[2, ...]
    r = torch.sqrt(x**2 + y**2 + z**2)
    without_mask = False
    if without_mask:
        z_r = torch.clamp(z / r, -1, 1)
        phi = torch.acos(z_r)
    else:
        # Use a mask to identify locations with r = 0 to avoid division by zero
        near_zero_radius_mask = r == 0
        # Temporarily set r to 1 where it is 0 to avoid division by zero in these cases
        r_safe = torch.where(near_zero_radius_mask, torch.tensor(1.0, device=tensor.device, dtype=tensor.dtype), r)
        z_r = torch.clamp(z / r_safe, -1, 1)
        phi = torch.acos(z_r)
        # Optionally, set phi to 0 (or any arbitrary value) where r is 0, as the angle is undefined
        phi = torch.where(near_zero_radius_mask,
                          torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype), phi)
    theta = torch.atan2(y, x)
    return torch.stack([r, phi, theta], dim=0)


def spherical_to_cartesian_torch(tensor):
    """
    Convert spherical coordinates to Cartesian coordinates using PyTorch.
    
    Args:
        tensor (torch.Tensor): A tensor of shape (3, ...) where the
            first dimension contains spherical coordinates (r, phi, theta).
        
    Returns:
        torch.Tensor: A tensor of shape (3, ...) where the first
            dimension contains Cartesian coordinates (x, y, z).
    """
    r, phi, theta = tensor[0, ...], tensor[1, ...], tensor[2, ...]
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)
    return torch.stack([x, y, z], dim=0)
