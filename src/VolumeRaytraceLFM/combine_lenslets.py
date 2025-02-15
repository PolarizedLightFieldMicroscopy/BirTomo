import numpy as np
import torch
from VolumeRaytraceLFM.abstract_classes import BackEnds


def gather_voxels_of_rays_pytorch_batch(
    microlens_offset, collision_indices, volume_shape, backend
):
    """In progress: Gathers the shifted voxel indices based on the microlens offset."""
    err_msg = "This function is for PyTorch backend only."
    assert backend == BackEnds.PYTORCH, err_msg

    nested_version = False
    if nested_version:
        collision_indices_tensors = [
            torch.tensor(sublist, dtype=torch.long) for sublist in collision_indices
        ]
        collision_indices_nested_tensor = torch.nested_tensor(collision_indices_tensors)
    else:
        # collision_indices_as_lists = [
        #     [list(item) for item in sublist] for sublist in collision_indices
        # ]
        # max_sublist_length = max(len(sublist) for sublist in collision_indices_as_lists)
        # padded_collision_indices = [
        #     sublist + [(-1, -1, -1)] * (max_sublist_length - len(sublist))
        #     for sublist in collision_indices_as_lists
        # ]
        # collision_indices_tensor = torch.tensor(
        #     padded_collision_indices, dtype=torch.long
        # )
        voxel_indices = pad_and_convert_to_tensor(collision_indices, pad_value=torch.nan)

    # Ensure microlens_offset is a tensor and compatible for broadcasting
    microlens_offset_tensor = torch.tensor(microlens_offset, dtype=torch.long)

    # Dynamically calculate dimensions
    n_offsets = microlens_offset.shape[0] * microlens_offset.shape[1]  # 5x5 => 25
    n_rays = voxel_indices.shape[0]  # 172
    n_segments = voxel_indices.shape[1]  # 7

    # Ensure microlens_offset is compatible with voxel_indices
    microlens_offset = microlens_offset_tensor.to(voxel_indices.device)

    # Reshape microlens_offset to (25, 2), assuming 5x5 microlenses = 25 total offsets
    microlens_offset_reshaped = microlens_offset.view(-1, 2)  # Shape: (25, 2)

    # Expand voxel_indices and microlens_offset to match the final shape
    voxel_indices_expanded = voxel_indices.unsqueeze(0).expand(n_offsets, -1, -1, -1)  # Shape: (25, 172, 7, 3)
    
    # Expand microlens_offset to match (25, 172, 7, 2)
    microlens_offset_expanded = microlens_offset_reshaped.unsqueeze(1).unsqueeze(1).expand(-1, n_rays, n_segments, -1)

    # Apply the microlens offset to the y and z dimensions of the voxel indices
    voxel_indices_shifted = voxel_indices_expanded.clone()
    voxel_indices_shifted[:, :, :, 1:3] += microlens_offset_expanded

    # Reshape voxel_indices_shifted to a 2D tensor of shape (25 * 172 * 7, 3)
    voxel_indices_flat = voxel_indices_shifted.view(-1, 3)  # Shape: (25 * 172 * 7, 3)

    # Apply ravel_index_tensor to the flattened indices
    raveled_indices_flat = ravel_index_tensor(voxel_indices_flat, volume_shape)

    # Reshape raveled_indices_flat back to (25, 172, 7)
    raveled_indices = raveled_indices_flat.view(n_offsets, n_rays, n_segments)

    return raveled_indices


def calculate_offsets_vectorized(n_micro_lenses, n_voxels_per_ml, central_vox_idx):
    """In progress: Calculate the offsets for all microlenses in the array"""
    ml_indices = np.arange(n_micro_lenses) - n_micro_lenses // 2
    jj, ii = np.meshgrid(ml_indices, ml_indices)
    offsets = calculate_all_offsets(n_micro_lenses, n_voxels_per_ml, central_vox_idx)
    # offsets = self.calculate_current_offset(ii.flatten(), jj.flatten(), n_voxels_per_ml, n_micro_lenses)
    jj, ii = np.meshgrid(np.arange(n_micro_lenses), np.arange(n_micro_lenses))
    mla_indices = np.stack((jj.flatten(), ii.flatten()), axis=-1)
    return offsets, mla_indices


def calculate_all_offsets(num_microlenses, num_voxels_per_ml, central_vox_idx):
    """In progress:
    Vectorized computation of offsets for all microlenses in the array.

    Args:
        num_voxels_per_ml (int): The number of voxels per microlens.
        num_microlenses (int): Total number of microlenses in one dimension.

    Returns:
        np.ndarray: Array of offsets for all microlenses.
    """
    # Create arrays for row and column indices for all microlenses
    indices = np.arange(num_microlenses)
    ml_jj_idx, ml_ii_idx = np.meshgrid(indices, indices, indexing="ij")
    n_ml_half = num_microlenses // 2
    ml_ii = ml_ii_idx - n_ml_half
    ml_jj = ml_jj_idx - n_ml_half

    # Scale these indices to voxel space
    scaled_indices = num_voxels_per_ml * np.stack((ml_jj, ml_ii), axis=-1)

    # Compute central offset, assumed to be set as an attribute like self.vox_ctr_idx = [x, y]
    central_offset = np.array(central_vox_idx[1:])  # adjust the index access as needed

    # Calculate half the total voxel span
    half_voxel_span = np.floor(num_voxels_per_ml * num_microlenses / 2.0)

    # Calculate final offsets for all microlenses
    all_offsets = scaled_indices + central_offset - half_voxel_span

    return all_offsets


def calculate_current_offset_tensor(
    row_index, col_index, num_voxels_per_ml, num_microlenses, vox_ctr_idx
):
    """Maps the position of microlenses in an array to the corresponding
    position in the volumetric data, identified by its row and column indices.
    This function calculates the offset to the top corner of the volume in
    front of the current microlens (row_index, col_index), using tensors.

    Args:
        row_index (Tensor): The row index tensor of the current microlenses in
                            the microlens array.
        col_index (Tensor): The column index tensor of the current microlenses
                            in the microlens array.
        num_voxels_per_ml (int): The number of voxels per microlens,
            indicating the size of the voxel area each microlens covers.
        num_microlenses (int): The total number of microlenses in one
                               dimension of the microlens array.
        vox_ctr_idx (list): The central voxel index of the volume.
    Returns:
        Tensor: A tensor representing the calculated offset in the volumetric
                data for the current microlenses.
    """
    row_index = row_index.to(torch.float32)
    col_index = col_index.to(torch.float32)

    num_voxels_per_ml = torch.tensor(num_voxels_per_ml, dtype=torch.float32)
    num_microlenses = torch.tensor(num_microlenses, dtype=torch.float32)

    # Scale row and column indices to voxel space. This is important when using supersampling.
    scaled_indices = torch.stack(
        [num_voxels_per_ml * row_index, num_voxels_per_ml * col_index], dim=-1
    )

    # Convert the central offset to a tensor and ensure it is on the same device as the input tensors
    central_offset = torch.tensor(vox_ctr_idx[1:], dtype=row_index.dtype, device=row_index.device)

    # Compute the midpoint of the total voxel space covered by the microlenses.
    # This will be a scalar, so we use normal Python math functions.
    half_voxel_span = num_voxels_per_ml * num_microlenses // 2

    # Calculate and return the final offset for the current microlens
    return scaled_indices + central_offset - half_voxel_span


def vectorized_offset_calculation(n_micro_lenses, n_voxels_per_ml, vox_ctr_idx):
    """Performs vectorized calculation of the current offsets."""

    # Create grid of indices for ml_ii and ml_jj
    ml_half = n_micro_lenses // 2
    indices = torch.arange(-ml_half, ml_half + 1)  # Create the index range for ml_ii and ml_jj

    # Generate the grid of (ml_ii, ml_jj) indices
    ml_ii_grid, ml_jj_grid = torch.meshgrid(indices, indices, indexing="ij")

    # Flatten the grid into 1D vectors for vectorized calculation
    ml_ii_flat = ml_ii_grid.flatten()
    ml_jj_flat = ml_jj_grid.flatten()

    # Calculate the current offset tensor for all indices at once
    current_offsets = calculate_current_offset_tensor(ml_ii_flat, ml_jj_flat, n_voxels_per_ml, n_micro_lenses, vox_ctr_idx)

    # Optionally, reshape back into the grid if needed
    current_offsets_reshaped = current_offsets.view(n_micro_lenses, n_micro_lenses, -1)

    return current_offsets_reshaped


def pad_and_convert_to_tensor(collision_indices, pad_value=-1):
    """Pads each list of tuples in collision_indices to the maximum
    length and converts it into a PyTorch tensor.

    Args:
        collision_indices (list of lists of tuples): The voxel collision coordinates with varying lengths.
        pad_value (int): The value to pad the shorter lists with.

    Returns:
        torch.Tensor: Padded tensor of collision_indices.
    """
    # Step 1: Determine the maximum length of the inner lists
    max_length = max(len(inner_list) for inner_list in collision_indices)

    # Step 2: Pad each list of tuples to the maximum length and ensure padding is applied to tuples
    padded_lists = []
    for inner_list in collision_indices:
        # Convert tuples to lists and pad each list of tuples to max_length
        padded_inner_list = list(inner_list) + [(pad_value, pad_value, pad_value)] * (max_length - len(inner_list))
        padded_lists.append(padded_inner_list)

    # Step 3: Convert the padded lists to a PyTorch tensor
    collision_indices_tensor = torch.tensor(padded_lists, dtype=torch.float32)

    return collision_indices_tensor


def ravel_index_tensor(x, dims):
    """Convert multi-dimensional indices to a 1D index using PyTorch operations."""
    dims = torch.tensor(dims, dtype=torch.long, device=x.device)
    c = torch.cumprod(
        torch.cat((torch.tensor([1], device=x.device), dims.flip(0))), 0
    )[:-1].flip(0)
    if x.ndim == 1:
        return torch.sum(c * x)  # torch.dot(c, x)
    elif x.ndim == 2:
        return torch.sum(c * x, dim=1)
    else:
        raise ValueError("Input tensor x must be 1D or 2D")
