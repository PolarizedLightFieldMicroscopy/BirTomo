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
        collision_indices_as_lists = [
            [list(item) for item in sublist] for sublist in collision_indices
        ]
        max_sublist_length = max(len(sublist) for sublist in collision_indices_as_lists)
        padded_collision_indices = [
            sublist + [(-1, -1, -1)] * (max_sublist_length - len(sublist))
            for sublist in collision_indices_as_lists
        ]
        collision_indices_tensor = torch.tensor(
            padded_collision_indices, dtype=torch.long
        )

    # Ensure microlens_offset is a tensor and compatible for broadcasting
    microlens_offset_tensor = torch.tensor(microlens_offset, dtype=torch.long)

    # Calculate new indices by adding microlens offset to y and z dimensions
    # Assuming collision_indices_tensor is of shape [N, V, 3]
    shifted_indices = collision_indices_tensor.unsqueeze(0).repeat(
        microlens_offset_tensor.size(0) * microlens_offset_tensor.size(1), 1, 1, 1
    )

    microlens_offset_tensor = microlens_offset_tensor.unsqueeze(2).unsqueeze(3)
    microlens_offset_tensor = microlens_offset_tensor.reshape(-1, 1, 1, 2)

    # TODO: make shifted_indices large enough to contain info for all microlenses
    shifted_indices[..., 1:] += microlens_offset_tensor

    # Calculate 1D indices for the volume shape
    flat_indices = (
        shifted_indices[..., 0] * volume_shape[1] * volume_shape[2]
        + shifted_indices[..., 1] * volume_shape[2]
        + shifted_indices[..., 2]
    )

    # Convert flat_indices to a list of lists
    return flat_indices.tolist()


def calculate_offsets_vectorized(n_micro_lenses, n_voxels_per_ml, central_vox_idx):
    """In progress: Calculate the offsets for all microlenses in the array"""
    ml_indices = np.arange(n_micro_lenses) - n_micro_lenses // 2
    jj, ii = np.meshgrid(ml_indices, ml_indices)
    offsets = calculate_all_offsets(n_micro_lenses, n_voxels_per_ml, central_vox_idx)
    # offsets = self.calculate_current_offset(ii.flatten(), jj.flatten(), n_voxels_per_ml, n_micro_lenses)
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
    row_indices, col_indices = np.meshgrid(indices, indices, indexing="ij")

    # Scale these indices to voxel space
    scaled_indices = num_voxels_per_ml * np.stack((row_indices, col_indices), axis=-1)

    # Compute central offset, assumed to be set as an attribute like self.vox_ctr_idx = [x, y]
    central_offset = np.array(central_vox_idx[1:])  # adjust the index access as needed

    # Calculate half the total voxel span
    half_voxel_span = np.floor(num_voxels_per_ml * num_microlenses / 2.0)

    # Calculate final offsets for all microlenses
    all_offsets = scaled_indices + central_offset - half_voxel_span

    return all_offsets
