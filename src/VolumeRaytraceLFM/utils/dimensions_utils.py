import torch
import copy
from math import floor


def get_region_of_ones_shape(mask):
    """
    Computes the shape of the smallest bounding box that contains all the ones in the input mask.
    Args:
        mask (torch.Tensor): binary mask tensor.
    Returns:
        shape (torch.Tensor): shape of the smallest bounding box that contains all the ones in the input mask.
    """
    indices = torch.nonzero(mask)
    if indices.numel() == 0:
        raise ValueError("Mask contains no ones.")
    min_indices = indices.min(dim=0)[0]
    max_indices = indices.max(dim=0)[0]
    shape = max_indices - min_indices + 1
    return shape


def crop_3d_tensor(tensor, new_shape):
    """
    Crops a 3D tensor to a specified new shape, keeping the central part of the original tensor.
    """
    D, H, W = tensor.shape
    new_D, new_H, new_W = new_shape
    start_D = (D - new_D) // 2
    end_D = start_D + new_D
    start_H = (H - new_H) // 2
    end_H = start_H + new_H
    start_W = (W - new_W) // 2
    end_W = start_W + new_W
    return tensor[start_D:end_D, start_H:end_H, start_W:end_W]


def reshape_crop_and_flatten_parameter(flattened_param, original_shape, new_shape):
    # Reshape the flattened parameter
    reshaped_param = flattened_param.view(original_shape)

    # Crop the tensor
    *_, D, H, W = original_shape
    new_D, new_H, new_W = new_shape
    start_D = (D - new_D) // 2
    end_D = start_D + new_D
    start_H = (H - new_H) // 2
    end_H = start_H + new_H
    start_W = (W - new_W) // 2
    end_W = start_W + new_W
    cropped_tensor = reshaped_param[..., start_D:end_D, start_H:end_H, start_W:end_W]

    # Flatten and convert back to a Parameter
    cropped_flattened_parameter = torch.nn.Parameter(cropped_tensor.flatten())
    return cropped_flattened_parameter


def reshape_and_crop(flattened_param, original_shape, new_shape):
    """
    Reshapes a flattened tensor to its original shape and crops it to a new shape.
    Args:
        flattened_param (torch.Tensor): Flattened tensor to be reshaped and cropped.
        original_shape (list): Original shape of the tensor before flattening.
        new_shape (list): Desired shape of the cropped tensor.
    Returns:
        torch.Tensor: Cropped tensor with the desired shape.
    """
    # Reshape the flattened parameter
    reshaped_param = flattened_param.view(original_shape)
    # Crop the tensor
    *_, D, H, W = original_shape
    new_D, new_H, new_W = new_shape
    start_D = (D - new_D) // 2
    end_D = start_D + new_D
    start_H = (H - new_H) // 2
    end_H = start_H + new_H
    start_W = (W - new_W) // 2
    end_W = start_W + new_W
    cropped_tensor = reshaped_param[..., start_D:end_D, start_H:end_H, start_W:end_W]
    return cropped_tensor


def store_as_pytorch_parameter(tensor, var_type: str):
    """
    Converts a tensor to a PyTorch parameter and flattens appropriately.
    Note: possibly .type(torch.get_default_dtype()) is needed.
    """
    if var_type == "scalar":
        parameter = torch.nn.Parameter(tensor.flatten())
    elif var_type == "vector":
        parameter = torch.nn.Parameter(tensor.reshape(3, -1))
    return parameter


def upscale_birefringence(bir_array, upscale_factor):
    """Increase resolution of birefringence tensor by
    tripling (if upscale_factor=3) each element along each dimension."""
    # Upscale along dimension 0 (depth)
    tensor_d0 = bir_array.repeat_interleave(upscale_factor, dim=0)
    # Upscale along dimension 1 (rows)
    tensor_d1 = tensor_d0.repeat_interleave(upscale_factor, dim=1)
    # Upscale along dimension 2 (columns)
    high_res_tensor = tensor_d1.repeat_interleave(upscale_factor, dim=2)
    return high_res_tensor


def upscale_optic_axis(optic_axis_array, upscale_factor):
    """Increase resolution of optic axis tensor by
    tripling (if upscale_factor=3) each element along each dimension."""
    # Upscale along dimension 1 (depth)
    tensor_d0 = optic_axis_array.repeat_interleave(upscale_factor, dim=1)
    # Upscale along dimension 2 (rows)
    tensor_d1 = tensor_d0.repeat_interleave(upscale_factor, dim=2)
    # Upscale along dimension 3 (columns)
    high_res_tensor = tensor_d1.repeat_interleave(upscale_factor, dim=3)
    return high_res_tensor


def upscale_voxel_resolution(volume, upscale_factor):
    """
    Increase resolution of birefringence and optic axis tensors by
    tripling (if upscale_factor=3) each element along each dimension.
    Args:
        volume (BirefringentVolume): Volume to be upscaled.
        upscale_factor (int): The factor by which each tensor element
                            is repeated in all dimensions.
    Returns:
        BirefringentVolume: Updated volume with upscaled birefringence
                            and optic axis.
    """
    vol = copy.deepcopy(volume)
    vol_shape_og = list(vol.optical_info["volume_shape"])
    bir_og = vol.get_delta_n()
    optic_axis_og = vol.get_optic_axis()
    bir_upscaled = upscale_birefringence(bir_og, upscale_factor)
    optic_axis_upscaled = upscale_optic_axis(optic_axis_og, upscale_factor)
    vol.Delta_n = store_as_pytorch_parameter(bir_upscaled, "scalar")
    vol.optic_axis = store_as_pytorch_parameter(optic_axis_upscaled, "vector")
    vol.optical_info["volume_shape"] = bir_upscaled.shape
    vol.optical_info["n_voxels_per_ml"] *= upscale_factor
    vol.optical_info["voxel_size_um"] = [
        x / upscale_factor for x in vol.optical_info["voxel_size_um"]
    ]
    vol_shape_final = list(vol.optical_info["volume_shape"])
    print(f"Volume shape upscaled from {vol_shape_og} to {vol_shape_final}.")
    return vol


def light_field_to_1D(light_field, n_micro_lenses, pixels_per_ml):
    """
    Converts a 4D light field tensor to a 1D tensor.
    Args:
        light_field (torch.Tensor): 4D light field tensor.
    Returns:
        torch.Tensor: 1D tensor.
    """
    lenslet_list = []
    for ml_ii_idx in range(n_micro_lenses):
        for ml_jj_idx in range(n_micro_lenses):
            lenslet = light_field[
                ml_ii_idx * pixels_per_ml : (ml_ii_idx + 1) * pixels_per_ml,
                ml_jj_idx * pixels_per_ml : (ml_jj_idx + 1) * pixels_per_ml,
            ]
            lenslet_list.append(lenslet.flatten())

    # Concatenate all the lenslets to form a single 1D array
    light_field_1D = torch.cat(lenslet_list)
    return light_field_1D


def oneD_to_light_field(light_field_1D, n_micro_lenses, pixels_per_ml):
    """
    Converts a 1D tensor back to a 4D light field tensor.
    Args:
        light_field_1D (torch.Tensor): 1D tensor.
        n_micro_lenses (int): Number of micro-lenses along one dimension.
        pixels_per_ml (int): Number of pixels per micro-lens.
    Returns:
        torch.Tensor: 4D light field tensor.
    """
    num_pixels_per_ml = pixels_per_ml * pixels_per_ml
    light_field = torch.empty(
        (n_micro_lenses * pixels_per_ml, n_micro_lenses * pixels_per_ml),
        dtype=light_field_1D.dtype,
    )
    # Counter to keep track of the current position in the 1D tensor
    current_pos = 0
    # Iterate through each microlens and place it in the correct
    #   position in the 4D light field
    for ml_ii_idx in range(n_micro_lenses):
        for ml_jj_idx in range(n_micro_lenses):
            # Extract the current micro-lens' flattened image from the 1D tensor
            lenslet_flat = light_field_1D[current_pos : current_pos + num_pixels_per_ml]
            # Reshape it back into a 2D tensor
            lenslet = lenslet_flat.view(pixels_per_ml, pixels_per_ml)
            # Place the reshaped lenslet into the correct position in the 4D light field
            light_field[
                ml_ii_idx * pixels_per_ml : (ml_ii_idx + 1) * pixels_per_ml,
                ml_jj_idx * pixels_per_ml : (ml_jj_idx + 1) * pixels_per_ml,
            ] = lenslet
            # Update the current position in the 1D tensor
            current_pos += num_pixels_per_ml
    return light_field
