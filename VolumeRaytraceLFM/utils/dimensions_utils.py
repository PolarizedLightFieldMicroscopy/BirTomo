import torch


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
    cropped_tensor = reshaped_param[...,
                                    start_D:end_D, start_H:end_H, start_W:end_W]

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
    cropped_tensor = reshaped_param[...,
                                    start_D:end_D, start_H:end_H, start_W:end_W]
    return cropped_tensor


def store_as_pytorch_parameter(tensor, var_type: str):
    '''
    Converts a tensor to a PyTorch parameter and flattens appropriately.
    Note: possibly .type(torch.get_default_dtype()) is needed.
    '''
    if var_type == 'scalar':
        parameter = torch.nn.Parameter(tensor.flatten())
    elif var_type == 'vector':
        parameter = torch.nn.Parameter(tensor.reshape(3, -1))
    return parameter
