"""Functions to create masks to apply to the volumes."""
import torch

def create_half_zero_mask(shape):
    """
    Creates a 3D mask with the first half of the elements in the
    third dimension set to zero.

    The mask is initialized as a 3D tensor filled with ones. The
    first half of the elements in the third dimension are then
    set to zero for each element in the first dimension. The
    resulting mask is then flattened into a 1-dimensional tensor.

    Args:
        shape (tuple): A 3-element tuple specifying the dimensions.

    Returns:
        torch.Tensor: The resulting mask, flattened into a 1D tensor.
    """
    mask = torch.ones(shape)
    half_elements = shape[2] // 2
    for i in range(shape[0]):
        mask[i, :, :half_elements] = 0
    return mask.flatten()


def create_half_zero_sandwich_mask(shape):
    """
    Creates a 3D mask with the first half of the elements in the
    third dimension set to zero.

    The mask is initialized as a 3D tensor filled with ones. The
    first half of the elements in the third dimension are then
    set to zero for each element in the first dimension. The
    resulting mask is then flattened into a 1-dimensional tensor.

    Args:
        shape (tuple): A 3-element tuple specifying the dimensions.

    Returns:
        torch.Tensor: The resulting mask, flattened into a 1D tensor.
    """
    mask = torch.ones(shape)
    half_elements = shape[2] // 2
    for i in range(shape[0]):
        mask[i, :, :half_elements] = 0
        mask[i, :, half_elements + 2:] = 0
    return mask.flatten()
