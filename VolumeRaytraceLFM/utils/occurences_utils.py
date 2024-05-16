"""Ultility functions for finding occurences of elements in a tensor."""


def indices_with_multiple_occurences(tensor, num_occurences):
    """
    Find the indices of elements in a 1D tensor that occur multiple times.
    Args:
        tensor (torch.Tensor): The input tensor.
    Returns:
        torch.Tensor: The indices of elements that occur multiple times.
    """
    unique, counts = tensor.unique(return_counts=True)
    mask = counts >= num_occurences
    filtered_unique = unique[mask]
    filtered_counts = counts[mask]
    return filtered_unique, filtered_counts
