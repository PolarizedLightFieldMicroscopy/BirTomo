'''Regularization functions that can use used in the optimization process.'''
import torch
import torch.nn.functional as F


def l1(data):
    return torch.abs(data).mean()


def l2(data):
    return torch.pow(data, 2).mean()


def linfinity(data, weight=1.0):
    return weight * torch.max(torch.abs(data))


def elastic_net(data, weight1=1.0, weight2=1.0):
    l1_term = torch.abs(data).sum()
    l2_term = torch.pow(data, 2).sum()
    return weight1 * l1_term + weight2 * l2_term


def negative_penalty(data):
    return torch.relu(-data).mean()


def total_variation_3d_volumetric(data):
    """
    Computes the Total Variation regularization for a 4D tensor representing volumetric data.
    Args:
        data (torch.Tensor): Input 3D tensor with shape [depth, height, width].
    Returns:
        torch.Tensor: Computed Total Variation regularization term.
    """
    # Calculate the differences between adjacent elements along each spatial dimension
    diff_depth = torch.pow(data[1:, :, :] - data[:-1, :, :], 2).mean()
    diff_height = torch.pow(data[:, 1:, :] - data[:, :-1, :], 2).mean()
    diff_width = torch.pow(data[:, :, 1:] - data[:, :, :-1], 2).mean()

    tv_reg = diff_depth + diff_height + diff_width
    return tv_reg


def weighted_local_cosine_similarity_loss(vector_arr, scalar_arr):
    """
    Compute a loss that encourages each vector to align
    with its neighbors, weighted by scalar.

    Args:
        vector_arr (torch.Tensor): Tensor of shape (3, D, H, W)
            representing a 3D volume of vectors.
        scalar_arr (torch.Tensor): Tensor of shape (D, H, W) with weights
            for each spatial point.

    Returns:
        torch.Tensor: Scalar tensor representing the loss.
    
    The loss is between 0 and 2.
    """
    # TODO: make unnormalized version to hopefully make this term decrease
    normalized_vector = F.normalize(vector_arr, p=2, dim=0)
    normalized_scalar = scalar_arr / scalar_arr.abs().max() / 2
    err_message = "Normalized birefringence values are not within the range [-0.5, 0.5]."
    assert torch.all((normalized_scalar >= -0.5) & (normalized_scalar <= 0.5)), err_message

    cos_sim_loss = 0.0
    valid_comparisons = 0

    # Compute cosine similarity with local neighbors along each dimension
    for i in range(1, 4):  # Iterate over D, H, W dimensions
        rolled_vector = torch.roll(normalized_vector, shifts=-1, dims=i)
        
        # Compute cosine similarity (dot product) along this dimension
        #   Array of shape (D, H, W) with floats between -1 and 1
        cos_sim = (normalized_vector * rolled_vector).sum(dim=0)

        rolled_scalar = torch.roll(normalized_scalar, shifts=-1, dims=i-1)

        weighted_cos_sim = cos_sim * (normalized_scalar.abs() + rolled_scalar.abs()) / 2

        # Create the valid mask to avoid boundary elements
        valid_mask = torch.ones_like(weighted_cos_sim, dtype=torch.bool)
        index_tensor = torch.tensor([weighted_cos_sim.size(i-1) - 1], device=weighted_cos_sim.device)
        valid_mask.index_fill_(i-1, index_tensor, False)

        cos_sim_loss += (1 - weighted_cos_sim[valid_mask]).sum()
        valid_comparisons += valid_mask.sum().item()

    cos_sim_loss /= valid_comparisons

    return cos_sim_loss
