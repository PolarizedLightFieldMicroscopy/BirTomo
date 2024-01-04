import torch
import torch.nn as nn
import torch.nn.functional as F


class VonMisesLoss(nn.Module):
    def __init__(self, kappa=1.0):
        super().__init__()
        self.kappa = kappa

    def forward(self, orientation_pred, orientation_gt):
        diff = orientation_pred - orientation_gt
        loss = 1 - torch.exp(self.kappa * torch.cos(diff))
        return loss.mean()


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, vector_pred, vector_gt):
        cos_sim = F.cosine_similarity(vector_pred, vector_gt, dim=-1)
        loss = 1 - cos_sim
        return loss.mean()


def apply_loss_function_and_reg(loss_type, reg_types, retardance_measurement, orientation_measurement,
                                retardance_estimate, orientation_estimate, ret_orie_weight=0.5,
                                volume_estimate=None, regularization_weights=0.01):
    ret_meas = retardance_measurement
    orien_meas = orientation_measurement
    ret_est = retardance_estimate
    orien_est = orientation_estimate
    if loss_type == 'vonMisses':
        retardance_loss = F.mse_loss(ret_meas, ret_est)
        angular_loss = (1 - F.cosine_similarity(orien_meas, orien_est)).mean()
        data_term = (1 - ret_orie_weight) * retardance_loss + ret_orie_weight * angular_loss
    # Vector difference
    if loss_type == 'vector':
        co_gt, ca_gt = ret_meas * torch.cos(orien_meas), ret_meas * torch.sin(orien_meas)
        co_pred, ca_pred = ret_est * torch.cos(orien_est), ret_est * torch.sin(orien_est)
        data_term = ((co_gt - co_pred)**2 + (ca_gt - ca_pred)**2).mean()
    elif loss_type == 'L1_cos':
        data_term = (1 - ret_orie_weight) * F.l1_loss(ret_meas, ret_est) + \
            ret_orie_weight * torch.cos(orien_meas - orien_est).abs().mean()
    elif loss_type == 'L1all':
        azimuth_damp_mask = (ret_meas / ret_meas.max()).detach()
        data_term = (ret_meas - ret_est).abs().mean() + \
            (2 * (1 - torch.cos(orien_meas - orien_est)) * azimuth_damp_mask).mean()

    if volume_estimate is not None:
        if not isinstance(reg_types, list):
            reg_types = [reg_types]
        if not isinstance(regularization_weights, list):
            regularization_weights = len(reg_types) * [regularization_weights]

        regularization_term_total = torch.zeros([1], device=ret_meas.device)
        for reg_type, reg_weight in zip(reg_types, regularization_weights):
            if reg_type == 'L1':
                # L1 or sparsity
                regularization_term = volume_estimate.Delta_n.abs().mean()
            # L2 or sparsity
            elif reg_type == 'L2':
                regularization_term = (volume_estimate.Delta_n**2).mean()
            # Unit length regularizer
            elif reg_type == 'unit':
                regularization_term = (
                    1-(volume_estimate.optic_axis[0, ...]**2+volume_estimate.optic_axis[1, ...]**2+volume_estimate.optic_axis[2, ...]**2)).abs().mean()
            elif reg_type == 'TV':
                delta_n = volume_estimate.get_delta_n()
                regularization_term = (delta_n[1:,   ...] - delta_n[:-1, ...]).pow(2).sum() + \
                    (delta_n[:, 1:, ...] - delta_n[:, :-1, ...]).pow(2).sum() + \
                    (delta_n[:, :,  1:] -
                     delta_n[:, :, :-1]).pow(2).sum()
            else:
                regularization_term = torch.zeros(
                    [1], device=ret_meas.device)
            regularization_term_total += regularization_term * reg_weight

    total_loss = data_term + regularization_term_total
    return total_loss, data_term, regularization_term_total


def cosine_similarity_loss_scaled(optic_axis, delta_n):
    """
    Compute a loss that encourages vectors in optic_axis to point in
    similar directions, scaled by the values in delta_n.

    Args:
        optic_axis (torch.Tensor): Tensor of shape (3, N) where N is the
                                   number of 3D vectors.
        delta_n (torch.Tensor): Tensor of shape (N,) with scaling
                                factors for each vector.

    Returns:
        torch.Tensor: Scalar tensor representing the loss.
    """

    # Normalize the vectors to unit length
    normalized_vectors = F.normalize(optic_axis, p=2, dim=0)

    # Compute pairwise cosine similarity (N, N)
    cos_sim_matrix = torch.matmul(normalized_vectors.transpose(0, 1), normalized_vectors)

    # Scale the cosine similarity matrix by delta_n
    # We use an outer product to create a scaling matrix of shape (N, N)
    scaling_matrix = torch.outer(delta_n, delta_n)
    scaled_cos_sim_matrix = cos_sim_matrix * scaling_matrix

    # Ignore the diagonal elements by setting them to zero
    scaled_cos_sim_matrix.fill_diagonal_(0)

    # Loss is the negative sum of the off-diagonal elements
    loss = -scaled_cos_sim_matrix.sum()

    # Normalize the loss by the number of comparisons
    num_vectors = optic_axis.size(1)
    loss /= (num_vectors * (num_vectors - 1))

    return loss
