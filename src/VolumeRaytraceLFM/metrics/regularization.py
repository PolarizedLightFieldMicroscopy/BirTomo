"""Regularization metrics for a birefringent volume"""

from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume
from VolumeRaytraceLFM.metrics.regularization_fundamentals import *
import torch.nn.functional as F


def l2_bir(volume: BirefringentVolume):
    return l2(volume.Delta_n)


def l2_bir_active(volume: BirefringentVolume):
    return l2(volume.birefringence_active)


def l1_bir(volume: BirefringentVolume):
    return l1(volume.Delta_n)


def total_variation_bir(volume: BirefringentVolume):
    birefringence = volume.get_delta_n()
    return total_variation_3d_volumetric(birefringence)


def total_variation_optax(volume: BirefringentVolume):
    optax = volume.get_optic_axis()
    return total_variation_3d_volumetric(optax)


def cosine_similarity_neighbors(volume: BirefringentVolume):
    """Compute a loss that encourages each vector in optic_axis to
    align with its neighbors, weighted by delta_n.
    """
    delta_n = volume.get_delta_n()
    optic_axis = volume.get_optic_axis()
    return weighted_local_cosine_similarity_loss(optic_axis, delta_n)


def neg_penalty_bir_active(volume: BirefringentVolume):
    return negative_penalty(volume.birefringence_active)


def pos_penalty_bir_active(volume: BirefringentVolume):
    return positive_penalty(volume.birefringence_active)


def pos_penalty_l2_bir_active(volume: BirefringentVolume):
    return positive_penalty_l2(volume.birefringence_active)


def total_variation_bir_subset(volume: BirefringentVolume):
    birefringence = volume.birefringence_active
    return total_variation(birefringence)


def masked_zero_loss(volume: BirefringentVolume, mask: torch.Tensor):
    """Compute the loss enforcing certain positions in the volume's prediction to
    be zero based on a mask.
    
    Args:
    - volume (BirefringentVolume): The volume object containing the predicted tensor.
    - mask (torch.Tensor): A binary mask tensor where positions to be zeroed are marked with 1.
    
    Returns:
    - torch.Tensor: The computed loss.
    """
    # Ensure the mask is a binary tensor
    mask = mask.float()
    
    # Invert the mask to get positions that should be zero
    inverted_mask = 1 - mask
    
    # Apply the inverted mask to the volume's birefringence
    masked_birefringence = volume.birefringence.flatten() * inverted_mask
    
    # Compute the loss as the mean squared error between the masked
    #   birefringence and a zero tensor
    zero_tensor = torch.zeros_like(volume.birefringence.flatten())
    loss = F.mse_loss(masked_birefringence, zero_tensor)
    return loss


def l2_biref(volume: BirefringentVolume):
    return l2(volume.birefringence)


def pos_penalty_biref(volume: BirefringentVolume):
    return positive_penalty(volume.birefringence)


def pos_penalty_l2_biref(volume: BirefringentVolume):
    return positive_penalty_l2(volume.birefringence)


class AnisotropyAnalysis:
    def __init__(self, volume: BirefringentVolume):
        self.volume = volume
        self.birefringence = volume.get_delta_n()
        self.optic_axis = volume.get_optic_axis()

    def l2_regularization(self):
        return l2(self.birefringence)

    def total_variation_regularization(self):
        return total_variation_3d_volumetric(self.birefringence)

    def process_optic_axis(self):
        # Example method to process optic axis data
        # Implement the specific logic you need for optic axis data
        pass
