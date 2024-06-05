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


def total_variation_bir_subset(volume: BirefringentVolume):
    birefringence = volume.birefringence_active
    return total_variation(birefringence)


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
