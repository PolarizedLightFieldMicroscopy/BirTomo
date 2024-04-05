'''Regularization metrics for a birefringent volume'''
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume
from VolumeRaytraceLFM.metrics.regularization_fundamentals import *


def l2_bir(volume: BirefringentVolume):
    return l2(volume.Delta_n)


def total_variation_bir(volume: BirefringentVolume):
    birefringence = volume.get_delta_n()
    return total_variation_3d_volumetric(birefringence)


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
