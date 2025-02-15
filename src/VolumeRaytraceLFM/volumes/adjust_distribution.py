import numpy as np
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume
from VolumeRaytraceLFM.volumes.modification import adjust_birefringence_distribution_from_retardance


def scale_bir_from_ret(volume: BirefringentVolume, ret_image_meas: np.ndarray) -> BirefringentVolume:
    # Adjust the birefringence distribution to align with the retardance image
    birefringence = volume.get_delta_n().detach().numpy()
    optical_info = volume.optical_info
    scaled_birefringence = adjust_birefringence_distribution_from_retardance(
        birefringence, ret_image_meas, optical_info
    )
    volume = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=optical_info,
        Delta_n=scaled_birefringence,
        optic_axis=volume.get_optic_axis(),
    )
    return volume
