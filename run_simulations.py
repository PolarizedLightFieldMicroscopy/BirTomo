"""This script uses numpy/pytorch back-end to:
- Compute the ray geometry depending on the light field microscope and volume configuration.
- Create a volume with different birefringent shapes.
- Traverse the rays through the volume.
- Compute the retardance and azimuth for every ray.
- Generate 2D images.
"""
import torch
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.simulations import ForwardModel
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume
from VolumeRaytraceLFM.volumes import volume_args
from VolumeRaytraceLFM.setup_parameters import setup_optical_parameters
from VolumeRaytraceLFM.visualization.plotting_volume import visualize_volume

# Select backend method
BACKEND = BackEnds.PYTORCH
# backend = BackEnds.NUMPY

if BACKEND == BackEnds.PYTORCH:
    import torch
    torch.set_grad_enabled(False)
else:
    device = 'cpu'


def adjust_volume(volume: BirefringentVolume):
    """Removes half of the values of the volume."""
    if BACKEND == BackEnds.PYTORCH:
        with torch.no_grad():
            volume.get_delta_n()[:optical_info['volume_shape']
                                 [0] // 2 + 2, ...] = 0
    else:
        volume.get_delta_n()[:optical_info['volume_shape']
                             [0] // 2 + 2, ...] = 0
    return volume


if __name__ == '__main__':
    optical_info = setup_optical_parameters(
        "config_settings/optical_config_voxel.json")
    optical_system = {'optical_info': optical_info}
    simulator = ForwardModel(optical_system, backend=BACKEND)
    volume_GT = BirefringentVolume(
        backend=BACKEND,
        optical_info=optical_info,
        volume_creation_args=volume_args.voxel_args
    )
    # volume_GT = adjust_volume(volume_GT)
    # volume_GT.save_as_file('volume_gt.h5', description="")
    # visualize_volume(volume_GT, optical_info)

    simulator.forward_model(volume_GT)
    simulator.view_images()
