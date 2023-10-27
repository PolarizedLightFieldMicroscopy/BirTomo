import torch
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.simulations import ForwardModel
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume
from VolumeRaytraceLFM.volumes import volume_args
from VolumeRaytraceLFM.setup_parameters import setup_optical_parameters

BACKEND = BackEnds.PYTORCH

def visualize_volume(volume: BirefringentVolume, optical_info: dict):
    # with torch.no_grad():
    plotly_figure = volume.plot_lines_plotly()
    plotly_figure = volume.plot_volume_plotly(optical_info,
                                            voxels_in=volume.get_delta_n(),
                                            opacity=0.02,
                                            fig=plotly_figure
                                            )
    plotly_figure.show()
    return

def adjust_volume(volume: BirefringentVolume):
    if BACKEND == BackEnds.PYTORCH:
        with torch.no_grad():
            volume.get_delta_n()[:optical_info['volume_shape'][0] // 2 + 2,...] = 0
    else:
        volume.get_delta_n()[:optical_info['volume_shape'][0] // 2 + 2,...] = 0
    return volume

if __name__ == '__main__':
    optical_info = setup_optical_parameters("VolumeRaytraceLFM\optical_config1.json")
    optical_system = {'optical_info': optical_info}
    simulator = ForwardModel(optical_system, backend=BACKEND)
    volume_GT = BirefringentVolume(
                    backend=BACKEND,
                    optical_info=optical_info,
                    volume_creation_args=volume_args.voxel_args
                    )
    # volume_GT = adjust_volume(volume_GT)
    visualize_volume(volume_GT, optical_info)
    simulator.forward_model(volume_GT)
    simulator.view_images()
