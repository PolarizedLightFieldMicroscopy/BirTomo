import torch
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.simulations import ForwardModel
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume
from VolumeRaytraceLFM.volumes import volume_args
from VolumeRaytraceLFM.setup_parameters import setup_optical_parameters, setup_iteration_parameters
from VolumeRaytraceLFM.reconstructions import ReconstructionConfig, Reconstructor

BACKEND = BackEnds.PYTORCH
SAVE_FORWARD_IMAGES = True
SESSION_DIR = 'Oct27'

DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

DIR_POSTFIX = 'mla5'
VOL_NAME = 'voxel_test'

def visualize_volume(volume: BirefringentVolume, optical_info: dict):
    with torch.no_grad():
        plotly_figure = volume.plot_lines_plotly()
        plotly_figure = volume.plot_volume_plotly(optical_info,
                                                voxels_in=volume.get_delta_n(),
                                                opacity=0.02,
                                                fig=plotly_figure
                                                )
        plotly_figure.show()
    return

if __name__ == '__main__':
    optical_info = setup_optical_parameters("VolumeRaytraceLFM\optical_config3.json")
    optical_system = {'optical_info': optical_info}
    simulator = ForwardModel(optical_system, backend=BACKEND)
    # Volume creation
    volume_GT = BirefringentVolume(
        backend=BACKEND,
        optical_info=optical_info,
        volume_creation_args=volume_args.voxel_args
    )
    visualize_volume(volume_GT, optical_info)
    simulator.forward_model(volume_GT)
    simulator.view_images()
    ret_image_meas = simulator.ret_img
    azim_image_meas = simulator.azim_img
    Delta_n_GT = volume_GT.get_delta_n().detach().clone()


    recon_optical_info = optical_info
    iteration_params = setup_iteration_parameters("VolumeRaytraceLFM\iter_config.json")
    initial_volume = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=recon_optical_info,
        volume_creation_args = volume_args.random_args
    )
    recon_config = ReconstructionConfig(optical_info, ret_image_meas, azim_image_meas, initial_volume, iteration_params)
    reconstructor = Reconstructor(recon_config)
    reconstructor.reconstruct()
