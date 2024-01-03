import os
import numpy as np
import torch
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.simulations import ForwardModel
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume
from VolumeRaytraceLFM.volumes import volume_args
from VolumeRaytraceLFM.setup_parameters import (
    setup_optical_parameters,
    setup_iteration_parameters
)
from VolumeRaytraceLFM.reconstructions import ReconstructionConfig, Reconstructor
from VolumeRaytraceLFM.visualization.plotting_volume import visualize_volume
from VolumeRaytraceLFM.utils.file_utils import create_unique_directory

BACKEND = BackEnds.PYTORCH
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def recon_sphere():
    optical_info = setup_optical_parameters(
        "config_settings/optical_config_sphere.json")
    simulate = True
    if simulate:
        optical_system = {'optical_info': optical_info}
        # Initialize the forward model. Raytracing is performed as part of the initialization.
        simulator = ForwardModel(optical_system, backend=BACKEND)
        # Volume creation
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.sphere_args6
        )
        # visualize_volume(volume_GT, optical_info)

        simulator.forward_model(volume_GT)
        simulator.view_images()
        ret_image_meas = simulator.ret_img
        azim_image_meas = simulator.azim_img
        # Save the images as numpy arrays
        if False:
            ret_numpy = ret_image_meas.detach().numpy()
            np.save('forward_images/ret_voxel_pos_1mla_17pix.npy', ret_numpy)
            azim_numpy = azim_image_meas.detach().numpy()
            np.save('forward_images/azim_voxel_pos_1mla_17pix.npy', azim_numpy)
    else:
        ret_image_meas = np.load(os.path.join(
            'forward_images', 'ret_voxel_pos_1mla_17pix.npy'))
        azim_image_meas = np.load(os.path.join(
            'forward_images', 'azim_voxel_pos_1mla_17pix.npy'))

    recon_optical_info = optical_info.copy()
    iteration_params = setup_iteration_parameters(
        "config_settings/iter_config_sphere.json")
    initial_volume = BirefringentVolume(
        backend=BACKEND,
        optical_info=recon_optical_info,
        volume_creation_args=volume_args.random_args
    )
    recon_directory = create_unique_directory("reconstructions")
    if not simulate:
        volume_GT = initial_volume
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas,
        azim_image_meas, initial_volume, iteration_params, gt_vol=volume_GT
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(recon_config, omit_rays_based_on_pixels=True)
    reconstructor.rays.verbose = False
    reconstructor.reconstruct(output_dir=recon_directory)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def recon_continuation():
    optical_info = setup_optical_parameters(
        "config_settings/optical_config_sphere.json")
    foward_img_str = 'sphere6_thick1_31mla_17pix.npy'
    volume_GT = BirefringentVolume(
        backend=BACKEND,
        optical_info=optical_info,
        volume_creation_args=volume_args.sphere_args6
    )
    # visualize_volume(volume_GT, optical_info)
    ret_image_meas = np.load(os.path.join(
        'forward_images', 'ret_' + foward_img_str))
    azim_image_meas = np.load(os.path.join(
        'forward_images', 'azim_' + foward_img_str))

    # reconstruction
    recon_optical_info = optical_info.copy()
    iteration_params = setup_iteration_parameters(
        "config_settings/iter_config_sphere.json")
    sphere_path = "volumes/2024-01-02_23-26-15/volume_ep_300_threshold_0.002_bir.h5"
    initial_volume = BirefringentVolume.init_from_file(
        sphere_path, BackEnds.PYTORCH, recon_optical_info)
    visualize_volume(initial_volume, recon_optical_info)

    recon_directory = create_unique_directory("reconstructions")
    # volume_GT = initial_volume

    # Compute the reconstuction
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas,
        azim_image_meas, initial_volume, iteration_params, gt_vol=volume_GT
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(recon_config, omit_rays_based_on_pixels=True)
    reconstructor.rays.verbose = False
    reconstructor.reconstruct(output_dir=recon_directory)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)  

def zero_near_zero(tensor, threshold):
    """
    Set elements of the tensor to zero if they are within a specified
    threshold of zero.

    Args:
        tensor (torch.Tensor): The input tensor.
        threshold (float): The threshold for determining "close to zero".

    Returns:
        torch.Tensor: The modified tensor with elements close to zero
                      set to zero.
    """
    # Identify elements within the threshold range
    close_to_zero = torch.logical_and(tensor > -threshold, tensor < threshold)

    # Set these elements to zero
    tensor[close_to_zero] = 0

    return tensor

def identify_close_to_zero(tensor, threshold):
    """
    Identify elements of the tensor that are within a specified
    threshold of zero.

    Args:
        tensor (torch.Tensor): The input tensor.
        threshold (float): The threshold for determining "close to zero".

    Returns:
        torch.Tensor: The modified tensor with elements close to zero
                      set to zero.
    """
    # Identify elements within the threshold range
    close_to_zero = torch.logical_and(tensor > -threshold, tensor < threshold)

    return close_to_zero

def zero_near_zero_voxels(volume : BirefringentVolume, threshold):
    """Set voxels of the volume to zero if they are within a specified
    threshold of zero."""
    volume.Delta_n.requires_grad_(False)
    volume.optic_axis.requires_grad_(False)
    close_to_zero = identify_close_to_zero(volume.Delta_n, threshold)
    volume.Delta_n[close_to_zero] = 0
    # volume.optic_axis[:, close_to_zero] = 0

    return volume

def threshold_and_save_volume(input_vol_path, output_vol_path, optical_info, threshold):
    """Load a volume, set voxels close to zero to zero, and save the volume."""
    input_volume = BirefringentVolume.init_from_file(
        input_vol_path, BackEnds.PYTORCH, optical_info)
    output_volume = zero_near_zero_voxels(input_volume, threshold)
    my_description = f"Volume {input_vol_path} thresholded at {threshold}"
    output_volume.save_as_file(output_vol_path, my_description)
    print(f"Saved volume to {output_vol_path}.")

if __name__ == '__main__':
    # ths = 0.002
    # optical_info = setup_optical_parameters("config_settings/optical_config_sphere.json")
    # threshold_and_save_volume(
    #     "volumes/2024-01-02_23-26-15/volume_ep_300.h5",
    #     f"volumes/2024-01-02_23-26-15/volume_ep_300_threshold_{ths}_bir.h5",
    #     optical_info, ths
    # )
    recon_continuation()
