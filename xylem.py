import os
import time
import torch
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume
from VolumeRaytraceLFM.volumes import volume_args
from VolumeRaytraceLFM.setup_parameters import (
    setup_optical_parameters,
    setup_iteration_parameters
    )
from VolumeRaytraceLFM.reconstructions import ReconstructionConfig, Reconstructor
from VolumeRaytraceLFM.visualization.plotting_volume import visualize_volume
from VolumeRaytraceLFM.utils.file_utils import create_unique_directory
from utils.polscope import prepare_ret_azim_images

BACKEND = BackEnds.PYTORCH
DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def recon_debug():
    """Reconstruct the xylem data set for debugging purposes."""
    start_time = time.time()
    recon_optical_info = setup_optical_parameters("config_settings/optical_config_xylem.json")
    iteration_params = setup_iteration_parameters("config_settings/iter_config_xylem.json")
    start_time = time.time()
    ret_image_meas, azim_image_meas = prepare_ret_azim_images(
        os.path.join('xylem', 'mla65', 'retardance.tif'),
        os.path.join('xylem', 'mla65', 'azimuth.tif'),
        60, recon_optical_info['wavelength']
    )
    end_time = time.time()
    print("Prepared measured images in {:.5f} seconds".format(end_time - start_time))
    initial_volume = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=recon_optical_info,
        volume_creation_args = volume_args.random_args1
    )
    recon_directory = create_unique_directory("reconstructions", postfix='xylem_debug')
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas, azim_image_meas,
                                        initial_volume, iteration_params, gt_vol=initial_volume)
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(recon_config, omit_rays_based_on_pixels=True, apply_volume_mask=True)
    reconstructor.rays.verbose = True
    reconstructor.reconstruct(output_dir=recon_directory, plot_live=True)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)
    end_time = time.time()
    print("CPU execution time: {:.2f} seconds".format(end_time - start_time))


def recon_xylem():
    """Reconstruct the xylem data set."""
    recon_optical_info = setup_optical_parameters("config_settings/optical_config_xylem.json")
    iteration_params = setup_iteration_parameters("config_settings/iter_config_xylem.json")
    ret_image_meas, azim_image_meas = prepare_ret_azim_images(
        os.path.join('xylem', 'mla65', 'retardance.tif'),
        os.path.join('xylem', 'mla65', 'azimuth.tif'),
        60, recon_optical_info['wavelength']
    )
    initial_volume = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=recon_optical_info,
        volume_creation_args = volume_args.random_args1
    )
    recon_directory = create_unique_directory("reconstructions", postfix='xylem_randstart')
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas, azim_image_meas,
                                        initial_volume, iteration_params, gt_vol=initial_volume)
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(recon_config, omit_rays_based_on_pixels=True, apply_volume_mask=False)
    reconstructor.rays.verbose = False
    reconstructor.reconstruct(output_dir=recon_directory)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def recon_cpu():
    start_time = time.time()
    recon_optical_info = setup_optical_parameters("config_settings/optical_config_xylem.json")
    iteration_params = setup_iteration_parameters("config_settings/iter_config_xylem_quick.json")
    ret_image_meas, azim_image_meas = prepare_ret_azim_images(
        os.path.join('xylem', 'mla65', 'retardance.tif'),
        os.path.join('xylem', 'mla65', 'azimuth.tif'),
        60, recon_optical_info['wavelength']
    )
    initial_volume = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=recon_optical_info,
        volume_creation_args = volume_args.random_args1
    )
    recon_directory = create_unique_directory("reconstructions")
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas, azim_image_meas,
                                        initial_volume, iteration_params, gt_vol=initial_volume)
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(recon_config)
    reconstructor.reconstruct(output_dir=recon_directory)
    # visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)
    end_time = time.time()
    print("CPU execution time: {:.2f} seconds".format(end_time - start_time))


def recon_gpu():
    '''Reconstruct a volume on the GPU.'''
    start_time = time.time()
    recon_optical_info = setup_optical_parameters("config_settings/optical_config_xylem.json")
    iteration_params = setup_iteration_parameters("config_settings/iter_config_xylem_quick.json")
    ret_image_meas, azim_image_meas = prepare_ret_azim_images(
        os.path.join('xylem', 'mla65', 'retardance.tif'),
        os.path.join('xylem', 'mla65', 'azimuth.tif'),
        60, recon_optical_info['wavelength']
    )
    initial_volume = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=recon_optical_info,
        volume_creation_args = volume_args.random_args1
    )
    initial_volume.to(DEVICE)  # Move the volume to the GPU

    recon_directory = create_unique_directory("reconstructions")
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas, azim_image_meas,
                                        initial_volume, iteration_params, gt_vol=initial_volume)
    recon_config.save(recon_directory)

    reconstructor = Reconstructor(recon_config, device=DEVICE)
    reconstructor.to_device(DEVICE)  # Move the reconstructor to the GPU

    reconstructor.reconstruct(output_dir=recon_directory)
    # visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)
    end_time = time.time()
    print("GPU execution time: {:.2f} seconds".format(end_time - start_time))


def recon_continuation(init_vol_path, recon_dir_postfix='xylem_continue'):
    """Reconstruct the xylem data set from a previous reconstruction."""
    recon_optical_info = setup_optical_parameters("config_settings/optical_config_xylem.json")
    iteration_params = setup_iteration_parameters("config_settings/iter_config_xylem_continue.json")
    ret_image_meas, azim_image_meas = prepare_ret_azim_images(
        os.path.join('xylem', 'mla65', 'retardance.tif'),
        os.path.join('xylem', 'mla65', 'azimuth.tif'),
        60, recon_optical_info['wavelength']
    )
    initial_volume = BirefringentVolume.init_from_file(
        init_vol_path, BackEnds.PYTORCH, recon_optical_info)
    # visualize_volume(initial_volume, recon_optical_info)
    recon_directory = create_unique_directory("reconstructions", postfix=recon_dir_postfix)
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas,
        azim_image_meas, initial_volume, iteration_params, gt_vol=initial_volume
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(recon_config, omit_rays_based_on_pixels=True, apply_volume_mask=True)
    reconstructor.rays.verbose = False
    reconstructor.reconstruct(output_dir=recon_directory)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


if __name__ == '__main__':
    recon_debug()
    # recon_gpu()
    # recon_xylem()
    # saved_recon_dir = "reconstructions/saved/xylem"
    # recon_filename = "2024-03-25_04-26-10_xylem_highL2_lr5/volume_ep_0100.h5"
    # xylem_vol_path = os.path.join(saved_recon_dir, recon_filename)
    # recon_continuation(xylem_vol_path, recon_dir_postfix='xylem_ths634_lr4')
    
    # Visualize a volume
    # optical_info = setup_optical_parameters("config_settings/optical_config_xylem.json")
    # volume = BirefringentVolume.init_from_file(
    #     xylem_vol_path, BackEnds.PYTORCH, optical_info)
    # visualize_volume(volume, optical_info)
