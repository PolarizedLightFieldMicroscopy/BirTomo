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
    recon_optical_info = setup_optical_parameters("config_settings/xylem/optical_config_xylem.json")
    iteration_params = setup_iteration_parameters("config_settings/xylem/iter_config_xylem.json")
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
    recon_directory = create_unique_directory("reconstructions", postfix='xylem_debug')
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas, azim_image_meas,
                                        initial_volume, iteration_params, gt_vol=initial_volume)
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(recon_config, output_dir=recon_directory,
                                  omit_rays_based_on_pixels=True, apply_volume_mask=False)
    reconstructor.rays.verbose = True
    reconstructor.reconstruct(plot_live=False)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def recon_xylem(recon_dir_postfix='xylem'):
    """Reconstruct the xylem data set."""
    recon_optical_info = setup_optical_parameters("config_settings/xylem/optical_config_xylem.json")
    iteration_params = setup_iteration_parameters("config_settings/xylem/iter_config_xylem.json")
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
    recon_directory = create_unique_directory("reconstructions", postfix=recon_dir_postfix)
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas, azim_image_meas,
                                        initial_volume, iteration_params, gt_vol=initial_volume)
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(recon_config, output_dir=recon_directory,
                                  omit_rays_based_on_pixels=True, apply_volume_mask=False)
    reconstructor.rays.verbose = False
    reconstructor.reconstruct()
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def recon_cpu():
    start_time = time.time()
    recon_optical_info = setup_optical_parameters("config_settings/xylem/optical_config_xylem.json")
    iteration_params = setup_iteration_parameters("config_settings/xylem/iter_config_xylem_quick.json")
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
    reconstructor.reconstruct()
    # visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)
    end_time = time.time()
    print(f"CPU execution time: {end_time - start_time:.2f} seconds")


def recon_gpu():
    '''Reconstruct a volume on the GPU.'''
    start_time = time.time()
    recon_optical_info = setup_optical_parameters("config_settings/xylem/optical_config_xylem.json")
    iteration_params = setup_iteration_parameters("config_settings/xylem/iter_config_xylem_quick.json")
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

    reconstructor = Reconstructor(recon_config, output_dir=recon_directory,
                                  device=DEVICE)
    reconstructor.to_device(DEVICE)  # Move the reconstructor to the GPU

    reconstructor.reconstruct()
    # visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)
    end_time = time.time()
    print(f"GPU execution time: {end_time - start_time:.2f} seconds")


def recon_continuation(init_vol_path, recon_dir_postfix='xylem_continue'):
    """Reconstruct the xylem data set from a previous reconstruction."""
    recon_optical_info = setup_optical_parameters("config_settings/xylem/optical_config_xylem.json")
    iteration_params = setup_iteration_parameters("config_settings/xylem/iter_config_xylem.json")
    iteration_params["initial volume path"] = init_vol_path
    ret_image_meas, azim_image_meas = prepare_ret_azim_images(
        os.path.join('xylem', 'mla65', 'retardance.tif'),
        os.path.join('xylem', 'mla65', 'azimuth.tif'),
        60, recon_optical_info['wavelength']
    )
    initial_volume = BirefringentVolume.init_from_file(
        init_vol_path, BackEnds.PYTORCH, recon_optical_info)
    # visualize_volume(initial_volume, recon_optical_info)
    parent_dir = os.path.join('reconstructions', 'xylem_20_100_100')
    recon_directory = create_unique_directory(parent_dir, postfix=recon_dir_postfix)
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas,
        azim_image_meas, initial_volume, iteration_params
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(recon_config, output_dir=recon_directory,
                        omit_rays_based_on_pixels=True, apply_volume_mask=False)
    reconstructor.rays.verbose = True
    reconstructor.reconstruct(all_prop_elements=False, plot_live=False)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def recon_up3_continuation(init_vol_path, recon_dir_postfix='xylem_continue'):
    """Reconstruct the xylem data set from a previous reconstruction."""
    recon_optical_info = setup_optical_parameters("config_settings/optical_config_xylem_upscaled3.json")
    iteration_params = setup_iteration_parameters("config_settings/iter_config_xylem_ss3.json")
    iteration_params["initial volume path"] = init_vol_path
    ret_image_meas, azim_image_meas = prepare_ret_azim_images(
        os.path.join('xylem', 'mla65', 'retardance.tif'),
        os.path.join('xylem', 'mla65', 'azimuth.tif'),
        60, recon_optical_info['wavelength']
    )
    initial_volume = BirefringentVolume.init_from_file(
        init_vol_path, BackEnds.PYTORCH, recon_optical_info)
    # visualize_volume(initial_volume, recon_optical_info)
    parent_dir = os.path.join('reconstructions', 'xylem_60_300_300')
    recon_directory = create_unique_directory(parent_dir, postfix=recon_dir_postfix)
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas,
        azim_image_meas, initial_volume, iteration_params
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(recon_config, output_dir=recon_directory,
                        omit_rays_based_on_pixels=True, apply_volume_mask=False)
    reconstructor.rays.verbose = True
    reconstructor.reconstruct(all_prop_elements=False, plot_live=False)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)



if __name__ == '__main__':
    # recon_debug()
    # recon_gpu()
    # recon_xylem(recon_dir_postfix='xylem_from_random')
    
    up3 = True
    if not up3:
        saved_recon_dir = os.path.join('reconstructions', 'saved', 'xylem65', 'cont_with_subset')
        recon_filename = os.path.join('2024-04-30_14-17-15_xylem_neg_penalty', 'volume_ep_0020.h5')
        xylem_vol_path = os.path.join(saved_recon_dir, recon_filename)
        recon_continuation(xylem_vol_path, recon_dir_postfix='xylem_neg_penalty_small')
    else:
        saved_recon_dir = os.path.join('reconstructions', 'saved', 'xylem65_up3', 'subset_from_beg')
        recon_upscaled_filename = os.path.join('2024-05-01_22-25-35_xylem_low_betas_high_lr_cont', 'volume_ep_0050.h5')
        xylem_vol_path = os.path.join(saved_recon_dir, recon_upscaled_filename)
        # xylem_vol_path = os.path.join('objects', 'random_vol_60_300_300.h5')
        # recon_up3_continuation(xylem_vol_path, recon_dir_postfix='xylem_low_betas24')
    
    # Visualize a volume
    optical_info = setup_optical_parameters("config_settings/xylem/optical_config_xylem.json")
    # optical_info['volume_shape'] = [20, 100, 100]
    xylem_vol_path = r"reconstructions\saved\xylem65_up3\upsampled_recon\2024-04-30_10-51-45_xylem_up3_from_lr6_crash\volume_ep_0001.h5"
    xylem_vol_path = r"reconstructions\saved\xylem65_up3\upsampled_recon\2024-04-30_13-24-52_xylem_up3_filled\volume_ep_0004.h5"
    xylem_vol_path = r"C:\Users\Geneva\Documents\Code\GeoBirT\reconstructions\xylem_60_300_300\2024-05-02_19-23-25_xylem_low_betas24\volume_ep_0030.h5"
    volume = BirefringentVolume.init_from_file(
        xylem_vol_path, BackEnds.PYTORCH, optical_info)
    visualize_volume(volume, optical_info)
