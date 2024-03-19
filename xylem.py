import os
import time
import torch
import skimage.io as io
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
from utils.polscope import normalize_retardance, normalize_azimuth

BACKEND = BackEnds.PYTORCH
DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

RET_FILENAME = 'retardance_mla66_ths2000.tif' # 'retardance_mla66.tif'

def get_ret_azim_images(recon_optical_info):
    wavelength_nm = recon_optical_info['wavelength'] * 1000
    ret_image_meas_polscope = io.imread(os.path.join('xylem', RET_FILENAME))
    azim_image_meas_polscope = io.imread(os.path.join('xylem', 'azimuth_mla66.tif'))
    # Note: potentially the images should be saved as np.float32
    ret_image_meas = normalize_retardance(ret_image_meas_polscope, 60, wavelength=wavelength_nm)
    azim_image_meas = normalize_azimuth(azim_image_meas_polscope)
    return ret_image_meas, azim_image_meas


def recon_debug():
    """Reconstruct the xylem data set for debugging purposes."""
    start_time = time.time()
    recon_optical_info = setup_optical_parameters("config_settings/optical_config_xylem_quick.json")
    iteration_params = setup_iteration_parameters("config_settings/iter_config_xylem_quick.json")
    ret_image_meas, azim_image_meas = get_ret_azim_images(recon_optical_info)

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
    reconstructor.reconstruct(output_dir=recon_directory)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)
    end_time = time.time()
    print("CPU execution time: {:.2f} seconds".format(end_time - start_time))


def recon_xylem():
    """Reconstruct the xylem data set."""
    recon_optical_info = setup_optical_parameters("config_settings/optical_config_xylem.json")
    iteration_params = setup_iteration_parameters("config_settings/iter_config_xylem.json")
    ret_image_meas, azim_image_meas = get_ret_azim_images(recon_optical_info)

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


def recon_continuation(init_vol_path):
    """Reconstruct the xylem data set from a previous reconstruction."""
    recon_optical_info = setup_optical_parameters("config_settings/optical_config_xylem.json")
    iteration_params = setup_iteration_parameters("config_settings/iter_config_xylem.json")
    ret_image_meas, azim_image_meas = get_ret_azim_images(recon_optical_info)

    initial_volume = BirefringentVolume.init_from_file(
        init_vol_path, BackEnds.PYTORCH, recon_optical_info)
    visualize_volume(initial_volume, recon_optical_info)
    recon_directory = create_unique_directory("reconstructions", postfix='xylem_continue')
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas,
        azim_image_meas, initial_volume, iteration_params, gt_vol=initial_volume
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(recon_config, omit_rays_based_on_pixels=True, apply_volume_mask=True)
    reconstructor.rays.verbose = False
    reconstructor.reconstruct(output_dir=recon_directory)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)

if __name__ == '__main__':
    # recon_debug()
    recon_xylem()
    # xylem_vol_path = os.path.join('reconstructions', '2024-01-04_17-57-13', 'volume_ep_100.h5')
    # recon_continuation(xylem_vol_path)
    
    # Visualize a volume
    # optical_info = setup_optical_parameters("config_settings/optical_config_xylem.json")
    # volume = BirefringentVolume.init_from_file(
    #     xylem_vol_path, BackEnds.PYTORCH, optical_info)
    # visualize_volume(volume, optical_info)
