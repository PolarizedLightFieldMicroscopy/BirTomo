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

def get_ret_azim_images(recon_optical_info):
    wavelength_nm = recon_optical_info['wavelength'] * 1000
    ret_image_meas_polscope = io.imread(os.path.join('xylem', 'retardance_mla66.tif'))
    azim_image_meas_polscope = io.imread(os.path.join('xylem', 'azimuth_mla66.tif'))
    # Note: potentially the images should be saved as np.float32
    ret_image_meas = normalize_retardance(ret_image_meas_polscope, 60, wavelength=wavelength_nm)
    azim_image_meas = normalize_azimuth(azim_image_meas_polscope)
    return ret_image_meas, azim_image_meas


def recon_debug():
    start_time = time.time()
    recon_optical_info = setup_optical_parameters("config_settings/optical_config_xylem.json")
    iteration_params = setup_iteration_parameters("config_settings/iter_config_xylem_quick.json")
    ret_image_meas, azim_image_meas = get_ret_azim_images(recon_optical_info)

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
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)
    end_time = time.time()
    print("CPU execution time: {:.2f} seconds".format(end_time - start_time))

if __name__ == '__main__':
    recon_debug()
