import os
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


def main():
    recon_optical_info = setup_optical_parameters("config_settings\optical_config_xylem_mla70.json")
    iteration_params = setup_iteration_parameters("config_settings\iter_config_xylem.json")

    wavelength_nm = recon_optical_info['wavelength'] * 1000
    ret_image_meas_polscope = io.imread(os.path.join('xylem', 'ret_mla70_slice14.png'))
    azim_image_meas_polscope = io.imread(os.path.join('xylem', 'azim_mla70_slice14.png'))
    # Note: potentially the images should be saved as np.float32
    ret_image_meas= normalize_retardance(ret_image_meas_polscope, 60, wavelength=wavelength_nm)
    azim_image_meas = normalize_azimuth(azim_image_meas_polscope)

    initial_volume = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=recon_optical_info,
        volume_creation_args = volume_args.random_args1
    )
    recon_directory = create_unique_directory("reconstructions")
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas, azim_image_meas,
                                        initial_volume, iteration_params, gt_vol=initial_volume)
    recon_config.save(recon_directory)
    # recon_config_recreated = ReconstructionConfig.load(recon_directory)
    reconstructor = Reconstructor(recon_config)
    reconstructor.reconstruct(output_dir=recon_directory)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)

if __name__ == '__main__':
    main()
