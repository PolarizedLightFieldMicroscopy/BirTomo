import os
import torch
import numpy as np
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume
from VolumeRaytraceLFM.volumes import volume_args
from VolumeRaytraceLFM.volumes.optic_axis import fill_vector_based_on_nonaxial
from VolumeRaytraceLFM.setup_parameters import (
    setup_optical_parameters,
    setup_iteration_parameters,
)
from VolumeRaytraceLFM.reconstructions import ReconstructionConfig, Reconstructor
from VolumeRaytraceLFM.visualization.plotting_volume import visualize_volume
from VolumeRaytraceLFM.utils.file_utils import create_unique_directory
from utils.polscope import prepare_ret_azim_images
from utils.logging import redirect_output_to_log, restore_output

BACKEND = BackEnds.PYTORCH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"

def recon_spicula(
    init_vol_path, recon_postfix, mla=86, ss_factor=1, volume_shape=[20, 80, 80]
):
    optical_info = setup_optical_parameters("config/spicula/optical_config.json")
    optical_info["volume_shape"] = volume_shape

    ret_image_path = os.path.join(
        "data", "spicula", f"mla{mla}", "retardance_zeroed_low_nbrs_radio10.tif")
    azim_image_path = os.path.join(
        "data", "spicula", f"mla{mla}", "azimuth.tif")
    radiometry_path = os.path.join(
        "data", "spicula", f"mla{mla}", "radiometry_10.tif")

    optical_info["n_micro_lenses"] = mla

    v0, v1, v2 = volume_shape
    parent_dir = os.path.join(
        "reconstructions", f"spicula_mla{mla}", f"ss{ss_factor}", f"{v0}_{v1}_{v2}"
    )
    optical_info["n_voxels_per_ml"] = ss_factor
    iteration_params = setup_iteration_parameters(
        "config/spicula/iter_config.json"
    )
    # iteration_params["saved_ray_path"] = os.path.join(
    #     "config", "rays", "water", f"mla{mla}_vol{v0}_{v1}_{v2}.pkl"
    # )
    print(f"Volume shape: {volume_shape} using supersampling of {ss_factor}")

    iteration_params["ret image path"] = ret_image_path
    iteration_params["azim image path"] = azim_image_path
    iteration_params["initial volume path"] = init_vol_path
    iteration_params["radiometry_path"] = radiometry_path

    ret_image_meas, azim_image_meas = prepare_ret_azim_images(

        ret_image_path, azim_image_path, 120, optical_info["wavelength"]
    )

    if init_vol_path is None:
        initial_volume = BirefringentVolume(
            backend=BackEnds.PYTORCH,
            optical_info=optical_info,
            volume_creation_args=volume_args.random_args1,
        )
        fill_vector_based_on_nonaxial(
            initial_volume.optic_axis, initial_volume.optic_axis[1:, ...]
        )
    else:
        initial_volume = BirefringentVolume.init_from_file(
            init_vol_path, BackEnds.PYTORCH, optical_info
        )

    initial_volume.to(DEVICE)

    recon_directory = create_unique_directory(parent_dir, postfix=recon_postfix)
    log_file_path = os.path.join(recon_directory, "output_log.txt")
    log_file = redirect_output_to_log(log_file_path)
    recon_config = ReconstructionConfig(
        optical_info,
        ret_image_meas,
        azim_image_meas,
        initial_volume,
        iteration_params,
        gt_vol=initial_volume,
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(
        recon_config,
        output_dir=recon_directory,
        device=DEVICE,
        omit_rays_based_on_pixels=True,
        apply_volume_mask=False,
    )

    reconstructor.to_device(DEVICE)
    reconstructor.rays.verbose = False
    reconstructor.reconstruct(plot_live=True)
    restore_output(log_file)
    # visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)

if __name__ == "__main__":
    init_vol_path = None
    plot_only = False
    if not plot_only:
        recon_spicula(
            init_vol_path,
            "debug",
            mla=88,
            volume_shape=[30, 100, 100],
            ss_factor=1,
        )
