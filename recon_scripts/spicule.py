import os
import torch
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume
from VolumeRaytraceLFM.volumes import volume_args
from VolumeRaytraceLFM.volumes.optic_axis import fill_vector_based_on_nonaxial
from VolumeRaytraceLFM.setup_parameters import (
    setup_optical_parameters,
    setup_iteration_parameters,
)
from VolumeRaytraceLFM.reconstructions import ReconstructionConfig, Reconstructor
from VolumeRaytraceLFM.utils.file_utils import create_unique_directory
from utils.polscope import prepare_ret_azim_images
from utils.logging import redirect_output_to_log, restore_output


BACKEND = BackEnds.PYTORCH
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def define_spicule_image_paths(mla):
    ret_image_path = os.path.join(
        "data", "spicule", f"mla{mla}", "retardance_zeroed.tif")
    azim_image_path = os.path.join(
        "data", "spicule", f"mla{mla}", "azimuth.tif")
    radiometry_path = os.path.join(
        "data", "spicule", f"mla{mla}", "radiometry.tif")
    return ret_image_path, azim_image_path, radiometry_path


def setup_spicule_iteration_params(init_vol_path, ret_image_path, 
    azim_image_path, radiometry_path, volume_shape, mla,
    load_rays=False):
    """Setup the iteration parameters for the spicule reconstruction."""
    iteration_params = setup_iteration_parameters(
        "config/spicule/iter_config.json"
    )
    iteration_params.update({
        "initial volume path": init_vol_path,
        "ret image path": ret_image_path,
        "azim image path": azim_image_path,
        "radiometry_path": radiometry_path
    })
    if load_rays:
        v0, v1, v2 = volume_shape
        iteration_params["saved_ray_path"] = os.path.join(
            "config", "spicule", "rays", f"mla{mla}_vol{v0}_{v1}_{v2}.pkl"
        )
    return iteration_params


def recon_spicule(
    init_vol_path, recon_postfix, mla=None, ss_factor=None, volume_shape=None, load_rays=False
):
    # Setup optical parameters
    optical_info = setup_optical_parameters("config/spicule/optical_config.json")
    if volume_shape is not None:
        optical_info["volume_shape"] = volume_shape
    if mla is not None:
        optical_info["n_micro_lenses"] = mla
    if ss_factor is not None:
        optical_info["n_voxels_per_ml"] = ss_factor
    volume_shape = optical_info["volume_shape"]
    mla = optical_info["n_micro_lenses"]
    ss_factor = optical_info["n_voxels_per_ml"]
    print(f"Volume shape: {volume_shape} using supersampling of {ss_factor}")

    # Define paths to the images
    ret_image_path, azim_image_path, radiometry_path = define_spicule_image_paths(mla)

    # Setup iteration parameters
    iteration_params = setup_spicule_iteration_params(
        init_vol_path,
        ret_image_path,
        azim_image_path,
        radiometry_path,
        volume_shape,
        mla,
        load_rays=load_rays
    )

    # Prepare the retardance and azimuth images
    ret_image_meas, azim_image_meas = prepare_ret_azim_images(
        ret_image_path, azim_image_path, 150, optical_info["wavelength"]
    )

    if init_vol_path is None:
        initial_volume = BirefringentVolume(
            backend=BackEnds.PYTORCH,
            optical_info=optical_info,
            volume_creation_args=volume_args.random_neg_args1,
        )
        # Remove after optic axis function (neg -> pos) is incorporated
        fill_vector_based_on_nonaxial(
            initial_volume.optic_axis, initial_volume.optic_axis[1:, ...]
        )
    else:
        initial_volume = BirefringentVolume.init_from_file(
            init_vol_path, BackEnds.PYTORCH, optical_info
        )

    # Create reconstruction directory and log file
    v0, v1, v2 = volume_shape
    parent_dir = os.path.join(
        "reconstructions", f"spicule_mla{mla}", f"ss{ss_factor}", f"{v0}_{v1}_{v2}"
    )
    recon_directory = create_unique_directory(parent_dir, postfix=recon_postfix)
    log_file_path = os.path.join(recon_directory, "output_log.txt")
    log_file = redirect_output_to_log(log_file_path)
    recon_config = ReconstructionConfig(
        optical_info,
        ret_image_meas,
        azim_image_meas,
        initial_volume,
        iteration_params,
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(
        recon_config,
        output_dir=recon_directory,
        device=DEVICE,
        omit_rays_based_on_pixels=True,
    )
    reconstructor.reconstruct(plot_live=True)

    # Restore output
    restore_output(log_file)


if __name__ == "__main__":
    init_vol_path = None
    recon_spicule(
        init_vol_path,
        "axial_voxels",
        load_rays=True
    )
