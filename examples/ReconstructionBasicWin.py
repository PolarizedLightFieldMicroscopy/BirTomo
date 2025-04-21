# %% Importing necessary libraries
import os
import numpy as np
import torch
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.simulations import ForwardModel
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume
from VolumeRaytraceLFM.volumes import volume_args
from VolumeRaytraceLFM.setup_parameters import (
    setup_optical_parameters,
    setup_iteration_parameters,
)
from VolumeRaytraceLFM.reconstructions import ReconstructionConfig, Reconstructor
from VolumeRaytraceLFM.utils.file_utils import (
    create_unique_directory,
)

# %% Configuration paramters to be changed
simulate = True

# Volume creation arguments, see src/VolumeRaytraceLFM/volumes/volume_args.py for more options
volume_gt_creation_args = volume_args.voxel_args
volume_initial_creation_args = volume_args.random_pos_args

# Paths to the optical and iteration configuration files
optical_config_file = os.path.join("..", "config", "optical_config_voxel.json")
iter_config_file = os.path.join("..", "config", "iter_config_reg.json")

# Path to the directory where the reconstruction will be saved
recon_output_dir = os.path.join("..", "reconstructions", "voxel")

# Whether to continue a previous reconstruction
continue_recon = False
recon_init_file_path = r"to be alterned.h5"

# For loading forward images that were saved in a previous reconstruction folder
measurement_dir = os.path.join(recon_output_dir, "to be altered")

BACKEND = BackEnds.PYTORCH
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# %% Simulate or load forward images
optical_info = setup_optical_parameters(optical_config_file)

if simulate:
    optical_system = {"optical_info": optical_info}
    simulator = ForwardModel(optical_system, backend=BACKEND)
    volume_GT = BirefringentVolume(
        backend=BACKEND,
        optical_info=optical_info,
        volume_creation_args=volume_gt_creation_args,
    )
    simulator.forward_model(volume_GT)
    simulator.view_images()
    ret_image_meas = simulator.ret_img.detach().numpy()
    azim_image_meas = simulator.azim_img.detach().numpy()
else:
    ret_image_meas = np.load(os.path.join(measurement_dir, "ret_image.npy"))
    azim_image_meas = np.load(os.path.join(measurement_dir, "azim_image.npy"))
    volume_GT = None

# %% Run reconstruction
recon_optical_info = optical_info.copy()
iteration_params = setup_iteration_parameters(iter_config_file)
recon_dir_postfix = iteration_params["general"]["output_directory_postfix"]
recon_directory = create_unique_directory(recon_output_dir, postfix=recon_dir_postfix)
if continue_recon:
    initial_volume = BirefringentVolume.init_from_file(recon_init_file_path, BACKEND, recon_optical_info)
else:
    initial_volume = BirefringentVolume(
        backend=BACKEND,
        optical_info=recon_optical_info,
        volume_creation_args=volume_initial_creation_args,
    )
recon_config = ReconstructionConfig(
    recon_optical_info,
    ret_image_meas,
    azim_image_meas,
    initial_volume,
    iteration_params,
    gt_vol=volume_GT
)
recon_config.save(recon_directory)
reconstructor = Reconstructor(
    recon_config,
    output_dir=recon_directory,
    device=DEVICE
)
reconstructor.reconstruct()
print("Reconstruction complete")

# %%
