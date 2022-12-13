'''(in progess) -12/13/2022
Attempting to use waveblocks workflow'''
# Third party libraries imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
import pathlib
from tifffile import imwrite

# Waveblocks imports
from waveblocks.microscopes.lightfield_micro import LFMicroscope
from waveblocks.blocks.optic_config import OpticConfig
from waveblocks.blocks.microlens_arrays import MLAType
import waveblocks.blocks.point_spread_function as psf
from waveblocks.utils.misc_utils import volume_2_projections

# Configuration parameters
file_path = pathlib.Path(__file__).parent.absolute()
data_path = file_path.parent.joinpath("data")
# Fetch Device to use: cpu or GPU?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ################### Object space specification
# depth_step = 0.43
# depth_range = [-depth_step*1, depth_step*1]
# vol_xy_size = 501
# depths = np.arange(depth_range[0], depth_range[1] + depth_step, depth_step)
# nDepths = len(depths)

# # Load volume to use as our object in front of the microscope
# vol_file = h5py.File(
#     data_path.joinpath("fish_phantom_251_251_51.h5"), "r"
# )
# GT_volume = (
#     torch.tensor(np.array(vol_file['fish_phantom']))
#     .permute(2, 1, 0)
#     .unsqueeze(0)
#     .unsqueeze(0)
#     .to(device)
# )
# GT_volume = torch.nn.functional.interpolate(
#     GT_volume, [vol_xy_size, vol_xy_size, nDepths]
# )
# # Set volume to correct shape [batch, z, x, y]
# GT_volume = GT_volume[:, 0, ...].permute(0, 3, 1, 2).contiguous()
# # Normalize volume
# GT_volume /= GT_volume.max()

import time         # to measure ray tracing time
import numpy as np  # to convert radians to degrees for plots
import matplotlib.pyplot as plt
from plotting_tools import plot_birefringence_lines, plot_birefringence_colorized
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume, \
                                                            BirefringentRaytraceLFM, \
                                                            JonesMatrixGenerators

# Select backend method
# backend = BackEnds.PYTORCH
backend = BackEnds.NUMPY

if backend == BackEnds.PYTORCH:
    from waveblocks.utils.misc_utils import *

optical_info = BirefringentVolume.get_optical_info_template()
# Alter some of the optical parameters
optical_info['volume_shape'] = [15, 51, 51]
optical_info['axial_voxel_size_um'] = 1.0
optical_info['pixels_per_ml'] = 17
optical_info['n_micro_lenses'] = 5
optical_info['n_voxels_per_ml'] = 1
# Create a volume
volume_type = 'shell'
shift_from_center = 0
volume_axial_offset = optical_info['volume_shape'][0] // 2 + shift_from_center # for center
my_volume = BirefringentVolume.create_dummy_volume(backend=backend, optical_info=optical_info, \
                                                    vol_type=volume_type, \
                                                    volume_axial_offset=volume_axial_offset)



################### Configure Microscope
# Create optical_config object with the information from the microscope
optical_config = OpticConfig()

# Update optical config from input PSF
psf_size = 255 # 17 pixel per microlenses, 11x11 microlenses, makes 17x11=187
optical_config.PSF_config.NA = 1.2 
optical_config.PSF_config.M = 60
optical_config.PSF_config.Ftl = 200000
optical_config.PSF_config.wvl = 0.593
optical_config.PSF_config.ni = 1.35
optical_config.PSF_config.ni0 = 1.35

# optical_config.PSF_config.depth_step = depth_step
# optical_config.PSF_config.depths = depths

# first zero found in the center at:
# depths = np.append(depths,2*optical_config.PSF_config.wvl/(optical_config.PSF_config.NA**2))

# Camera
optical_config.camera_config.sensor_pitch = 6.5

# Microlens array
optical_config.use_mla = True
optical_config.mla_config.pitch = 100
optical_config.mla_config.camera_distance = 2500
optical_config.mla_config.focal_length = 2500

# Update the optical configuration with the new parameters
optical_config.setup_parameters()

# # Create and compute PSF
# PSF = psf.PSF(optical_config)
# _, psf_in = PSF.forward(
#     optical_config.PSF_config.voxel_size[0], psf_size, optical_config.PSF_config. depths
# )

# Create a ligth-field WaveBlocks microscope Microscope
# Where: optical_config  = all microscope related information
# members_to_learn      = list of strings indicating wich parameters of the microscope to optimize, None in this case
# psf_in                = PSF complex wavefront at front focal plane of the tube lens
lf_microscope = LFMicroscope(optic_config=optical_config, members_to_learn=[], psf_in=psf_in).to(device)
# Here we tell Pytorch to set this module to evaluation mode, avoiding gradient computation.
lf_microscope.eval()


################### Compute a LF image through a forward projection
# We forward project the 3D volume GT_volume and generate a 2D LF image
# This is similar than calling lf_microscope.forward() function, this is an inbuilt Pytorch function
lf_image = lf_microscope(GT_volume)

volume_MIP = volume_2_projections(GT_volume)[0,0,...].cpu().numpy() # We remove the first two dimensions to convert to numpy compatible image
plt.subplot(1,2,1)
plt.imshow(volume_MIP)
plt.title('Volume MIP')
plt.subplot(1,2,2)
plt.imshow(lf_image[0,0,...].detach().cpu().numpy())
plt.title('Light-field image')
plt.savefig('output_PolScopeSettings_ni135.png')
# LF_center = LF_psf[0,:,-1,-1,:,:].detach().numpy() ; 
# imwrite('PSF_center.tif', LF_center)
plt.show()
