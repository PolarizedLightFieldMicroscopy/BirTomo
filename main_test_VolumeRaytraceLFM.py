import torch
from VolumeRaytraceLFM.birefringence_implementations import *

# Set objective info
optic_config = OpticConfig()
optic_config.PSF_config.M = 60      # Objective magnification
optic_config.PSF_config.NA = 1.2    # Objective NA
optic_config.PSF_config.ni = 1.52   # Refractive index of sample (experimental)
optic_config.PSF_config.ni0 = 1.52  # Refractive index of sample (design value)
optic_config.PSF_config.wvl = 0.550
optic_config.mla_config.n_pixels_per_mla = 16
optic_config.camera_config.sensor_pitch = 6.5
optic_config.mla_config.pitch = optic_config.mla_config.n_pixels_per_mla * optic_config.camera_config.sensor_pitch
optic_config.mla_config.n_mlas = 100

optic_config.volume_config.volume_shape = [10, 10, 10]
optic_config.volume_config.voxel_size_um = [1,] + 2*[optic_config.mla_config.pitch / optic_config.PSF_config.M]
optic_config.volume_config.volume_size_um = np.array(optic_config.volume_config.volume_shape) * np.array(optic_config.volume_config.voxel_size_um)

# Create a Birefringent Raytracer
BF_raytrace = BirefringentRaytraceLFM(optic_config=optic_config, members_to_learn=[])
# Compute the rays and use the Siddon's algorithm to compute the intersections with voxels.
# If a filepath is passed as argument, the object with all its calculations get stored/loaded from a file

BF_raytrace = BF_raytrace.compute_rays_geometry('test_ray_geometry')

# Create a Birefringent volume, with random 
volume = BF_raytrace.init_volume(init_mode='random')

volume_3_planes = BF_raytrace.init_volume(init_mode='3planes')
# Plot the volume with plotly
volume_3_planes.plot_volume_plotly()

