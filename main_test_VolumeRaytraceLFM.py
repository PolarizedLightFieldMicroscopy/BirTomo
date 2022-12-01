import torch
from VolumeRaytraceLFM.birefringence_implementations import *

from ray import *
from jones import *

# Set objective info
optic_config = OpticConfig()
optic_config.PSF_config.M = 60      # Objective magnification
optic_config.PSF_config.NA = 1.2    # Objective NA
optic_config.PSF_config.ni = 1.52   # Refractive index of sample (experimental)
optic_config.PSF_config.ni0 = 1.52  # Refractive index of sample (design value)
optic_config.PSF_config.wvl = 0.550
optic_config.mla_config.n_pixels_per_mla = 7
optic_config.camera_config.sensor_pitch = 6.5
optic_config.mla_config.pitch = optic_config.mla_config.n_pixels_per_mla * optic_config.camera_config.sensor_pitch
optic_config.mla_config.n_mlas = 100

optic_config.volume_config.volume_shape = [5, 5, 5]
optic_config.volume_config.voxel_size_um = [1,] + 2*[optic_config.mla_config.pitch / optic_config.PSF_config.M]
optic_config.volume_config.volume_size_um = np.array(optic_config.volume_config.volume_shape) * np.array(optic_config.volume_config.voxel_size_um)

# Create a Birefringent Raytracer
BF_raytrace = BirefringentRaytraceLFM(optic_config=optic_config, members_to_learn=[])
# Compute the rays and use the Siddon's algorithm to compute the intersections with voxels.
# If a filepath is passed as argument, the object with all its calculations get stored/loaded from a file

BF_raytrace = BF_raytrace.compute_rays_geometry() #'test_ray_geometry'

# Create a Birefringent volume, with random 
volume = BF_raytrace.init_volume(init_mode='random')

volume_3_planes = BF_raytrace.init_volume(init_mode='3planes')
# # Plot the volume with plotly
# volume_3_planes.plot_volume_plotly()

my_volume = BF_raytrace.init_volume(init_mode='zeros')
my_volume.voxel_parameters[:, BF_raytrace.vox_ctr_idx[0], BF_raytrace.vox_ctr_idx[1], BF_raytrace.vox_ctr_idx[2]] = torch.tensor([np.pi, 1, 0, 0])
my_volume.plot_volume_plotly()

ray_enter, ray_exit, ray_diff = rays_through_vol(optic_config.mla_config.n_pixels_per_mla, optic_config.PSF_config.NA, optic_config.PSF_config.ni, BF_raytrace.volCtr)
ret_image, azim_image = ret_and_azim_images(ray_enter, ray_exit, ray_diff, optic_config.mla_config.n_pixels_per_mla, my_volume.voxel_parameters)

# f, axarr = plt.subplots(2,1)
# axarr[0].imshow(ret_image)
# axarr[0].set_title("Retardance")
# plt.colorbar()
# axarr[1].imshow(azim_image)
# axarr[1].set_title("Azimuth")
# plt.show()

plt.subplot(1,2,1)
plt.imshow(ret_image)
plt.colorbar()
plt.title('Retardance')
plt.subplot(1,2,2)
plt.imshow(azim_image)
plt.colorbar()
plt.title('Azimuth')
plt.show(block=True)

# plt.imshow(ret_image)
# plt.show()

# plt.imshow(azim_image)
# plt.show()
