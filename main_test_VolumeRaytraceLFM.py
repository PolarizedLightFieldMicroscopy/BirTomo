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
optic_config.mla_config.n_pixels_per_mla = 17
optic_config.camera_config.sensor_pitch = 6.5
optic_config.mla_config.pitch = optic_config.mla_config.n_pixels_per_mla * optic_config.camera_config.sensor_pitch
optic_config.mla_config.n_mlas = 100

optic_config.volume_config.volume_shape = [11, 10, 10]
optic_config.volume_config.voxel_size_um = [1,] + 2*[optic_config.mla_config.pitch / optic_config.PSF_config.M]
optic_config.volume_config.volume_size_um = np.array(optic_config.volume_config.volume_shape) * np.array(optic_config.volume_config.voxel_size_um)

# Create a Birefringent Raytracer
BF_raytrace = BirefringentRaytraceLFM(optic_config=optic_config, members_to_learn=[])
# Compute the rays and use the Siddon's algorithm to compute the intersections with voxels.
# If a filepath is passed as argument, the object with all its calculations get stored/loaded from a file

BF_raytrace = BF_raytrace.compute_rays_geometry() #'test_ray_geometry'

# Create a Birefringent volume, with random 
volume = BF_raytrace.init_volume(init_mode='random')


# Single voxel
if True:
    my_volume = BF_raytrace.init_volume(init_mode='zeros')
    my_volume.voxel_parameters[:, BF_raytrace.vox_ctr_idx[0], BF_raytrace.vox_ctr_idx[1], BF_raytrace.vox_ctr_idx[2]] = torch.tensor([np.pi, 1, 0, 0])
else: # whole plane
    my_volume = BF_raytrace.init_volume(init_mode='1planes')
my_volume.plot_volume_plotly(opacity=0.1)


# Computed with numpy functions
ray_enter, ray_exit, ray_diff = rays_through_vol(optic_config.mla_config.n_pixels_per_mla, optic_config.PSF_config.NA, optic_config.PSF_config.ni, BF_raytrace.volCtr)

# Comparing ray_enter/exit/diff with BF_raytrace.ray_entry/exit/direction... All the same :) 
ret_image, azim_image = ret_and_azim_images(ray_enter, ray_exit, ray_diff, optic_config.mla_config.n_pixels_per_mla, my_volume.voxel_parameters, optic_config)

plt.figure()
plt.subplot(2,2,1)
plt.imshow(ret_image)
plt.colorbar()
plt.title('Retardance Ray.py')
plt.subplot(2,2,2)
plt.imshow(azim_image)
plt.colorbar()
plt.title('Azimuth')

# Computed with BF_raytrace
ray_enter, ray_exit, ray_diff = BF_raytrace.ray_entry.numpy(), BF_raytrace.ray_exit.numpy(), BF_raytrace.ray_direction.numpy()
# Comparing ray_enter/exit/diff with BF_raytrace.ray_entry/exit/direction... All the same :) 
ret_image, azim_image = ret_and_azim_images(ray_enter, ray_exit, ray_diff, optic_config.mla_config.n_pixels_per_mla, my_volume.voxel_parameters, optic_config)

plt.subplot(2,2,3)
plt.imshow(ret_image)
plt.colorbar()
plt.title('Retardance BF_raytrace')
plt.subplot(2,2,4)
plt.imshow(azim_image)
plt.colorbar()
plt.title('Azimuth')
plt.show(block=True)

# plt.imshow(ret_image)
# plt.show()

# plt.imshow(azim_image)
# plt.show()
