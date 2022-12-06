import torch
from VolumeRaytraceLFM.birefringence_implementations import *

from ray import *
from jones import *
import time

# Set objective info
optic_config = OpticConfig()
optic_config.PSF_config.M = 60      # Objective magnification
optic_config.PSF_config.NA = 1.2    # Objective NA
optic_config.PSF_config.ni = 1.52   # Refractive index of sample (experimental)
optic_config.PSF_config.ni0 = 1.52  # Refractive index of sample (design value)
optic_config.PSF_config.wvl = 0.550
optic_config.mla_config.n_pixels_per_mla = 17
optic_config.mla_config.n_voxels_per_ml = 1
optic_config.camera_config.sensor_pitch = 6.5
optic_config.mla_config.pitch = optic_config.mla_config.n_pixels_per_mla * optic_config.camera_config.sensor_pitch
optic_config.mla_config.n_micro_lenses = 1

# Define workspace
volume_shape = [11, 11, 11]
optic_config.volume_config.volume_shape = volume_shape
optic_config.volume_config.voxel_size_um = 3*[optic_config.mla_config.pitch / optic_config.PSF_config.M]
optic_config.volume_config.volume_size_um = np.array(optic_config.volume_config.volume_shape) * np.array(optic_config.volume_config.voxel_size_um)


############### Implementation

# Disable torch gradients, as we aren't doing any training or optimization 
with torch.no_grad():
    # Create a Birefringent Raytracer
    BF_raytrace = BirefringentRaytraceLFM(optic_config=optic_config, members_to_learn=[])

    # Compute the rays and use the Siddon's algorithm to compute the intersections with voxels.
    # If a filepath is passed as argument, the object with all its calculations get stored/loaded from a file
    startTime = time.time()
    BF_raytrace = BF_raytrace.compute_rays_geometry() #'test_ray_geometry'
    executionTime = (time.time() - startTime)
    print('Ray-tracing time in seconds: ' + str(executionTime))

    # Create a Birefringent volume, with a single birefringent voxel
    if True:
        my_volume = BF_raytrace.init_volume(init_mode='zeros')
        offset = 0
        my_volume.voxel_parameters.requires_grad = False
        my_volume.voxel_parameters[
                :, 
                BF_raytrace.vox_ctr_idx[0], 
                BF_raytrace.vox_ctr_idx[1]+offset, 
                BF_raytrace.vox_ctr_idx[2]+offset] \
                = torch.tensor([-0.1, 1/2, np.sqrt(3)/2, 0])    # birefringence and optic axis
    else: # whole plane
        my_volume = BF_raytrace.init_volume(init_mode='1planes')
    # Plot the volume in 3D
    # my_volume.plot_volume_plotly(opacity=0.1)

    # Traverse volume for every ray, and generate retardance and azimuth images
    startTime = time.time()
    ret_image_torch, azim_image_torch = BF_raytrace.ret_and_azim_images(my_volume)
    executionTime = (time.time() - startTime)

    print('Execution time in seconds with Torch: ' + str(executionTime))

    colormap = 'viridis'
    plt.figure(figsize=(10,2.5))
    plt.subplot(1,3,1)
    plt.imshow(ret_image_torch, origin='lower', cmap=colormap)
    plt.colorbar()
    plt.title('Retardance torch')
    plt.subplot(1,3,2)
    plt.imshow(azim_image_torch, origin='lower', cmap=colormap)
    plt.colorbar()
    plt.title('Azimuth torch')
    ax = plt.subplot(1,3,3)
    im = plot_birefringence_lines(ret_image_torch.numpy(), azim_image_torch.numpy(), origin='lower', cmap=colormap, line_color='white', ax=ax)
    plt.colorbar(im)
    plt.title('Ret+Azim')
    plt.show(block=True)
