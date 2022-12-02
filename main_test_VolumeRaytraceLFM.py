import torch
from VolumeRaytraceLFM.birefringence_implementations import *

from ray import *
from jones import *
import time

# Set objective info
optic_config = OpticConfig()
optic_config.PSF_config.M = 60      # Objective magnification
optic_config.PSF_config.NA = 1.2    # Objective NA
optic_config.PSF_config.ni = 1.33   # Refractive index of sample (experimental)
optic_config.PSF_config.ni0 = 1.33  # Refractive index of sample (design value)
optic_config.PSF_config.wvl = 0.550
optic_config.mla_config.n_pixels_per_mla = 17
optic_config.camera_config.sensor_pitch = 6.5
optic_config.mla_config.pitch = optic_config.mla_config.n_pixels_per_mla * optic_config.camera_config.sensor_pitch
optic_config.mla_config.n_mlas = 100

optic_config.volume_config.volume_shape = [11, 11, 11]
optic_config.volume_config.voxel_size_um = [1,] + 2*[optic_config.mla_config.pitch / optic_config.PSF_config.M]
optic_config.volume_config.volume_size_um = np.array(optic_config.volume_config.volume_shape) * np.array(optic_config.volume_config.voxel_size_um)

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

    # Create a Birefringent volume, with random 
    volume = BF_raytrace.init_volume(init_mode='random')


    # Single voxel
    if True:
        my_volume = BF_raytrace.init_volume(init_mode='zeros')
        offset = 0
        my_volume.voxel_parameters.requires_grad = False
        my_volume.voxel_parameters[
                :, 
                BF_raytrace.vox_ctr_idx[0], 
                BF_raytrace.vox_ctr_idx[1]+offset, 
                BF_raytrace.vox_ctr_idx[2]+offset] \
                = torch.tensor([0.1, np.sqrt(2), 0, np.sqrt(2)])
    else: # whole plane
        my_volume = BF_raytrace.init_volume(init_mode='1planes')
    # my_volume.plot_volume_plotly(opacity=0.1)

    # Perform same calculation with torch
    startTime = time.time()
    ret_image_torch, azim_image_torch = BF_raytrace.ret_and_azim_images(my_volume)
    executionTime = (time.time() - startTime)


    print('Execution time in seconds with Torch: ' + str(executionTime))
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.imshow(ret_image_torch)
    plt.colorbar()
    plt.title('Retardance torch')
    plt.subplot(1,2,2)
    plt.imshow(azim_image_torch)
    plt.colorbar()
    plt.title('Azimuth torch')
    plt.show(block=True)
