import time
import matplotlib.pyplot as plt
from plotting_tools import *

""" This script using numpy and main_test_VolumeRaytraceLFM using Pytorch
    have the exact same functionality:
    - Create a volume with a single birefringent voxel.
    - Compute the ray geometry depending on the Light field microscope.
    - Traverse the rays through the volume and accumulate the retardance.
    - Compute the final ray retardance and azimuth for every ray.
    - Generate 2D images of a single lenslet. """

camera_pix_pitch = 6.5
objective_M = 60
voxel_size_um = 3*[camera_pix_pitch / objective_M]
n_micro_lenses = 25
n_pixels_per_ml = 17

optical_info={
            'volume_shape' : [9,51,51], 
            'voxel_size_um' : n_pixels_per_ml, 
            'pixels_per_ml' : 17, 
            'na_obj' : 1.2, 
            'n_medium' : 1.52,
            'wavelength' : 0.55,
            'n_micro_lenses' : n_micro_lenses,
            'n_voxels_per_ml' : 1}

back_end = BackEnds.PYTORCH

# Create a Birefringent Raytracer
BF_raytrace = BirefringentRaytraceLFM(back_end=back_end, optical_info=optical_info)

# Compute the rays and use the Siddon's algorithm to compute the intersections with voxels.
# If a filepath is passed as argument, the object with all its calculations get stored/loaded from a file
startTime = time.time()
BF_raytrace.compute_rays_geometry() 
executionTime = (time.time() - startTime)
print('Ray-tracing time in seconds: ' + str(executionTime))

# Move ray tracer to GPU
device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
print(f'Using computing device: {device}')
BF_raytrace = BF_raytrace.to(device)


# Single voxel
if False:
    my_volume = BF_raytrace.init_volume(init_mode='zeros')
    voxel_birefringence = 0.1
    voxel_birefringence_axis = torch.tensor([-0.5,0.5,0])
    voxel_birefringence_axis /= voxel_birefringence_axis.norm()
    offset = 0
    # Disable gradients in volume, as in-place assignment on tensors with gradients is not allowed in Pytorch
    my_volume.voxel_parameters.requires_grad = False
    my_volume.voxel_parameters[:,   BF_raytrace.vox_ctr_idx[0]-2, 
                                    BF_raytrace.vox_ctr_idx[1]+offset, 
                                    BF_raytrace.vox_ctr_idx[2]+offset] = torch.tensor([
                                                                                        voxel_birefringence, 
                                                                                        voxel_birefringence_axis[0], 
                                                                                        voxel_birefringence_axis[1], 
                                                                                        voxel_birefringence_axis[2]])
    # Re-enable gradients                                                                                           
    my_volume.voxel_parameters.requires_grad = True

else: # whole plane
    my_volume = BF_raytrace.init_volume(volume_shape=optical_info['volume_shape'], init_mode='ellipsoid')

# my_volume.plot_volume_plotly(opacity=0.1)



with torch.no_grad():
    # Perform same calculation with torch
    startTime = time.time()
    ret_image_measured, azim_image_measured = BF_raytrace.ray_trace_through_volume(my_volume) 
    executionTime = (time.time() - startTime)
    print('Execution time in seconds with Torch: ' + str(executionTime))

    # Store GT images
    Delta_n_GT = my_volume.Delta_n.detach().clone()
    optic_axis_GT = my_volume.optic_axis.detach().clone()
    ret_image_measured = ret_image_measured.detach()
    azim_image_measured = azim_image_measured.detach()


############# Torch
# Let's create an optimizer
# Initial guess
my_volume = BF_raytrace.init_volume(volume_shape=optical_info['volume_shape'], init_mode='random')

# Plot
colormap = 'viridis'
plt.figure(figsize=(10,2.5))
plt.subplot(1,3,1)
plt.imshow(ret_image,cmap=colormap)
plt.colorbar()
plt.title('Retardance numpy')
plt.subplot(1,3,2)
plt.imshow(azim_image,cmap=colormap)
plt.colorbar()
plt.title('Azimuth numpy')
ax = plt.subplot(1,3,3)
im = plot_birefringence_lines(ret_image, azim_image,cmap=colormap, line_color='white', ax=ax)
plt.colorbar(im)
plt.title('Ret+Azim')
plt.show(block=True)

plt.show(block=True)
