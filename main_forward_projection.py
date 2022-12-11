import time
import matplotlib.pyplot as plt
from plotting_tools import *
from VolumeRaytraceLFM.birefringence_implementations import *
from waveblocks.utils.misc_utils import *

""" This script using numpy/pytorch back-end to:
    - Create a volume with different birefringent shapes.
    - Compute the ray geometry depending on the Light field microscope and volume configuration.
    - Traverse the rays through the volume.
    - Compute the retardance and azimuth for every ray.
    - Generate 2D images."""

# Select back ends
# back_end = BackEnds.PYTORCH
back_end = BackEnds.NUMPY
torch.set_default_tensor_type(torch.DoubleTensor)

camera_pix_pitch = 6.5
objective_M = 60
pixels_per_ml = 17
optical_info={
            'volume_shape' : [15,51,51], 
            'voxel_size_um' : 3*[camera_pix_pitch * pixels_per_ml / objective_M], 
            'pixels_per_ml' : pixels_per_ml, 
            'na_obj' : 1.2, 
            'n_medium' : 1.52,
            'wavelength' : 0.55,
            'n_micro_lenses' : 5,
            'n_voxels_per_ml' : 1,
            'polarizer' : np.array([[1, 0], [0, 1]]),
            'analyzer' : np.array([[1, 0], [0, 1]])
            }



# Volume type
# number is the shift from the end of the volume, change it as you wish, do single_voxel{volume_shape[0]//2} for a voxel in the center
# for shift in range(-5,6):
shift_from_center = 0
volume_axial_offset = optical_info['volume_shape'][0]//2+shift_from_center #for center
# volume_type = 'ellipsoid'
volume_type = 'shell'
# volume_type = 'single_voxel' 

# Plot azimuth
# azimuth_plot_type = 'lines'
azimuth_plot_type = 'hsv'


if volume_type == 'single_voxel':
    optical_info['n_micro_lenses'] = 1
    azimuth_plot_type = 'lines'



# Create a Birefringent Raytracer
BF_raytrace = BirefringentRaytraceLFM(back_end=back_end, optical_info=optical_info)

# Compute the rays and use the Siddon's algorithm to compute the intersections with voxels.
# If a filepath is passed as argument, the object with all its calculations get stored/loaded from a file
startTime = time.time()
BF_raytrace.compute_rays_geometry() 
executionTime = (time.time() - startTime)
print('Ray-tracing time in seconds: ' + str(executionTime))

# Move ray tracer to GPU
if back_end == BackEnds.PYTORCH:
    # Disable gradients 
    torch.set_grad_enabled(False)
    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    print(f'Using computing device: {device}')
    BF_raytrace = BF_raytrace.to(device)


# Single voxel
if volume_type == 'single_voxel':
    voxel_delta_n = 0.1
    voxel_birefringence_axis = torch.tensor([1,0.0,0])
    voxel_birefringence_axis /= voxel_birefringence_axis.norm()

    # Create empty volume
    my_volume = BF_raytrace.init_volume(optical_info['volume_shape'], init_mode='zeros')
    # Set delta_n
    my_volume.Delta_n[volume_axial_offset, 
                                    BF_raytrace.vox_ctr_idx[1], 
                                    BF_raytrace.vox_ctr_idx[2]] = voxel_delta_n
    # set optical_axis
    my_volume.optic_axis[:, volume_axial_offset, 
                            BF_raytrace.vox_ctr_idx[1], 
                            BF_raytrace.vox_ctr_idx[2]] = torch.tensor([voxel_birefringence_axis[0], 
                                                                            voxel_birefringence_axis[1], 
                                                                            voxel_birefringence_axis[2]]) if back_end == BackEnds.PYTORCH else voxel_birefringence_axis


elif volume_type == 'shell' or volume_type == 'ellipsoid': # whole plane
    ellipsoid_args = {  'radius' : [5.5, 9.5, 5.5],
                        'center' : [volume_axial_offset/optical_info['volume_shape'][0], 0.50, 0.5],   # from 0 to 1
                        'delta_n' : -0.1,
                        'border_thickness' : 0.3}

    my_volume = BF_raytrace.init_volume(volume_shape=optical_info['volume_shape'], init_mode='ellipsoid', init_args=ellipsoid_args)


    # Do we want a shell? let's remove some of the volume
    if volume_type == 'shell':
        my_volume.Delta_n[:optical_info['volume_shape'][0]//2+2,...] = 0

# my_volume.plot_volume_plotly(optical_info, voxels=my_volume.Delta_n, opacity=0.1)



# Perform same calculation with torch
startTime = time.time()
# Create non-identity polarizers and analyzers
if False:
    # LC-PolScope setup
    optical_info['polarizer'] = JonesMatrixGenerators.left_circular_polarizer()
    optical_info['analyzer'] = JonesMatrixGenerators.universal_compensator(np.pi / 4, np.pi / 2)
ret_image, azim_image = BF_raytrace.ray_trace_through_volume(my_volume) 
executionTime = (time.time() - startTime)
print(f'Execution time in seconds with backend {back_end}: ' + str(executionTime))

if back_end == BackEnds.PYTORCH:
    ret_image, azim_image = ret_image.numpy(), azim_image.numpy()

# Plot
colormap = 'viridis'
plt.clf()
plt.rcParams['image.origin'] = 'lower'
plt.figure(figsize=(12,2.5))
plt.subplot(1,3,1)
plt.imshow(ret_image,cmap=colormap)
plt.colorbar(fraction=0.046, pad=0.04)
plt.title(F'Retardance {back_end}')
plt.subplot(1,3,2)
plt.imshow(np.rad2deg(azim_image), cmap=colormap)
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Azimuth')
ax = plt.subplot(1,3,3)
if azimuth_plot_type == 'lines':
    im = plot_birefringence_lines(ret_image, azim_image,cmap=colormap, line_color='white', ax=ax)
else:
    im = plot_birefringence_colorized(ret_image, azim_image, ax=ax)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.title('Ret+Azim')

plt.pause(0.2)
plt.show()
# plt.savefig(f'Forward_projection_off_axis_thickness03_deltan-01_{volume_type}_axial_offset_{volume_axial_offset}.pdf')
plt.pause(0.2)

# plt.show()

