"""Main script to run forward projection
This script using numpy/pytorch back-end to:
    - Create a volume with different birefringent shapes.
    - Compute the ray geometry depending on the Light field microscope and volume configuration.
    - Traverse the rays through the volume.
    - Compute the retardance and azimuth for every ray.
    - Generate 2D images.
"""
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
    torch.set_grad_enabled(False)


# Get optical parameters template
optical_info = BirefringentVolume.get_optical_info_template()
# Alter some of the optical parameters
optical_info['volume_shape'] = [15, 51, 51]
optical_info['axial_voxel_size_um'] = 1.0
optical_info['pixels_per_ml'] = 17
optical_info['n_micro_lenses'] = 5
optical_info['n_voxels_per_ml'] = 1
# Create non-identity polarizers and analyzers
# LC-PolScope setup
# optical_info['polarizer'] = JonesMatrixGenerators.polscope_analyzer()
# optical_info['analyzer'] = JonesMatrixGenerators.universal_compensator_modes(setting=0, swing=0)


# Volume type
# number is the shift from the end of the volume, change it as you wish,
#       do single_voxel{volume_shape[0]//2} for a voxel in the center
shift_from_center = 0
volume_axial_offset = optical_info['volume_shape'][0] // 2 + shift_from_center # for center
# volume_type = 'ellipsoid'
volume_type = 'shell'
# volume_type = '2ellipsoids'
# volume_type = 'single_voxel'

# Plot azimuth
# azimuth_plot_type = 'lines'
azimuth_plot_type = 'hsv'


# Different treatment to visualize single voxel
if volume_type == 'single_voxel':
    optical_info['n_micro_lenses'] = 1
    azimuth_plot_type = 'lines'


# Create a Birefringent Raytracer
rays = BirefringentRaytraceLFM(backend=backend, optical_info=optical_info)

# Compute the rays and use the Siddon's algorithm to compute the intersections with voxels.
# If a filepath is passed as argument, the object with all its calculations
#   get stored/loaded from a file
startTime = time.time()
rays.compute_rays_geometry()
executionTime = (time.time() - startTime)
print('Ray-tracing time in seconds: ' + str(executionTime))

# Move ray tracer to GPU
if backend == BackEnds.PYTORCH:
    # Disable gradients
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using computing device: {device}')
    rays = rays.to(device)

# Load volume from a file
# loaded_volume = BirefringentVolume.init_from_file("objects/bundleY.h5", backend, optical_info)

# Create a volume
my_volume = BirefringentVolume.create_dummy_volume(backend=backend, optical_info=optical_info, \
                                                    vol_type=volume_type, \
                                                    volume_axial_offset=volume_axial_offset)

# Plot ray geometry
# rays.plot_rays()

# Plot the volume
plotly_figure = my_volume.plot_lines_plotly()
# Append volumes to plot
plotly_figure = my_volume.plot_volume_plotly(optical_info, voxels_in=my_volume.get_delta_n(), opacity=0.01, fig=plotly_figure)
plotly_figure.show()

startTime = time.time()
ret_image, azim_image = rays.ray_trace_through_volume(my_volume)
executionTime = (time.time() - startTime)
print(f'Execution time in seconds with backend {backend}: ' + str(executionTime))

if backend == BackEnds.PYTORCH:
    ret_image, azim_image = ret_image.detach().cpu().numpy(), azim_image.detach().cpu().numpy()

# Plot
colormap = 'viridis'
plt.rcParams['image.origin'] = 'lower'
plt.figure(figsize=(12,2.5))
plt.subplot(1,3,1)
plt.imshow(ret_image,cmap=colormap)
plt.colorbar(fraction=0.046, pad=0.04)
plt.title(F'Retardance {backend}')
plt.subplot(1,3,2)
plt.imshow(np.rad2deg(azim_image), cmap=colormap)
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Azimuth')
ax = plt.subplot(1,3,3)
if azimuth_plot_type == 'lines':
    im = plot_birefringence_lines(ret_image, azim_image,cmap=colormap, line_color='white', ax=ax)
else:
    plot_birefringence_colorized(ret_image, azim_image)
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Ret+Azim')

plt.pause(0.2)
plt.show(block=True)
# plt.savefig(f'Forward_projection_off_axis_thickness03_deltan-01_{volume_type}_axial_offset_{volume_axial_offset}.pdf')
# plt.pause(0.2)
