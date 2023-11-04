"""Script to run forward intensity projection
This script using numpy back-end to:
    - Compute the ray geometry depending on the Light field microscope and volume configuration.
    - Traverse the rays through the volume.
    - Compute the intensity measurements for every ray fot various polarization settings.
    - Generate 2D images.
"""
import time         # to measure ray tracing time
import matplotlib.pyplot as plt
from VolumeRaytraceLFM.visualization.plotting_intensity import plot_intensity_images
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import  (
    BirefringentVolume,
    BirefringentRaytraceLFM,
    JonesMatrixGenerators
)

# Select backend method
backend = BackEnds.NUMPY
# backend = BackEnds.PYTORCH

# Get optical parameters template
optical_info = BirefringentVolume.get_optical_info_template()
# Alter some of the optical parameters
optical_info['volume_shape'] = [15, 51, 51]
optical_info['axial_voxel_size_um'] = 1.0
optical_info['cube_voxels'] = True
optical_info['pixels_per_ml'] = 17
optical_info['n_micro_lenses'] = 9
optical_info['n_voxels_per_ml'] = 1
# Create non-identity polarizers and analyzers
# LC-PolScope setup
optical_info['analyzer'] = JonesMatrixGenerators.left_circular_polarizer()
optical_info['polarizer_swing'] = 0.03

# number is the shift from the end of the volume, change it as you wish,
#       do single_voxel{volume_shape[0]//2} for a voxel in the center
shift_from_center = 0
volume_axial_offset = optical_info['volume_shape'][0] // 2 + shift_from_center # for center

# Create a Birefringent Raytracer!
rays = BirefringentRaytraceLFM(backend=backend, optical_info=optical_info)

# Compute the rays and use the Siddon's algorithm to compute the intersections with voxels.
# If a filepath is passed as argument, the object with all its calculations
#   get stored/loaded from a file
startTime = time.time()
rays.compute_rays_geometry()
executionTime = time.time() - startTime
print('Ray-tracing time in seconds: ' + str(executionTime))

# Load volume from a file
loaded_volume = BirefringentVolume.init_from_file("objects/single_voxel.h5", backend, optical_info)
loaded_volume = BirefringentVolume.init_from_file("objects/shell.h5", backend, optical_info)
my_volume = loaded_volume

image_list = rays.ray_trace_through_volume(my_volume, intensity=True)
executionTime = time.time() - startTime
print(f'Execution time in seconds with backend {backend}: ' + str(executionTime))

if backend==BackEnds.PYTORCH:
    image_list = [img.detach().cpu().numpy() for img in image_list]
my_fig = plot_intensity_images(image_list)
plt.pause(0.2)
plt.show(block=True)
