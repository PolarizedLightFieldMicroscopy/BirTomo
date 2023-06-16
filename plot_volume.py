import time  # to measure ray tracing time
import matplotlib.pyplot as plt
from plotting_tools import plot_retardance_orientation
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import (
    BirefringentVolume,
    BirefringentRaytraceLFM,
    JonesMatrixGenerators,
)

backend = BackEnds.NUMPY
# if backend == BackEnds.PYTORCH:
#     import torch
#     torch.set_grad_enabled(False)


# Get optical parameters template
optical_info = BirefringentVolume.get_optical_info_template()
# Alter some of the optical parameters
optical_info["volume_shape"] = [15, 51, 51]
optical_info["axial_voxel_size_um"] = 1.0
optical_info["cube_voxels"] = True
optical_info["pixels_per_ml"] = 17
optical_info["n_micro_lenses"] = 1
optical_info["n_voxels_per_ml"] = 1


# loaded_volume = BirefringentVolume.init_from_file(
#     "objects/bundleY_SN.h5", backend, optical_info
# )
# loaded_volume = BirefringentVolume.init_from_file(
#     "objects/shell_JOSAA2022.h5", backend, optical_info
# )
loaded_volume = BirefringentVolume.init_from_file("objects/single_voxel.h5", backend, optical_info)
my_volume = loaded_volume

# import plotly

# plotly_figure = my_volume.plot_lines_plotly()
# plotly_figure = my_volume.plot_volume_plotly(optical_info, voxels_in=my_volume.get_delta_n(),
#                                               opacity=0.01, fig=plotly_figure)
# plotly_figure.show()


rays = BirefringentRaytraceLFM(backend=backend, optical_info=optical_info)



# mask = rays.get_volume_reachable_region()

startTime = time.time()
rays.compute_rays_geometry()
# new_vol = rays.pad_volume(my_volume)
# mask = rays.get_volume_reachable_region()
executionTime = (time.time() - startTime)
print('Ray-tracing time in seconds: ' + str(executionTime))

new_vol = rays.create_reachable_volume_region(my_volume)
# new_vol = my_volume


startTime = time.time()
ret_image, azim_image = rays.ray_trace_through_volume(new_vol)
executionTime = (time.time() - startTime)
print(f'Execution time in seconds with backend {backend}: ' + str(executionTime))

# Plot retardance and orientation images
azimuth_plot_type = 'hsv'
azimuth_plot_type = 'lines'
my_fig = plot_retardance_orientation(ret_image, azim_image, azimuth_plot_type)
plt.pause(0.2)
plt.show(block=True)