from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import (
    BirefringentVolume,
    BirefringentRaytraceLFM,
)
from VolumeRaytraceLFM.visualization.plotting_ret_azim import (
    plot_retardance_orientation,
)
import numpy as np


optical_info = BirefringentVolume.get_optical_info_template()
# optical_info['pixels_per_ml'] = 8
# optical_info['n_micro_lenses'] = 4
# optical_info['volume_shape'] = [4, 8, 8]
optical_info["pixels_per_ml"] = 9
optical_info["n_micro_lenses"] = 27
optical_info["volume_shape"] = [9, 27, 27]
volume_axial_offset = 0
# backend = BackEnds.PYTORCH
backend = BackEnds.NUMPY
plot_volume = False


sphere_volume = BirefringentVolume.create_dummy_volume(
    backend=backend,
    optical_info=optical_info,
    vol_type="small_sphere",
    volume_axial_offset=volume_axial_offset,
)
if plot_volume:
    plotly_figure = sphere_volume.plot_lines_plotly()
    plotly_figure = sphere_volume.plot_volume_plotly(
        optical_info,
        voxels_in=sphere_volume.get_delta_n(),
        opacity=0.01,
        fig=plotly_figure,
    )
    plotly_figure.show()


optical_info["n_voxels_per_ml"] = 1
rays = BirefringentRaytraceLFM(backend=backend, optical_info=optical_info)
rays.compute_rays_geometry()
[ret_image, azim_image] = rays.ray_trace_through_volume(sphere_volume)
combined_data = np.stack([ret_image, azim_image], axis=0, dtype=np.float32)

plot_retardance_orientation(ret_image, azim_image)
