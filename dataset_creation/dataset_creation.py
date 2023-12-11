'''Creates a dataset of simulated objects and their corresponding images.'''
from abstract_classes import BackEnds
from birefringence_implementations import  (
    BirefringentVolume,
    BirefringentRaytraceLFM
)
# from VolumeRaytraceLFM.visualization.plotting_ret_azim import plot_retardance_orientation
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imsave
import os


optical_info = BirefringentVolume.get_optical_info_template()
# optical_info['pixels_per_ml'] = 8
# optical_info['n_micro_lenses'] = 4
# optical_info['volume_shape'] = [4, 8, 8]
optical_info['pixels_per_ml'] = 16
optical_info['n_micro_lenses'] = 16
optical_info['volume_shape'] = [8, 32, 32]
volume_axial_offset = 0
# backend = BackEnds.PYTORCH
backend = BackEnds.NUMPY
plot_volume = False


sphere_volume = BirefringentVolume.create_dummy_volume(backend=backend, optical_info=optical_info,
                                                vol_type='small_sphere_pos',
                                                volume_axial_offset=volume_axial_offset)
if plot_volume:
    plotly_figure = sphere_volume.plot_lines_plotly()
    plotly_figure = sphere_volume.plot_volume_plotly(optical_info, voxels_in=sphere_volume.get_delta_n(),
                                                opacity=0.01, fig=plotly_figure)
    plotly_figure.show()


optical_info['n_voxels_per_ml'] = 1
rays = BirefringentRaytraceLFM(backend=backend, optical_info=optical_info)
rays.compute_rays_geometry()
# [ret_image, azim_image] = rays.ray_trace_through_volume(sphere_volume)
# combined_data = np.stack([ret_image, azim_image], axis=0, dtype=np.float32)
# plot_retardance_orientation(ret_image, azim_image)
# plt.pause(0.2)
# plt.show(block=True)

def single_small_sphere_pos(rays, i):
    sphere_volume = BirefringentVolume.create_dummy_volume(backend=backend, optical_info=optical_info,
                                                vol_type='small_sphere_pos',
                                                volume_axial_offset=volume_axial_offset)
    idx_str = str(i).zfill(4)
    obj_filename = 'small_sphere1000/objects/' + idx_str + '_sphere.tiff'
    sphere_volume.save_as_tiff(obj_filename)
    plot_volume = False
    if plot_volume:
        plotly_figure = sphere_volume.plot_lines_plotly()
        plotly_figure = sphere_volume.plot_volume_plotly(optical_info, voxels_in=sphere_volume.get_delta_n(),
                                                    opacity=0.01, fig=plotly_figure)
        plotly_figure.show()

    [ret_image, azim_image] = rays.ray_trace_through_volume(sphere_volume)
    combined_data = np.stack([ret_image, azim_image], axis=0, dtype=np.float32)
    img_filename = 'small_sphere1000/images/' + idx_str + '_sphere.tiff'
    imsave(img_filename, combined_data)
    
def single_small_sphere_random_bir(rays, i, save_dir):
    sphere_volume = BirefringentVolume.create_dummy_volume(backend=backend, optical_info=optical_info,
                                                vol_type='small_sphere_rand_bir',
                                                volume_axial_offset=volume_axial_offset)
    idx_str = str(i).zfill(4)
    obj_filename = os.path.join(save_dir, 'objects', idx_str + '_sphere.tiff')
    sphere_volume.save_as_tiff(obj_filename)
    plot_volume = False
    if plot_volume:
        plotly_figure = sphere_volume.plot_lines_plotly()
        plotly_figure = sphere_volume.plot_volume_plotly(optical_info, voxels_in=sphere_volume.get_delta_n(),
                                                    opacity=0.01, fig=plotly_figure)
        plotly_figure.show()

    [ret_image, azim_image] = rays.ray_trace_through_volume(sphere_volume)
    combined_data = np.stack([ret_image, azim_image], axis=0, dtype=np.float32)
    img_filename = os.path.join(save_dir, 'images', idx_str + '_sphere.tiff')
    imsave(img_filename, combined_data)


save_dir = 'dataset_tmp1/'
if not os.path.exists(save_dir):
    os.makedirs(os.path.join(save_dir, 'objects'))
    os.makedirs(os.path.join(save_dir, 'images'))
for i in range(1):
    single_small_sphere_random_bir(rays, i, save_dir=save_dir)

