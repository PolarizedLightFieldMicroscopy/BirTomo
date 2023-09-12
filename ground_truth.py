"""Generate volumes using forward projection
This script using numpy/pytorch back-end to:
    - Create a volume with different birefringent shapes.
    - Compute the ray geometry depending on the Light field microscope and volume configuration.
    - Traverse the rays through the volume.
    - Compute the retardance and azimuth for every ray.
    - Generate 2D images.
"""
import time         # to measure ray tracing time
import numpy as np
from tifffile import imsave, imread
import matplotlib.pyplot as plt
from plotting_tools import plot_retardance_orientation
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import  (
    BirefringentVolume,
    BirefringentRaytraceLFM
)

# Select backend method
# backend = BackEnds.PYTORCH
backend = BackEnds.NUMPY

if backend == BackEnds.PYTORCH:
    import torch
    torch.set_grad_enabled(False)


# Get optical parameters template
optical_info = BirefringentVolume.get_optical_info_template()
# Alter some of the optical parameters
# optical_info['volume_shape'] = [15, 51, 51]
optical_info['volume_shape'] = [45, 153, 153]
optical_info['axial_voxel_size_um'] = 1.0
optical_info['cube_voxels'] = True
optical_info['pixels_per_ml'] = 17
optical_info['n_micro_lenses'] = 3
optical_info['n_voxels_per_ml'] = 3


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

volume_shape = optical_info['volume_shape']

def create_rays(n_mla, n_pix, ss, vol_shape):
    optical_info['n_micro_lenses'] = n_mla
    optical_info['pixels_per_ml'] = n_pix
    optical_info['n_voxels_per_ml'] = ss
    optical_info['volume_shape'] = vol_shape
    rays = BirefringentRaytraceLFM(backend=backend, optical_info=optical_info)
    rays.compute_rays_geometry()
    return rays

def single_shell():
    my_volume = BirefringentVolume.create_dummy_volume(backend=backend, optical_info=optical_info,
                                                       vol_type=volume_type,
                                                       volume_axial_offset=volume_axial_offset)
    plotly_figure = my_volume.plot_lines_plotly()
    plotly_figure = my_volume.plot_volume_plotly(optical_info, voxels_in=my_volume.get_delta_n(),
                                                  opacity=0.01, fig=plotly_figure)
    plotly_figure.show()

    my_volume.save_as_tiff('raw/shell.tiff')
    [ret_image, azim_image] = rays.ray_trace_through_volume(my_volume)
    combined_data = np.stack([ret_image, azim_image], axis=0, dtype=np.float32)
    imsave('raw/image.tiff', combined_data)

# startTime = time.time()

def single_ellipse(rays, i):
    ellip_volume = BirefringentVolume.create_dummy_volume(backend=BackEnds.NUMPY, optical_info=optical_info,
                                                    vol_type='ellipsoids_random',
                                                    volume_axial_offset=volume_axial_offset)
    ellip_volume.save_as_tiff(f'raw/objects/{i}_ell.tiff')
    plot_volume = False
    if plot_volume:
        plotly_figure = ellip_volume.plot_lines_plotly()
        plotly_figure = ellip_volume.plot_volume_plotly(optical_info, voxels_in=ellip_volume.get_delta_n(),
                                                    opacity=0.01, fig=plotly_figure)
        plotly_figure.show()

    [ret_image, azim_image] = rays.ray_trace_through_volume(ellip_volume)
    combined_data = np.stack([ret_image, azim_image], axis=0, dtype=np.float32)
    imsave(f'raw/images/{i}_ell.tiff', combined_data)

def single_sphere(rays, i):
    ellip_volume = BirefringentVolume.create_dummy_volume(backend=BackEnds.NUMPY, optical_info=optical_info,
                                                    vol_type='sphere',
                                                    volume_axial_offset=volume_axial_offset)
    ellip_volume.save_as_tiff(f'sphere/{i}_sphere.tiff')
    plot_volume = False
    if plot_volume:
        plotly_figure = ellip_volume.plot_lines_plotly()
        plotly_figure = ellip_volume.plot_volume_plotly(optical_info, voxels_in=ellip_volume.get_delta_n(),
                                                    opacity=0.01, fig=plotly_figure)
        plotly_figure.show()

    [ret_image, azim_image] = rays.ray_trace_through_volume(ellip_volume)
    combined_data = np.stack([ret_image, azim_image], axis=0, dtype=np.float32)
    imsave(f'sphere/{i}_sphere.tiff', combined_data)

def read_plot_vol_tiff(filename, axial=0):
    # Read the TIFF file
    image = imread(filename)

    # Check if image has 4 channels
    if image.shape[0] != 4:
        raise ValueError("The image does not have 4 channels!")

    # Plot each channel
    fig, axarr = plt.subplots(1, 4, figsize=(20, 5))

    channel_names = ['birefringence', 'optic axis 1', 'optic axis 2', 'optic axis 3']
    for i in range(4):
        axarr[i].imshow(image[i, axial, ...], cmap='gray')
        axarr[i].axis('off')
        axarr[i].set_title(channel_names[i])

    plt.tight_layout()
    plt.show()    
    
def read_plot_img_tiff(filename):
    # Read the TIFF file
    image = imread(filename)

    # Check if image has 2 channels
    if image.shape[0] != 2:
        raise ValueError("The image does not have 2 channels!")

    # Plot each channel
    fig, axarr = plt.subplots(1, 2, figsize=(10, 5))

    channel_names = ['retardance', 'orientation']
    for i in range(2):
        axarr[i].imshow(image[i, ...], cmap='gray')
        axarr[i].axis('off')
        axarr[i].set_title(channel_names[i])

    plt.tight_layout()
    plt.show()

def single_small_sphere(rays, i):
    ellip_volume = BirefringentVolume.create_dummy_volume(backend=BackEnds.NUMPY, optical_info=optical_info,
                                                    vol_type='small_sphere',
                                                    volume_axial_offset=volume_axial_offset)
    ellip_volume.save_as_tiff(f'small_sphere/objects/{i}_sphere.tiff')
    plot_volume = False
    if plot_volume:
        plotly_figure = ellip_volume.plot_lines_plotly()
        plotly_figure = ellip_volume.plot_volume_plotly(optical_info, voxels_in=ellip_volume.get_delta_n(),
                                                    opacity=0.01, fig=plotly_figure)
        plotly_figure.show()

    [ret_image, azim_image] = rays.ray_trace_through_volume(ellip_volume)
    combined_data = np.stack([ret_image, azim_image], axis=0, dtype=np.float32)
    imsave(f'small_sphere/images/{i}_sphere.tiff', combined_data)

if __name__ == "__main__":
    # rays = create_rays(11, 3)
    # for i in range(1):
    #     single_ellipse(rays, i)


    # vol_shape = [4, 8, 8]
    # rays = create_rays(4, 8, 1, vol_shape)
    # for i in range(1):
    #     single_small_sphere(rays, i)

    read_plot_vol_tiff('raw/objects/6_ell.tiff')
    read_plot_img_tiff('raw/images/6_ell.tiff')
