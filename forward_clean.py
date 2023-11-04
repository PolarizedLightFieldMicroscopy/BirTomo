''' Temporary script to generate forward projections.
- modular version of main_forward_projection.py
'''
import os
import torch
import matplotlib.pyplot as plt
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume, BirefringentRaytraceLFM
from VolumeRaytraceLFM.visualization.plotting_ret_azim import plot_retardance_orientation
from VolumeRaytraceLFM.ray import setup_raytracer

SAVE_FORWARD_IMAGES = True
backend = BackEnds.PYTORCH
shift_from_center = 0
DIR_POSTFIX = 'mla7'
volume_type = 'sphere_plot'
# volume_type = '1planes'
# volume_type = 'single_voxel'
sphere_args = {
    'init_mode' : 'ellipsoid',
    'init_args' : {
    'radius' : [4.5, 4.5, 4.5],
    'center' : [0.5, 0.5, 0.5],
    'delta_n' : -0.01,
    'border_thickness' : 1
        }
    }

plane_args = {
    'init_mode' : '1planes',
    'init_args' : {
        }
    }

voxel_args = {
    'init_mode' : 'single_voxel',
    'init_args' : {
        'delta_n' : -0.05,
        'offset' : [0, 0, 0]
        }
    }

DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

def setup_optical_parameters():
    """Setup optical parameters."""
    optical_info = BirefringentVolume.get_optical_info_template()
    optical_info['volume_shape'] = [11, 51, 51]
    optical_info['axial_voxel_size_um'] = 1.0
    optical_info['pixels_per_ml'] = 17
    optical_info['n_micro_lenses'] = 7
    optical_info['n_voxels_per_ml'] = 1
    return optical_info

def forward_model(volume_GT, rays, savedir):
    """Compute output of forward model."""
    with torch.no_grad():
        ret_image_measured, azim_image_measured = rays.ray_trace_through_volume(volume_GT)
        torch.save(ret_image_measured, os.path.join(savedir, 'ret_image_measured.pt'))
        torch.save(azim_image_measured, os.path.join(savedir, 'azim_image_measured.pt'))
        volume_GT.save_as_file(os.path.join(savedir, 'volume_gt.h5'))
    return ret_image_measured, azim_image_measured

def forward_model_handling(volume_type, optical_info, rays):
    """Handle forward image processing."""
    base_dir = 'forward_images'
    forward_img_dir = (f'{volume_type}_{optical_info["volume_shape"][0]}'
                       f'x{optical_info["volume_shape"][1]}x{optical_info["volume_shape"][2]}_{DIR_POSTFIX}')
    path = os.path.join(base_dir, 'Oct13', forward_img_dir)

    if SAVE_FORWARD_IMAGES:
        os.makedirs(path, exist_ok=True)
        volume_axial_offset = optical_info['volume_shape'][0] // 2 + shift_from_center # for center
        volume_GT = BirefringentVolume.create_dummy_volume(backend=backend, optical_info=optical_info,
                                                           vol_type=volume_type, volume_axial_offset=volume_axial_offset).to(DEVICE)

        with torch.no_grad():
            ret_image_measured, azim_image_measured = rays.ray_trace_through_volume(volume_GT)
            torch.save(ret_image_measured, os.path.join(path, 'ret_image_measured.pt'))
            torch.save(azim_image_measured, os.path.join(path, 'azim_image_measured.pt'))
            volume_GT.save_as_file(os.path.join(path, 'volume_gt.h5'))
    else:
        ret_image_measured = torch.load(os.path.join(path, 'ret_image_measured.pt'))
        azim_image_measured = torch.load(os.path.join(path, 'azim_image_measured.pt'))
        volume_GT = BirefringentVolume.init_from_file(os.path.join(path, 'volume_gt.h5'), backend=backend, optical_info=optical_info)
    return volume_GT, ret_image_measured, azim_image_measured


def load_volume_from_file(filename, optical_info):
    # Loading the reconstructed volume
    volume_recon = BirefringentVolume.init_from_file(
        filename,
        backend=backend,
        optical_info=optical_info
    )
    return volume_recon

def visualize_volume(volume_recon, optical_info):
    with torch.no_grad():
        plotly_figure = volume_recon.plot_lines_plotly()
        plotly_figure = volume_recon.plot_volume_plotly(optical_info, voxels_in=volume_recon.get_delta_n(), opacity=0.02, fig=plotly_figure)
        plotly_figure.show()
    return

def create_savedir(base_dir, sub_dir, optical_info):
    base_dir = 'forward_images'
    forward_img_dir = (f'{volume_type}_{optical_info["volume_shape"][0]}'
                       f'x{optical_info["volume_shape"][1]}x{optical_info["volume_shape"][2]}_{DIR_POSTFIX}')
    savedir = os.path.join(base_dir, sub_dir, forward_img_dir)
    os.makedirs(savedir, exist_ok=True)
    return savedir

def plot_volume(vol : BirefringentVolume):
    import numpy as np
    # Extract the birefringence and optic axis from the volume
    birefringence = vol.get_delta_n().detach().cpu().numpy()
    optic_axis = vol.get_optic_axis().detach().cpu().numpy()
    
    # For this example, pick the slice along the axial dimension at the voxel center
    axial_center = np.where(birefringence != 0)[0][0]
    birefringence_slice = birefringence[axial_center]
    optic_axis_slice = optic_axis[:, axial_center]

    # Plot birefringence
    plt.imshow(birefringence_slice, cmap='gray', interpolation='none')
    plt.colorbar(label='Birefringence')

    # Plot optic axis using a quiver plot
    # Here, we are plotting only non-zero optic axis values (i.e., the voxel center)
    x, y = np.where(birefringence_slice != 0)
    u = optic_axis_slice[0][x, y]
    v = optic_axis_slice[1][x, y]
    plt.quiver(y, x, u, v, scale=10, color='red')  # Note the swap of x and y due to imshow's coordinate system

    plt.title('Volume Slice at Axial Center')
    plt.show()


def main():
    optical_info = setup_optical_parameters()
    rays = setup_raytracer(optical_info)
    savedir = create_savedir('forward_images', 'Oct24', optical_info)
    # volume_axial_offset = optical_info['volume_shape'][0] // 2 + shift_from_center
    # volume_GT = BirefringentVolume.create_dummy_volume(backend=backend,
    #                                                    optical_info=optical_info,
    #                                                    vol_type=volume_type,
    #                                                    volume_axial_offset=volume_axial_offset
    #                                                    )
    volume_GT = BirefringentVolume(
        backend=backend,
        optical_info=optical_info,
        volume_creation_args=sphere_args
    )
    volume_GT = volume_GT.to(DEVICE)
    visualize_volume(volume_GT, optical_info)
    ret_image, azim_image = forward_model(volume_GT, rays, savedir)
    # Plot retardance and orientation images
    my_fig = plot_retardance_orientation(ret_image.cpu().numpy(), azim_image.cpu().numpy(), 'hsv', include_labels=True)
    plt.pause(0.2)
    plt.show(block=True)
    my_fig.savefig(savedir + '/ret_azim.png', bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    main()
