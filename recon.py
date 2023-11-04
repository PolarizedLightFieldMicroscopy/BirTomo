"""Temporary script to run 3D reconstruction
- modular version of main_3d_reconstruction.py
- includes forward projection
"""
import os
import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume, BirefringentRaytraceLFM
from VolumeRaytraceLFM.visualization.plotting_volume import volume_2_projections
from VolumeRaytraceLFM.visualization.plotting_iterations import plot_iteration_update
from VolumeRaytraceLFM.visualization.plotting_ret_azim import plot_retardance_orientation
from VolumeRaytraceLFM.volumes import volume_args
# from VolumeRaytraceLFM.ray import setup_raytracer
# from VolumeRaytraceLFM.visualization_utils import handle_visualization_and_saving

SAVE_FORWARD_IMAGES = True
SESSION_DIR = 'Oct27'
backend = BackEnds.PYTORCH
# shift_from_center = -1

DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

# DIR_POSTFIX = 'mla25_TVregaxisx'
DIR_POSTFIX = 'mla1'
VOL_NAME = 'voxel_optimzer'

vol_args = volume_args.voxel_args

def setup_optical_parameters():
    """Setup optical parameters."""
    optical_info = BirefringentVolume.get_optical_info_template()
    optical_info['volume_shape'] = [1, 3, 3]
    optical_info['axial_voxel_size_um'] = 1.0
    optical_info['pixels_per_ml'] = 17
    optical_info['n_micro_lenses'] = 1
    optical_info['n_voxels_per_ml'] = 1
    return optical_info

def setup_training_parameters():
    """Setup training parameters."""
    return {
        'n_epochs': 11,
        'azimuth_weight': .5,
        'regularization_weight': 0.1,
        'lr': 1e-3,
        'output_posfix': ''
    }

def setup_raytracer(optical_info):
    """Initialize Birefringent Raytracer."""
    print(f'For raytracing, using computing device: cpu')
    rays = BirefringentRaytraceLFM(backend=backend, optical_info=optical_info).to('cpu')
    start_time = time.time()
    rays.compute_rays_geometry()
    print(f'Ray-tracing time in seconds: {time.time() - start_time}')
    return rays.to(DEVICE)

def forward_model(volume_GT, rays, savedir):
    """Compute output of forward model."""
    with torch.no_grad():
        ret_image_measured, azim_image_measured = rays.ray_trace_through_volume(volume_GT)
        torch.save(ret_image_measured, os.path.join(savedir, 'ret_image_measured.pt'))
        torch.save(azim_image_measured, os.path.join(savedir, 'azim_image_measured.pt'))
        volume_GT.save_as_file(os.path.join(savedir, 'volume_gt.h5'))
    return ret_image_measured, azim_image_measured

def forward_image_handling(volume_type, optical_info, training_params, rays):
    """Handle forward image processing."""
    base_dir = 'forward_images'
    forward_img_dir = (f'{volume_type}_{optical_info["volume_shape"][0]}'
                       f'x{optical_info["volume_shape"][1]}x{optical_info["volume_shape"][2]}_{DIR_POSTFIX}')
    path = os.path.join(base_dir, forward_img_dir)

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

def setup_volume_estimation(optical_info, rays):
    """Setup volume estimation."""
    volume_estimation = BirefringentVolume(backend=backend,
                                           optical_info=optical_info,
                                           volume_creation_args = volume_args.random_args
                                           )
    # Let's rescale the random to initialize the volume
    volume_estimation.Delta_n.requires_grad = False
    volume_estimation.optic_axis.requires_grad = False
    # volume_estimation.Delta_n *= -0.01
    # # And mask out volume that is outside FOV of the microscope
    # mask = rays.get_volume_reachable_region()
    # volume_estimation.Delta_n[mask.view(-1)==0] = 0
    volume_estimation.Delta_n.requires_grad = True
    volume_estimation.optic_axis.requires_grad = True
    # Indicate to this object that we are going to optimize Delta_n and optic_axis
    volume_estimation.members_to_learn.append('Delta_n')
    volume_estimation.members_to_learn.append('optic_axis')
    volume_estimation = volume_estimation.to(DEVICE)
    return volume_estimation

def optimizer_setup(volume_estimation, training_params):
    """Setup optimizer."""
    trainable_parameters = volume_estimation.get_trainable_variables()
    return torch.optim.Adam(trainable_parameters, lr=training_params['lr'])

def replace_nans(volume, ep):
    with torch.no_grad():
        num_nan_vecs = torch.sum(torch.isnan(volume.optic_axis[0, :]))
        if num_nan_vecs > 0:
            replacement_vecs = torch.nn.functional.normalize(torch.rand(3, int(num_nan_vecs)), p=2, dim=0)
            volume.optic_axis[:, torch.isnan(volume.optic_axis[0, :])] = replacement_vecs
            if ep == 0:
                print(f"Replaced {num_nan_vecs} NaN optic axis vectors with random unit vectors.")

def setup_visualization():
    plt.ion()
    figure = plt.figure(figsize=(18, 9))
    plt.rcParams['image.origin'] = 'lower'
    return figure

def compute_losses(co_gt, ca_gt, ret_image_current, azim_image_current, volume_estimation, training_params):
    # Compute data term loss
    co_pred, ca_pred = ret_image_current * torch.cos(azim_image_current), ret_image_current * torch.sin(azim_image_current)
    data_term = ((co_gt - co_pred) ** 2 + (ca_gt - ca_pred) ** 2).mean()

    # Compute regularization term
    delta_n = volume_estimation.get_delta_n()
    TV_reg = (
        (delta_n[1:, ...] - delta_n[:-1, ...]).pow(2).sum() +
        (delta_n[:, 1:, ...] - delta_n[:, :-1, ...]).pow(2).sum() +
        (delta_n[:, :, 1:] - delta_n[:, :, :-1]).pow(2).sum()
    )
    axis_x = volume_estimation.get_optic_axis()[0, ...]
    TV_reg_axis_x = (
        (axis_x[1:, ...] - axis_x[:-1, ...]).pow(2).sum() +
        (axis_x[:, 1:, ...] - axis_x[:, :-1, ...]).pow(2).sum() +
        (axis_x[:, :, 1:] - axis_x[:, :, :-1]).pow(2).sum()
    )
    # regularization_term = TV_reg + 1000 * (volume_estimation.Delta_n ** 2).mean() + TV_reg_axis_x / 100000
    regularization_term = TV_reg + 1000 * (volume_estimation.Delta_n ** 2).mean()

    # Total loss
    L = data_term + training_params['regularization_weight'] * regularization_term
    return L, data_term, regularization_term

def one_iteration(co_gt, ca_gt, optimizer, rays, volume_estimation, training_params):
    optimizer.zero_grad()

    # Forward project
    [ret_image_current, azim_image_current] = rays.ray_trace_through_volume(volume_estimation)
    L, data_term, regularization_term = compute_losses(co_gt, ca_gt, ret_image_current, azim_image_current, volume_estimation, training_params)

    L.backward()
    optimizer.step()

    return L.item(), data_term.item(), regularization_term.item(), ret_image_current, azim_image_current

def handle_visualization_and_saving(ep, Delta_n_GT, ret_image_measured, azim_image_measured, volume_estimation, figure, output_dir, ret_image_current, azim_image_current, losses, data_term_losses, regularization_term_losses):
    if ep % 1 == 0:
        plt.clf()
        plot_iteration_update(
            volume_2_projections(Delta_n_GT.unsqueeze(0))[0, 0].detach().cpu().numpy(),
            ret_image_measured.detach().cpu().numpy(),
            azim_image_measured.detach().cpu().numpy(),
            volume_2_projections(volume_estimation.get_delta_n().unsqueeze(0))[0, 0].detach().cpu().numpy(),
                ret_image_current.detach().cpu().numpy(),
                azim_image_current.detach().cpu().numpy(),
                losses,
                data_term_losses,
                regularization_term_losses
        )
        figure.canvas.draw()
        figure.canvas.flush_events()
        time.sleep(0.1)
        plt.savefig(f"{output_dir}/Optimization_ep_{'{:02d}'.format(ep)}.pdf")
        time.sleep(0.1)

    if ep % 100 == 0:
        volume_estimation.save_as_file(f"{output_dir}/volume_ep_{'{:02d}'.format(ep)}.h5")
    return

def reconstruct(ret_image_measured, azim_image_measured, training_params, optimizer, rays, Delta_n_GT, volume_estimation, figure, output_dir):
    # Vector difference GT
    co_gt, ca_gt = ret_image_measured * torch.cos(azim_image_measured), ret_image_measured * torch.sin(azim_image_measured)

    # Lists to store losses
    losses = []
    data_term_losses = []
    regularization_term_losses = []

    # Training loop
    for ep in tqdm(range(training_params['n_epochs']), "Minimizing"):
        loss, data_term_loss, regularization_term_loss, ret_image_current, azim_image_current = one_iteration(co_gt, ca_gt, optimizer, rays, volume_estimation, training_params)

        # Record losses
        losses.append(loss)
        data_term_losses.append(data_term_loss)
        regularization_term_losses.append(regularization_term_loss)

        # Visualization and saving
        handle_visualization_and_saving(ep, Delta_n_GT, ret_image_measured, azim_image_measured,
                                        volume_estimation, figure, output_dir, ret_image_current,
                                        azim_image_current,
                                        losses, data_term_losses, regularization_term_losses
                                        )
    # Final visualizations after training completes
    plt.savefig(f"{output_dir}/Optimization_final.pdf")
    plt.show()
    # Save final volume
    volume_estimation.save_as_file(f"{output_dir}/volume_final_ep_{'{:02d}'.format(ep)}.h5")
    return volume_estimation

def create_savedir(base_dir, sub_dir, optical_info, VOL_NAME=VOL_NAME):
    base_dir = 'forward_images'
    forward_img_dir = (f'{VOL_NAME}_{optical_info["volume_shape"][0]}'
                       f'x{optical_info["volume_shape"][1]}x{optical_info["volume_shape"][2]}_{DIR_POSTFIX}')
    savedir = os.path.join(base_dir, sub_dir, forward_img_dir)
    os.makedirs(savedir, exist_ok=True)
    return savedir

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

def execute_pipeline(vol_args, 
                     SESSION_DIR=SESSION_DIR, 
                     VOL_NAME=VOL_NAME, 
                     DIR_POSTFIX=DIR_POSTFIX):
    optical_info = setup_optical_parameters()
    training_params = setup_training_parameters()
    rays = setup_raytracer(optical_info)
    savedir = create_savedir('forward_images', SESSION_DIR, optical_info, VOL_NAME=VOL_NAME)
    # volume_type = 'shell'  # Define this as needed
    volume_GT = BirefringentVolume(
        backend=backend,
        optical_info=optical_info,
        volume_creation_args=vol_args
    )
    volume_GT = volume_GT.to(DEVICE)
    visualize_volume(volume_GT, optical_info)
    ret_image_meas, azim_image_meas = forward_model(volume_GT, rays, savedir)
    my_fig = plot_retardance_orientation(ret_image_meas.cpu().numpy(), azim_image_meas.cpu().numpy(), 'hsv', include_labels=True)
    # plt.pause(0.2)
    # plt.show(block=True)
    my_fig.tight_layout()
    my_fig.savefig(savedir + '/ret_azim.png', dpi=300)
    Delta_n_GT = volume_GT.get_delta_n().detach().clone()
    volume_estimation = setup_volume_estimation(optical_info, rays)
    optimizer = optimizer_setup(volume_estimation, training_params)

    # Starting the reconstruction
    output_dir = f'reconstructions/{SESSION_DIR}/{VOL_NAME}_{optical_info["volume_shape"][0]}' \
                + f'x{optical_info["volume_shape"][1]}x{optical_info["volume_shape"][2]}_{DIR_POSTFIX}'
    os.makedirs(output_dir, exist_ok=True)
    torch.save({'optical_info' : optical_info,
            'training_params' : training_params,
            'vol_args' : vol_args}, f'{output_dir}/parameters.pt')

    # Initialize visualization settings and figures
    figure = setup_visualization()

    volume_recon = reconstruct(ret_image_meas, azim_image_meas, training_params, optimizer, rays, Delta_n_GT, volume_estimation, figure, output_dir)
    visualize_volume(volume_recon, optical_info)
    print(f"Reconstructed {VOL_NAME}")


def main():
    optical_info = setup_optical_parameters()
    training_params = setup_training_parameters()
    rays = setup_raytracer(optical_info)
    savedir = create_savedir('forward_images', SESSION_DIR, optical_info)
    # volume_type = 'shell'  # Define this as needed
    volume_GT = BirefringentVolume(
        backend=backend,
        optical_info=optical_info,
        volume_creation_args=vol_args
    )
    # with torch.no_grad():
    #     volume_GT.get_delta_n()[:optical_info['volume_shape'][0] // 2 + 2,...] = 0
    volume_GT = volume_GT.to(DEVICE)
    # visualize_volume(volume_GT, optical_info)
    ret_image_meas, azim_image_meas = forward_model(volume_GT, rays, savedir)
    my_fig = plot_retardance_orientation(ret_image_meas.cpu().numpy(), azim_image_meas.cpu().numpy(), 'hsv', include_labels=True)
    # plt.pause(0.2)
    # plt.show(block=True)
    my_fig.tight_layout()
    my_fig.savefig(savedir + '/ret_azim.png', dpi=300)
    Delta_n_GT = volume_GT.get_delta_n().detach().clone()
    volume_estimation = setup_volume_estimation(optical_info, rays)
    optimizer = optimizer_setup(volume_estimation, training_params)

    # Starting the reconstruction
    output_dir = f'reconstructions/{SESSION_DIR}/{VOL_NAME}_{optical_info["volume_shape"][0]}' \
                + f'x{optical_info["volume_shape"][1]}x{optical_info["volume_shape"][2]}_{DIR_POSTFIX}'
    os.makedirs(output_dir, exist_ok=True)
    torch.save({'optical_info' : optical_info,
            'training_params' : training_params,
            'vol_args' : vol_args}, f'{output_dir}/parameters.pt')

    # Initialize visualization settings and figures
    figure = setup_visualization()

    volume_recon = reconstruct(ret_image_meas, azim_image_meas, training_params, optimizer, rays, Delta_n_GT, volume_estimation, figure, output_dir)
    visualize_volume(volume_recon, optical_info)

if __name__ == "__main__":
    main()
    # params_sets = [
    #     {"vol_args": volume_args.sphere_args2, "VOL_NAME": "sphere2pos"},
    #     {"vol_args": volume_args.sphere_args3, "VOL_NAME": "sphere3pos"},
    #     {"vol_args": volume_args.sphere_args4, "VOL_NAME": "sphere4pos"},
    #     {"vol_args": volume_args.sphere_args5, "VOL_NAME": "sphere5pos"},
    #     {"vol_args": volume_args.sphere_args6, "VOL_NAME": "sphere6pos"},
    # ]

    # params_sets = [
    #     {"vol_args": volume_args.sphere_args2, "VOL_NAME": "sphere2pos"},
    #     {"vol_args": volume_args.sphere_args3, "VOL_NAME": "sphere3pos"},
    #     {"vol_args": volume_args.sphere_args4, "VOL_NAME": "sphere4pos"},
    #     {"vol_args": volume_args.sphere_args5, "VOL_NAME": "sphere5pos"},
    #     {"vol_args": volume_args.sphere_args6, "VOL_NAME": "sphere6pos"},
    # ]
    # for params in params_sets:
    #     execute_pipeline(**params)
