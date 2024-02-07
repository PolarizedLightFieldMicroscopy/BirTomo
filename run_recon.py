import os
import numpy as np
import torch
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.simulations import ForwardModel
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume
from VolumeRaytraceLFM.volumes import volume_args
from VolumeRaytraceLFM.setup_parameters import (
    setup_optical_parameters,
    setup_iteration_parameters
)
from VolumeRaytraceLFM.reconstructions import ReconstructionConfig, Reconstructor
from VolumeRaytraceLFM.visualization.plotting_volume import visualize_volume
from VolumeRaytraceLFM.utils.file_utils import (
    create_unique_directory,
    get_forward_img_str_postfix
)

BACKEND = BackEnds.PYTORCH
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def recon_gpu():
    '''Reconstruct a volume on the GPU.'''
    optical_info = setup_optical_parameters(
        "config_settings/optical_config3.json")
    optical_system = {'optical_info': optical_info}
    # Initialize the forward model. Raytracing is performed as part of the initialization.
    simulator = ForwardModel(optical_system, backend=BACKEND, device=DEVICE)
    simulator.to_device(DEVICE)  # Move the simulator to the GPU

    # Volume creation
    volume_GT = BirefringentVolume(
        backend=BACKEND,
        optical_info=optical_info,
        volume_creation_args=volume_args.ellipsoid_args2
    )
    volume_GT.to(DEVICE)  # Move the volume to the GPU

    visualize_volume(volume_GT, optical_info)
    simulator.forward_model(volume_GT)
    ret_image_meas = simulator.ret_img
    azim_image_meas = simulator.azim_img

    recon_optical_info = optical_info
    iteration_params = setup_iteration_parameters(
        "config_settings/iter_config.json")
    initial_volume = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=recon_optical_info,
        volume_creation_args=volume_args.random_args
    )
    initial_volume.to(DEVICE)  # Move the volume to the GPU

    recon_directory = create_unique_directory("reconstructions")
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas, azim_image_meas,
                                        initial_volume, iteration_params, gt_vol=volume_GT)
    recon_config.save(recon_directory)

    reconstructor = Reconstructor(recon_config, device=DEVICE)
    reconstructor.to_device(DEVICE)  # Move the reconstructor to the GPU

    reconstructor.reconstruct(output_dir=recon_directory)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def recon():
    optical_info = setup_optical_parameters(
        "config_settings/optical_config_voxel.json")
    simulate = False
    if simulate:
        optical_system = {'optical_info': optical_info}
        # Initialize the forward model. Raytracing is performed as part of the initialization.
        simulator = ForwardModel(optical_system, backend=BACKEND)
        # Volume creation
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.voxel_shifted_args #voxel_args
        )
        visualize_volume(volume_GT, optical_info)

        simulator.forward_model(volume_GT)
        simulator.view_images()
        ret_image_meas = simulator.ret_img
        azim_image_meas = simulator.azim_img
        # Save the images as numpy arrays
        if True:
            ret_numpy = ret_image_meas.detach().numpy()
            np.save('forward_images/ret_voxel_pos_1mla_17pix.npy', ret_numpy)
            azim_numpy = azim_image_meas.detach().numpy()
            np.save('forward_images/azim_voxel_pos_1mla_17pix.npy', azim_numpy)
    else:
        ret_image_meas = np.load(os.path.join(
            'forward_images', 'ret_voxel_pos_1mla_17pix.npy'))
        azim_image_meas = np.load(os.path.join(
            'forward_images', 'azim_voxel_pos_1mla_17pix.npy'))

    recon_optical_info = optical_info.copy()
    iteration_params = setup_iteration_parameters(
        "config_settings/iter_config.json")
    initial_volume = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=recon_optical_info,
        volume_creation_args=volume_args.random_args
    )
    recon_directory = create_unique_directory("reconstructions")
    if not simulate:
        volume_GT = initial_volume
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas,
        azim_image_meas, initial_volume, iteration_params, gt_vol=volume_GT
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(recon_config, omit_rays_based_on_pixels=True)
    reconstructor.rays.verbose = False
    reconstructor.reconstruct(output_dir=recon_directory)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def recon_sphere():
    optical_info = setup_optical_parameters(
        "config_settings/optical_config_sphere.json")
    postfix = get_forward_img_str_postfix(optical_info)
    forward_img_str = 'sphere6_thick1' + postfix + '.npy'
    simulate = True
    if simulate:
        optical_system = {'optical_info': optical_info}
        # Initialize the forward model. Raytracing is performed as part of the initialization.
        simulator = ForwardModel(optical_system, backend=BACKEND)
        # Volume creation
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.sphere_args6
        )
        # visualize_volume(volume_GT, optical_info)

        simulator.forward_model(volume_GT)
        simulator.view_images()
        ret_image_meas = simulator.ret_img
        azim_image_meas = simulator.azim_img
        # Save the images as numpy arrays
        if True:
            ret_numpy = ret_image_meas.detach().numpy()
            np.save('forward_images/ret_' + forward_img_str, ret_numpy)
            azim_numpy = azim_image_meas.detach().numpy()
            np.save('forward_images/azim_' + forward_img_str, azim_numpy)
    else:
        ret_image_meas = np.load(os.path.join(
            'forward_images', 'ret_' + forward_img_str))
        azim_image_meas = np.load(os.path.join(
            'forward_images', 'azim_' + forward_img_str))

    recon_optical_info = optical_info.copy()
    iteration_params = setup_iteration_parameters(
        "config_settings/iter_config_sphere_large_vol.json")
    initial_volume = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=recon_optical_info,
        volume_creation_args=volume_args.random_args
    )
    recon_directory = create_unique_directory("reconstructions")
    if not simulate:
        volume_GT = initial_volume
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas,
        azim_image_meas, initial_volume, iteration_params, gt_vol=volume_GT
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(recon_config, omit_rays_based_on_pixels=True)
    reconstructor.rays.verbose = False
    reconstructor.reconstruct(output_dir=recon_directory)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def recon_sphere_from_prev_try():
    optical_info = setup_optical_parameters(
        "config_settings/optical_config_sphere.json")
    postfix = get_forward_img_str_postfix(optical_info)
    forward_img_str = 'sphere6_thick1' + postfix + '.npy'
    simulate = True
    if simulate:
        optical_system = {'optical_info': optical_info}
        # Initialize the forward model. Raytracing is performed as part of the initialization.
        simulator = ForwardModel(optical_system, backend=BACKEND)
        # Volume creation
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.sphere_args6
        )
        # visualize_volume(volume_GT, optical_info)

        simulator.forward_model(volume_GT)
        simulator.view_images()
        ret_image_meas = simulator.ret_img
        azim_image_meas = simulator.azim_img
        # Save the images as numpy arrays
        if True:
            ret_numpy = ret_image_meas.detach().numpy()
            np.save('forward_images/ret_' + forward_img_str, ret_numpy)
            azim_numpy = azim_image_meas.detach().numpy()
            np.save('forward_images/azim_' + forward_img_str, azim_numpy)
    else:
        ret_image_meas = np.load(os.path.join(
            'forward_images', 'ret_' + forward_img_str))
        azim_image_meas = np.load(os.path.join(
            'forward_images', 'azim_' + forward_img_str))

    recon_optical_info = optical_info.copy()
    iteration_params = setup_iteration_parameters(
        "config_settings/iter_config_sphere.json")
    sphere_path = "volumes/2024-01-02_23-26-15/volume_ep_300.h5"
    initial_volume = BirefringentVolume.init_from_file(
        sphere_path, BackEnds.PYTORCH, recon_optical_info)
    recon_directory = create_unique_directory("reconstructions")
    if not simulate:
        volume_GT = initial_volume
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas,
        azim_image_meas, initial_volume, iteration_params, gt_vol=volume_GT
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(recon_config, omit_rays_based_on_pixels=True)
    reconstructor.rays.verbose = False
    reconstructor.reconstruct(output_dir=recon_directory)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def recon_voxel():
    optical_info = setup_optical_parameters(
        "config_settings/optical_config_voxel.json")
    postfix = get_forward_img_str_postfix(optical_info)
    forward_img_str = 'voxel_pos' + postfix + '.npy'
    simulate = False
    if simulate:
        optical_system = {'optical_info': optical_info}
        # Initialize the forward model. Raytracing is performed as part of the initialization.
        simulator = ForwardModel(optical_system, backend=BACKEND)
        # Volume creation
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.voxel_args
        )
        visualize_volume(volume_GT, optical_info)

        simulator.forward_model(volume_GT)
        simulator.view_images()
        ret_image_meas = simulator.ret_img
        azim_image_meas = simulator.azim_img
        # Save the images as numpy arrays
        if True:
            ret_numpy = ret_image_meas.detach().numpy()
            np.save('forward_images/ret_' + forward_img_str, ret_numpy)
            azim_numpy = azim_image_meas.detach().numpy()
            np.save('forward_images/azim_' + forward_img_str, azim_numpy)
    else:
        ret_image_meas = np.load(os.path.join(
            'forward_images', 'ret_' + forward_img_str))
        azim_image_meas = np.load(os.path.join(
            'forward_images', 'azim_' + forward_img_str))

    recon_optical_info = optical_info.copy()
    iteration_params = setup_iteration_parameters(
        "config_settings/iter_config.json")
    continue_recon = False
    if continue_recon:
        volume_path = "reconstructions/2024-01-04_16-31-34/volume_ep_100.h5"
        initial_volume = BirefringentVolume.init_from_file(
        volume_path, BackEnds.PYTORCH, recon_optical_info)
    else:
        initial_volume = BirefringentVolume(
            backend=BACKEND,
            optical_info=recon_optical_info,
            volume_creation_args=volume_args.random_args
        )
    recon_directory = create_unique_directory("reconstructions")
    if not simulate:
        volume_GT = initial_volume
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas,
        azim_image_meas, initial_volume, iteration_params, gt_vol=volume_GT
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(recon_config, omit_rays_based_on_pixels=True, apply_volume_mask=True)
    reconstructor.rays.store_shifted_vox_indices()
    reconstructor.rays.verbose = False
    reconstructor.reconstruct(output_dir=recon_directory)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def recon_continuation(shape='sphere', init_volume_path=None):
    if shape == 'sphere':
        optical_info = setup_optical_parameters(
            "config_settings/optical_config_sphere.json")
        postfix = get_forward_img_str_postfix(optical_info)
        forward_img_str = 'sphere6_thick1' + postfix + '.npy'
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.sphere_args6
        )
        iteration_params = setup_iteration_parameters(
            "config_settings/iter_config_sphere.json")
    elif shape == 'helix':
        optical_info = setup_optical_parameters(
            "config_settings/optical_config_helix_z7.json")
        postfix = get_forward_img_str_postfix(optical_info)
        forward_img_str = 'helix1' + postfix + '_7z.npy'
        helix_path = "objects/Helix/Helix1_resaved.h5"
        volume_GT = BirefringentVolume.init_from_file(
            helix_path, BackEnds.PYTORCH, optical_info)
        iteration_params = setup_iteration_parameters(
            "config_settings/iter_config_helix.json")
    else:
        raise ValueError(f"Unknown shape {shape}")

    iteration_params["notes"] = f"Continuation of previous reconstruction {init_volume_path}"
    # visualize_volume(volume_GT, optical_info)
    ret_image_meas = np.load(os.path.join(
        'forward_images', 'ret_' + forward_img_str))
    azim_image_meas = np.load(os.path.join(
        'forward_images', 'azim_' + forward_img_str))

    recon_optical_info = optical_info.copy()
    initial_volume = BirefringentVolume.init_from_file(
        init_volume_path, BackEnds.PYTORCH, recon_optical_info)
    visualize_volume(initial_volume, recon_optical_info)

    recon_directory = create_unique_directory("reconstructions")

    # Compute the reconstuction
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas,
        azim_image_meas, initial_volume, iteration_params, gt_vol=volume_GT
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(recon_config, omit_rays_based_on_pixels=True)
    reconstructor.rays.verbose = False
    reconstructor.reconstruct(output_dir=recon_directory)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)    


def resave_volume(old_path, new_path):
    """Resave a volume to a new path."""
    optical_info = setup_optical_parameters(
        "config_settings/optical_config_helix.json")
    volume = BirefringentVolume.init_from_file(
        old_path, BackEnds.PYTORCH, optical_info)
    my_description = "Helix1 resaved on 2024-01-10"
    volume.save_as_file(new_path, description=my_description)
    print(f"Saved volume to {new_path}")


def simulate_helix_images():
    optical_info = setup_optical_parameters(
        "config_settings/optical_config_helix_z7.json")
    optical_system = {'optical_info': optical_info}  
    simulator = ForwardModel(optical_system, backend=BACKEND)
    # Volume creation
    helix_path = "objects/Helix/Helix1_resaved.h5"
    volume_GT = BirefringentVolume.init_from_file(
        helix_path, BackEnds.PYTORCH, optical_info)
    # visualize_volume(volume_GT, optical_info)
    simulator.forward_model(volume_GT)
    simulator.view_images()


def recon_helix():
    optical_info = setup_optical_parameters(
        "config_settings/optical_config_helix_z7.json")
    postfix = get_forward_img_str_postfix(optical_info)
    forward_img_str = 'helix1' + postfix + '_7z.npy'
    simulate = False
    if simulate:
        optical_system = {'optical_info': optical_info}
        # Initialize the forward model. Raytracing is performed as part of the initialization.
        simulator = ForwardModel(optical_system, backend=BACKEND)
        # Volume creation
        helix_path = "objects/Helix/Helix1_resaved.h5"
        volume_GT = BirefringentVolume.init_from_file(
            helix_path, BackEnds.PYTORCH, optical_info)
        visualize_volume(volume_GT, optical_info)

        # # The following load from file does not work because the voxel
        # #   size is not saved in the file.
        # volume_GT = BirefringentVolume.load_from_file(
        #     helix_path, backend_type='torch')
        # visualize_volume(volume_GT, volume_GT.optical_info)

        simulator.forward_model(volume_GT)
        simulator.view_images()
        ret_image_meas = simulator.ret_img
        azim_image_meas = simulator.azim_img
        # Save the images as numpy arrays
        if True:
            ret_numpy = ret_image_meas.detach().numpy()
            np.save('forward_images/ret_' + forward_img_str, ret_numpy)
            azim_numpy = azim_image_meas.detach().numpy()
            np.save('forward_images/azim_' + forward_img_str, azim_numpy)
    else:
        ret_image_meas = np.load(os.path.join(
            'forward_images', 'ret_' + forward_img_str))
        azim_image_meas = np.load(os.path.join(
            'forward_images', 'azim_' + forward_img_str))

    recon_optical_info = optical_info.copy()
    iteration_params = setup_iteration_parameters(
        "config_settings/iter_config_helix.json")
    initial_volume = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=recon_optical_info,
        volume_creation_args=volume_args.random_args1
    )
    recon_directory = create_unique_directory("reconstructions")
    if not simulate:
        try:
            helix_path = "objects/Helix/Helix1_resaved.h5"
            volume_GT = BirefringentVolume.init_from_file(
                helix_path, BackEnds.PYTORCH, optical_info)
        except:
            volume_GT = initial_volume
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas,
        azim_image_meas, initial_volume, iteration_params, gt_vol=volume_GT
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(recon_config, omit_rays_based_on_pixels=True)
    reconstructor.rays.verbose = False
    reconstructor.reconstruct(output_dir=recon_directory)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def helix():
    """Helix-related functions."""
    # simulate_helix_images()
    # resave_volume("objects/Helix/Helix1.h5", "objects/Helix/Helix1_resaved.h5")
    # recon_helix()
    helix_recon_path = "reconstructions/2024-01-10_21-44-20/volume_ep_300.h5"
    recon_continuation(shape='helix', init_volume_path=helix_recon_path)


def main():
    optical_info = setup_optical_parameters(
        "config_settings/optical_config_largemla.json")
    optical_system = {'optical_info': optical_info}
    # Initialize the forward model. Raytracing is performed as part of the initialization.
    simulator = ForwardModel(optical_system, backend=BACKEND)
    # Volume creation
    volume_GT = BirefringentVolume(
        backend=BACKEND,
        optical_info=optical_info,
        volume_creation_args=volume_args.sphere_args5  # ellipsoid_args2 #voxel_args
    )
    # visualize_volume(volume_GT, optical_info)
    simulator.forward_model(volume_GT)
    simulator.view_images()
    ret_image_meas = simulator.ret_img
    azim_image_meas = simulator.azim_img

    recon_optical_info = optical_info.copy()
    iteration_params = setup_iteration_parameters(
        "config_settings/iter_config.json")
    initial_volume = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=recon_optical_info,
        volume_creation_args=volume_args.random_args
    )
    recon_directory = create_unique_directory("reconstructions")
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas, azim_image_meas,
                                        initial_volume, iteration_params, gt_vol=volume_GT)
    recon_config.save(recon_directory)
    # recon_config_recreated = ReconstructionConfig.load(recon_directory)
    reconstructor = Reconstructor(recon_config)
    reconstructor.reconstruct(output_dir=recon_directory)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


if __name__ == '__main__':
    # recon()
    # recon_sphere()
    recon_voxel()
    # recon_sphere_from_prev_try()

    # helix()
