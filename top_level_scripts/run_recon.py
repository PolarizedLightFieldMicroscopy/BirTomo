import os
import numpy as np
import torch
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.simulations import ForwardModel
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume
from VolumeRaytraceLFM.volumes import volume_args
from VolumeRaytraceLFM.setup_parameters import (
    setup_optical_parameters,
    setup_iteration_parameters,
)
from VolumeRaytraceLFM.reconstructions import ReconstructionConfig, Reconstructor
from VolumeRaytraceLFM.visualization.plotting_volume import visualize_volume
from VolumeRaytraceLFM.utils.file_utils import (
    create_unique_directory,
    get_forward_img_str_postfix,
)

BACKEND = BackEnds.PYTORCH
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DEVICE = "cpu"


def recon_gpu(postfix="gpu"):
    """Reconstruct a volume on the GPU."""
    optical_info = setup_optical_parameters("config_settings/optical_config3.json")
    optical_system = {"optical_info": optical_info}
    # Initialize the forward model. Raytracing is performed as part of the initialization.
    simulator = ForwardModel(optical_system, backend=BACKEND, device=DEVICE)
    simulator.to_device(DEVICE)  # Move the simulator to the GPU

    # Volume creation
    volume_GT = BirefringentVolume(
        backend=BACKEND,
        optical_info=optical_info,
        volume_creation_args=volume_args.ellipsoid_args2,
    )
    volume_GT.to(DEVICE)  # Move the volume to the GPU

    # visualize_volume(volume_GT, optical_info)
    simulator.rays.to_device(DEVICE)  # Move the rays to the GPU
    simulator.forward_model(volume_GT)
    ret_image_meas = simulator.ret_img
    azim_image_meas = simulator.azim_img

    recon_optical_info = optical_info
    iteration_params = setup_iteration_parameters("config_settings/iter_config.json")
    from_file = False
    if from_file:
        initial_volume = BirefringentVolume.init_from_file(
            r"reconstructions\2024-04-23_22-06-35_gpu\config_parameters\initial_volume.h5",
            BackEnds.PYTORCH,
            recon_optical_info,
        )
    else:
        initial_volume = BirefringentVolume(
            backend=BackEnds.PYTORCH,
            optical_info=recon_optical_info,
            volume_creation_args=volume_args.random_args,
        )
    initial_volume.to(DEVICE)  # Move the volume to the GPU

    recon_directory = create_unique_directory("reconstructions", postfix=postfix)
    recon_config = ReconstructionConfig(
        recon_optical_info,
        ret_image_meas,
        azim_image_meas,
        initial_volume,
        iteration_params,
        gt_vol=volume_GT,
    )
    recon_config.save(recon_directory)

    reconstructor = Reconstructor(
        recon_config, output_dir=recon_directory, device=DEVICE
    )
    reconstructor.to_device(DEVICE)  # Move the reconstructor to the GPU
    reconstructor.rays.verbose = False
    reconstructor.reconstruct()
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def recon():
    optical_info = setup_optical_parameters("config_settings/optical_config_voxel.json")
    simulate = False
    if simulate:
        optical_system = {"optical_info": optical_info}
        # Initialize the forward model. Raytracing is performed as part of the initialization.
        simulator = ForwardModel(optical_system, backend=BACKEND)
        # Volume creation
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.voxel_shifted_args,  # voxel_args
        )
        visualize_volume(volume_GT, optical_info)

        simulator.forward_model(volume_GT)
        simulator.view_images()
        ret_image_meas = simulator.ret_img
        azim_image_meas = simulator.azim_img
        # Save the images as numpy arrays
        if True:
            ret_numpy = ret_image_meas.detach().numpy()
            np.save("data/forward_images/ret_voxel_pos_1mla_17pix.npy", ret_numpy)
            azim_numpy = azim_image_meas.detach().numpy()
            np.save("data/forward_images/azim_voxel_pos_1mla_17pix.npy", azim_numpy)
    else:
        ret_image_meas = np.load(
            os.path.join("data/forward_images", "ret_voxel_pos_1mla_17pix.npy")
        )
        azim_image_meas = np.load(
            os.path.join("data/forward_images", "azim_voxel_pos_1mla_17pix.npy")
        )

    recon_optical_info = optical_info.copy()
    iteration_params = setup_iteration_parameters("config_settings/iter_config.json")
    initial_volume = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=recon_optical_info,
        volume_creation_args=volume_args.random_args,
    )
    recon_directory = create_unique_directory("reconstructions")
    if not simulate:
        volume_GT = initial_volume
    recon_config = ReconstructionConfig(
        recon_optical_info,
        ret_image_meas,
        azim_image_meas,
        initial_volume,
        iteration_params,
        gt_vol=volume_GT,
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(
        recon_config, output_dir=recon_directory, omit_rays_based_on_pixels=True
    )
    reconstructor.rays.verbose = False
    reconstructor.reconstruct()
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def recon_sphere():
    optical_info = setup_optical_parameters(
        "config_settings/basic_adjusted/optical_config_sphere3.json"
    )
    postfix = get_forward_img_str_postfix(optical_info)
    forward_img_str = "sphere3" + postfix + ".npy"
    simulate = False
    if simulate:
        optical_system = {"optical_info": optical_info}
        # Initialize the forward model. Raytracing is performed as part of the initialization.
        simulator = ForwardModel(optical_system, backend=BACKEND)
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.sphere_args3,
        )
        # visualize_volume(volume_GT, optical_info)

        simulator.forward_model(volume_GT)
        simulator.view_images()
        ret_image_meas = simulator.ret_img
        azim_image_meas = simulator.azim_img
        # Save the images as numpy arrays
        if True:
            ret_numpy = ret_image_meas.detach().numpy()
            np.save("data/forward_images/ret_" + forward_img_str, ret_numpy)
            azim_numpy = azim_image_meas.detach().numpy()
            np.save("data/forward_images/azim_" + forward_img_str, azim_numpy)
    else:
        ret_image_meas = np.load(
            os.path.join("data/forward_images", "ret_" + forward_img_str)
        )
        azim_image_meas = np.load(
            os.path.join("data/forward_images", "azim_" + forward_img_str)
        )

    recon_optical_info = optical_info.copy()
    ss_factor = 1
    if ss_factor == 2:
        recon_optical_info["n_voxels_per_ml"] = 2
        recon_optical_info["volume_shape"] = [22, 50, 50]
    elif ss_factor == 3:
        recon_optical_info["n_voxels_per_ml"] = 3
        recon_optical_info["volume_shape"] = [33, 75, 75]
    iteration_params = setup_iteration_parameters(
        "config_settings/iter_config_sphere.json"
    )
    # iteration_params["saved_ray_path"] = r"config_settings\rays\mla13_vol11_25_25\rays.pkl"
    # initial_volume = BirefringentVolume(
    #     backend=BackEnds.PYTORCH,
    #     optical_info=recon_optical_info,
    #     volume_creation_args=volume_args.random_args
    # )
    # optical_info['voxel_size_um'] = initial_volume.optical_info['voxel_size_um']
    # visualize_volume(initial_volume, optical_info)
    init_volume_path = "objects/sphere3/random_11_30_30.h5"
    initial_volume = BirefringentVolume.init_from_file(
        init_volume_path, BackEnds.PYTORCH, recon_optical_info
    )
    initial_volume = BirefringentVolume(
        backend=BACKEND,
        optical_info=recon_optical_info,
        volume_creation_args=volume_args.random_args2,
    )
    # Hmm should supersample on forward model too. At least should both give the same output

    iteration_params["initial volume path"] = init_volume_path

    parent_dir = os.path.join("reconstructions", "sphere3", f"ss{ss_factor}")
    recon_directory = create_unique_directory(parent_dir, postfix="ravel")
    if not simulate:
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.sphere_args3,
        )
    recon_config = ReconstructionConfig(
        recon_optical_info,
        ret_image_meas,
        azim_image_meas,
        initial_volume,
        iteration_params,
        gt_vol=volume_GT,
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(
        recon_config,
        output_dir=recon_directory,
        omit_rays_based_on_pixels=True,
        apply_volume_mask=False,
    )
    reconstructor.rays.verbose = False
    reconstructor.reconstruct(plot_live=True)
    reconstructor.rays.print_timing_info()
    # visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def recon_sphere_ss3_cont(init_volume_path, recon_dir_postfix=""):
    optical_info = setup_optical_parameters(
        "config_settings/optical_config_sphere_ss3.json"
    )
    postfix = get_forward_img_str_postfix(optical_info)
    forward_img_str = "sphere3" + postfix + "_ss3.npy"
    simulate = True
    if simulate:
        optical_system = {"optical_info": optical_info}
        # Initialize the forward model. Raytracing is performed as part of the initialization.
        simulator = ForwardModel(optical_system, backend=BACKEND)
        # Volume creation
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.sphere_args3_ss3,
        )
        visualize_volume(volume_GT, optical_info)

        simulator.forward_model(volume_GT, intensity=True)
        simulator.view_images()
        ret_image_meas = simulator.ret_img
        azim_image_meas = simulator.azim_img
        intensity_images_meas = simulator.img_list
        # Save the images as numpy arrays
        if True:
            for i, img in enumerate(intensity_images_meas):
                img_numpy = img.detach().numpy()
                np.save(
                    f"data/forward_images/intensity_{i}_{forward_img_str}", img_numpy
                )
        if False:
            ret_numpy = ret_image_meas.detach().numpy()
            np.save("data/forward_images/ret_" + forward_img_str, ret_numpy)
            azim_numpy = azim_image_meas.detach().numpy()
            np.save("data/forward_images/azim_" + forward_img_str, azim_numpy)
    else:
        ret_image_meas = np.load(
            os.path.join("data/forward_images", "ret_" + forward_img_str)
        )
        azim_image_meas = np.load(
            os.path.join("data/forward_images", "azim_" + forward_img_str)
        )
        intensity_images_meas = []
        for i in range(5):
            intensity_images_meas.append(
                np.load(
                    os.path.join(
                        "data/forward_images", f"intensity_{i}_{forward_img_str}"
                    )
                )
            )

    recon_optical_info = setup_optical_parameters(
        "config_settings/optical_config_sphere.json"
    )
    iteration_params = setup_iteration_parameters(
        "config_settings/iter_config_sphere.json"
    )
    initial_volume = BirefringentVolume.init_from_file(
        init_volume_path, BackEnds.PYTORCH, recon_optical_info
    )
    recon_directory = create_unique_directory(
        "reconstructions", postfix=recon_dir_postfix
    )
    if not simulate:
        # volume_GT = initial_volume
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.sphere_args3_ss3,
        )
    recon_config = ReconstructionConfig(
        recon_optical_info,
        ret_image_meas,
        azim_image_meas,
        initial_volume,
        iteration_params,
        gt_vol=volume_GT,
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(
        recon_config, output_dir=recon_directory, apply_volume_mask=True
    )
    reconstructor.rays.verbose = False
    reconstructor.reconstruct()
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def recon_sphere6_ss3_cont(init_volume_path, recon_dir_postfix=""):
    optical_info = setup_optical_parameters(
        "config_settings/optical_config_sphere6_ss3.json"
    )
    postfix = get_forward_img_str_postfix(optical_info)
    forward_img_str = "sphere6" + postfix + "_ss3.npy"
    simulate = False
    if simulate:
        optical_system = {"optical_info": optical_info}
        # Initialize the forward model. Raytracing is performed as part of the initialization.
        simulator = ForwardModel(optical_system, backend=BACKEND)
        # Volume creation
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.sphere_args6_thick_ss3,
        )
        visualize_volume(volume_GT, optical_info)

        simulator.forward_model(volume_GT)
        simulator.view_images()
        ret_image_meas = simulator.ret_img
        azim_image_meas = simulator.azim_img
        # Save the images as numpy arrays
        if True:
            ret_numpy = ret_image_meas.detach().numpy()
            np.save("data/forward_images/ret_" + forward_img_str, ret_numpy)
            azim_numpy = azim_image_meas.detach().numpy()
            np.save("data/forward_images/azim_" + forward_img_str, azim_numpy)
    else:
        ret_image_meas = np.load(
            os.path.join("data/forward_images", "ret_" + forward_img_str)
        )
        azim_image_meas = np.load(
            os.path.join("data/forward_images", "azim_" + forward_img_str)
        )

    recon_optical_info = setup_optical_parameters(
        "config_settings/optical_config_sphere6.json"
    )
    iteration_params = setup_iteration_parameters(
        "config_settings/iter_config_sphere.json"
    )
    initial_volume = BirefringentVolume.init_from_file(
        init_volume_path, BackEnds.PYTORCH, recon_optical_info
    )
    recon_directory = create_unique_directory(
        "reconstructions", postfix=recon_dir_postfix
    )
    if not simulate:
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.sphere_args6_thick_ss3,
        )
    recon_config = ReconstructionConfig(
        recon_optical_info,
        ret_image_meas,
        azim_image_meas,
        initial_volume,
        iteration_params,
        gt_vol=volume_GT,
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(
        recon_config,
        output_dir=recon_directory,
        omit_rays_based_on_pixels=False,
        apply_volume_mask=True,
    )
    reconstructor.rays.verbose = False
    reconstructor.reconstruct()
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def recon_sphere6_prep():
    optical_info = setup_optical_parameters(
        "config_settings/optical_config_sphere6_small.json"
    )
    postfix = get_forward_img_str_postfix(optical_info)
    forward_img_str = "sphere6" + postfix + ".npy"
    simulate = True
    if simulate:
        optical_system = {"optical_info": optical_info}
        # Initialize the forward model. Raytracing is performed as part of the initialization.
        simulator = ForwardModel(optical_system, backend=BACKEND)
        # Volume creation
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.sphere_args6_thick,
        )
        # visualize_volume(volume_GT, optical_info)

        simulator.forward_model(volume_GT)
        simulator.view_images()
        ret_image_meas = simulator.ret_img
        azim_image_meas = simulator.azim_img
        # Save the images as numpy arrays
        if True:
            ret_numpy = ret_image_meas.detach().numpy()
            np.save("data/forward_images/ret_" + forward_img_str, ret_numpy)
            azim_numpy = azim_image_meas.detach().numpy()
            np.save("data/forward_images/azim_" + forward_img_str, azim_numpy)
    else:
        ret_image_meas = np.load(
            os.path.join("data/forward_images", "ret_" + forward_img_str)
        )
        azim_image_meas = np.load(
            os.path.join("data/forward_images", "azim_" + forward_img_str)
        )


def recon_sphere_from_prev_try():
    optical_info = setup_optical_parameters(
        "config_settings/optical_config_sphere.json"
    )
    postfix = get_forward_img_str_postfix(optical_info)
    forward_img_str = "sphere6_thick1" + postfix + ".npy"
    simulate = True
    if simulate:
        optical_system = {"optical_info": optical_info}
        # Initialize the forward model. Raytracing is performed as part of the initialization.
        simulator = ForwardModel(optical_system, backend=BACKEND)
        # Volume creation
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.sphere_args6,
        )
        # visualize_volume(volume_GT, optical_info)

        simulator.forward_model(volume_GT)
        simulator.view_images()
        ret_image_meas = simulator.ret_img
        azim_image_meas = simulator.azim_img
        # Save the images as numpy arrays
        if True:
            ret_numpy = ret_image_meas.detach().numpy()
            np.save("data/forward_images/ret_" + forward_img_str, ret_numpy)
            azim_numpy = azim_image_meas.detach().numpy()
            np.save("data/forward_images/azim_" + forward_img_str, azim_numpy)
    else:
        ret_image_meas = np.load(
            os.path.join("data/forward_images", "ret_" + forward_img_str)
        )
        azim_image_meas = np.load(
            os.path.join("data/forward_images", "azim_" + forward_img_str)
        )

    recon_optical_info = optical_info.copy()
    iteration_params = setup_iteration_parameters(
        "config_settings/iter_config_sphere.json"
    )
    sphere_path = "volumes/2024-01-02_23-26-15/volume_ep_300.h5"
    initial_volume = BirefringentVolume.init_from_file(
        sphere_path, BackEnds.PYTORCH, recon_optical_info
    )
    recon_directory = create_unique_directory("reconstructions")
    if not simulate:
        volume_GT = initial_volume
    recon_config = ReconstructionConfig(
        recon_optical_info,
        ret_image_meas,
        azim_image_meas,
        initial_volume,
        iteration_params,
        gt_vol=volume_GT,
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(
        recon_config, output_dir=recon_directory, omit_rays_based_on_pixels=True
    )
    reconstructor.rays.verbose = False
    reconstructor.reconstruct()
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def recon_voxel():
    optical_info = setup_optical_parameters("config_settings/optical_config_voxel.json")
    postfix = get_forward_img_str_postfix(optical_info)
    forward_img_str = "voxel_pos" + postfix + ".npy"
    simulate = False
    if simulate:
        optical_system = {"optical_info": optical_info}
        # Initialize the forward model. Raytracing is performed as part of the initialization.
        simulator = ForwardModel(optical_system, backend=BACKEND)
        # Volume creation
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.voxel_args,
        )
        visualize_volume(volume_GT, optical_info)

        simulator.forward_model(volume_GT)
        simulator.view_images()
        ret_image_meas = simulator.ret_img
        azim_image_meas = simulator.azim_img
        # Save the images as numpy arrays
        if True:
            ret_numpy = ret_image_meas.detach().numpy()
            np.save("data/forward_images/ret_" + forward_img_str, ret_numpy)
            azim_numpy = azim_image_meas.detach().numpy()
            np.save("data/forward_images/azim_" + forward_img_str, azim_numpy)
    else:
        ret_image_meas = np.load(
            os.path.join("data/forward_images", "ret_" + forward_img_str)
        )
        azim_image_meas = np.load(
            os.path.join("data/forward_images", "azim_" + forward_img_str)
        )

    recon_optical_info = optical_info.copy()
    # recon_optical_info["n_voxels_per_ml"] = 3
    # recon_optical_info["volume_shape"] = [9, 27, 27]
    recon_optical_info["n_voxels_per_ml"] = 2
    recon_optical_info["volume_shape"] = [6, 18, 18]
    iteration_params = setup_iteration_parameters("config_settings/iter_config.json")
    continue_recon = False
    if continue_recon:
        volume_path = (
            r"reconstructions/voxel\2024-05-11_15-32-33_vox_debug_ss3\volume_ep_0100.h5"
        )
        initial_volume = BirefringentVolume.init_from_file(
            volume_path, BackEnds.PYTORCH, recon_optical_info
        )
    else:
        initial_volume = BirefringentVolume(
            backend=BACKEND,
            optical_info=recon_optical_info,
            volume_creation_args=volume_args.random_args,
        )
    recon_directory = create_unique_directory(
        "reconstructions/voxel/ss2", postfix="debug"
    )
    if not simulate:
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.voxel_args,
        )
        # volume_GT = initial_volume
    recon_config = ReconstructionConfig(
        recon_optical_info,
        ret_image_meas,
        azim_image_meas,
        initial_volume,
        iteration_params,
        gt_vol=volume_GT,
    )
    recon_config.save(recon_directory)

    # Changes made for developing masking process
    reconstructor = Reconstructor(
        recon_config,
        output_dir=recon_directory,
        omit_rays_based_on_pixels=True,
        apply_volume_mask=False,
    )
    # reconstructor.rays._count_vox_raytrace_occurrences(zero_retardance_voxels=True)
    # vox_list = reconstructor.rays.identify_voxels_repeated_zero_ret()
    # reconstructor.voxel_mask_setup()

    reconstructor.rays.verbose = False
    # reconstructor.rays.use_lenslet_based_filtering = False
    # with profiler.profile(profile_memory=True, use_cuda=True) as prof:
    #     reconstructor = Reconstructor(recon_config, output_dir=recon_directory, apply_volume_mask=True)
    # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    reconstructor.reconstruct(all_prop_elements=False, plot_live=True)
    print(
        f"Raytracing time: {reconstructor.rays.times['ray_trace_through_volume'] * 1000:.2f}"
    )
    reconstructor.rays.print_timing_info()
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def recon_voxel_neg():
    optical_info = setup_optical_parameters("config_settings/optical_config_voxel.json")
    postfix = get_forward_img_str_postfix(optical_info)
    forward_img_str = "voxel_neg" + postfix + ".npy"
    simulate = False
    if simulate:
        optical_system = {"optical_info": optical_info}
        simulator = ForwardModel(optical_system, backend=BACKEND)
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.voxel_neg_args,
        )
        visualize_volume(volume_GT, optical_info)
        simulator.forward_model(volume_GT)
        simulator.view_images()
        ret_image_meas = simulator.ret_img
        azim_image_meas = simulator.azim_img
        # Save the images as numpy arrays
        if True:
            ret_numpy = ret_image_meas.detach().numpy()
            np.save("data/forward_images/ret_" + forward_img_str, ret_numpy)
            azim_numpy = azim_image_meas.detach().numpy()
            np.save("data/forward_images/azim_" + forward_img_str, azim_numpy)
    else:
        ret_image_meas = np.load(
            os.path.join("data/forward_images", "ret_" + forward_img_str)
        )
        azim_image_meas = np.load(
            os.path.join("data/forward_images", "azim_" + forward_img_str)
        )

    recon_optical_info = optical_info.copy()
    # recon_optical_info["n_voxels_per_ml"] = 3
    # recon_optical_info["volume_shape"] = [9, 27, 27]
    # recon_optical_info["n_voxels_per_ml"] = 2
    # recon_optical_info["volume_shape"] = [6, 18, 18]
    iteration_params = setup_iteration_parameters("config_settings/iter_config.json")
    continue_recon = True
    if continue_recon:
        volume_path = r"C:\Users\Geneva\Documents\Code\GeoBirT\reconstructions\voxel_neg\ss1\2024-06-03_13-43-49_debug\config_parameters\initial_volume.h5"
        initial_volume = BirefringentVolume.init_from_file(
            volume_path, BackEnds.PYTORCH, recon_optical_info
        )
    else:
        initial_volume = BirefringentVolume(
            backend=BACKEND,
            optical_info=recon_optical_info,
            volume_creation_args=volume_args.random_args,
        )
    recon_directory = create_unique_directory(
        "reconstructions/voxel_neg/ss1", postfix="debug"
    )
    if not simulate:
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.voxel_neg_args,
        )
        # volume_GT = initial_volume
    recon_config = ReconstructionConfig(
        recon_optical_info,
        ret_image_meas,
        azim_image_meas,
        initial_volume,
        iteration_params,
        gt_vol=volume_GT,
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(
        recon_config,
        output_dir=recon_directory,
        omit_rays_based_on_pixels=True,
        apply_volume_mask=False,
    )
    reconstructor.rays.verbose = False
    reconstructor.reconstruct(all_prop_elements=False, plot_live=True)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def recon_voxel_upsampled():
    optical_info = setup_optical_parameters("config_settings/optical_config_voxel.json")
    postfix = get_forward_img_str_postfix(optical_info)
    forward_img_str = "voxel_pos" + postfix + ".npy"
    simulate = False
    if simulate:
        optical_system = {"optical_info": optical_info}
        # Initialize the forward model. Raytracing is performed as part of the initialization.
        simulator = ForwardModel(optical_system, backend=BACKEND)
        # Volume creation
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.voxel_args,
        )
        visualize_volume(volume_GT, optical_info)

        simulator.forward_model(volume_GT)
        simulator.view_images()
        ret_image_meas = simulator.ret_img
        azim_image_meas = simulator.azim_img
        # Save the images as numpy arrays
        if True:
            ret_numpy = ret_image_meas.detach().numpy()
            np.save("data/forward_images/ret_" + forward_img_str, ret_numpy)
            azim_numpy = azim_image_meas.detach().numpy()
            np.save("data/forward_images/azim_" + forward_img_str, azim_numpy)
    else:
        ret_image_meas = np.load(
            os.path.join("data/forward_images", "ret_" + forward_img_str)
        )
        azim_image_meas = np.load(
            os.path.join("data/forward_images", "azim_" + forward_img_str)
        )

    recon_optical_info = setup_optical_parameters(
        "config_settings/optical_config_voxel_upsample.json"
    )
    ideal_volume_path = "objects/upsampled_voxel.h5"
    ideal_volume = BirefringentVolume.init_from_file(
        ideal_volume_path, BackEnds.PYTORCH, recon_optical_info
    )
    iteration_params = setup_iteration_parameters("config_settings/iter_config.json")
    continue_recon = False
    if continue_recon:
        volume_path = "reconstructions/2024-01-04_16-31-34/volume_ep_100.h5"
        initial_volume = BirefringentVolume.init_from_file(
            volume_path, BackEnds.PYTORCH, recon_optical_info
        )
    else:
        initial_volume = BirefringentVolume(
            backend=BACKEND,
            optical_info=recon_optical_info,
            volume_creation_args=volume_args.random_args,
        )
    recon_directory = create_unique_directory("reconstructions", postfix="vox_upsample")
    volume_GT = ideal_volume
    recon_config = ReconstructionConfig(
        recon_optical_info,
        ret_image_meas,
        azim_image_meas,
        initial_volume,
        iteration_params,
        gt_vol=volume_GT,
    )
    recon_config.save(recon_directory)

    # Changes made for developing masking process
    reconstructor = Reconstructor(
        recon_config,
        output_dir=recon_directory,
        omit_rays_based_on_pixels=True,
        apply_volume_mask=False,
    )
    # reconstructor.rays._count_vox_raytrace_occurrences(zero_retardance_voxels=True)
    # vox_list = reconstructor.rays.identify_voxels_repeated_zero_ret()
    # reconstructor.voxel_mask_setup()

    reconstructor.rays.verbose = False
    # reconstructor.rays.use_lenslet_based_filtering = False
    # with profiler.profile(profile_memory=True, use_cuda=True) as prof:
    #     reconstructor = Reconstructor(recon_config, output_dir=recon_directory, apply_volume_mask=True)
    # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    reconstructor.reconstruct(plot_live=True)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def recon_voxel_shifted():
    optical_info = setup_optical_parameters("config_settings/optical_config_voxel.json")
    postfix = get_forward_img_str_postfix(optical_info)
    forward_img_str = "voxel_pos_shiftedy" + postfix + ".npy"
    simulate = True
    if simulate:
        optical_system = {"optical_info": optical_info}
        simulator = ForwardModel(optical_system, backend=BACKEND)
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.voxel_shiftedy_args,
        )
        visualize_volume(volume_GT, optical_info)
        simulator.forward_model(volume_GT)
        simulator.view_images()
        ret_image_meas = simulator.ret_img
        azim_image_meas = simulator.azim_img
        # Save the images as numpy arrays
        if True:
            ret_numpy = ret_image_meas.detach().numpy()
            np.save("data/forward_images/ret_" + forward_img_str, ret_numpy)
            azim_numpy = azim_image_meas.detach().numpy()
            np.save("data/forward_images/azim_" + forward_img_str, azim_numpy)
    else:
        ret_image_meas = np.load(
            os.path.join("data/forward_images", "ret_" + forward_img_str)
        )
        azim_image_meas = np.load(
            os.path.join("data/forward_images", "azim_" + forward_img_str)
        )

    recon_optical_info = optical_info.copy()
    iteration_params = setup_iteration_parameters("config_settings/iter_config.json")
    initial_volume = BirefringentVolume(
        backend=BACKEND,
        optical_info=recon_optical_info,
        volume_creation_args=volume_args.random_args,
    )
    recon_directory = create_unique_directory("reconstructions", postfix="voxshifted")
    if not simulate:
        volume_GT = initial_volume
    recon_config = ReconstructionConfig(
        recon_optical_info,
        ret_image_meas,
        azim_image_meas,
        initial_volume,
        iteration_params,
        gt_vol=volume_GT,
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(
        recon_config,
        output_dir=recon_directory,
        omit_rays_based_on_pixels=True,
        apply_volume_mask=True,
    )
    reconstructor.reconstruct()
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def recon_continuation(shape="sphere", init_volume_path=None):
    if shape == "sphere":
        optical_info = setup_optical_parameters(
            "config_settings/optical_config_sphere.json"
        )
        postfix = get_forward_img_str_postfix(optical_info)
        forward_img_str = "sphere6_thick1" + postfix + ".npy"
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.sphere_args6,
        )
        iteration_params = setup_iteration_parameters(
            "config_settings/iter_config_sphere.json"
        )
    elif shape == "helix":
        optical_info = setup_optical_parameters(
            "config_settings/optical_config_helix_z7.json"
        )
        postfix = get_forward_img_str_postfix(optical_info)
        forward_img_str = "helix1" + postfix + "_7z.npy"
        helix_path = "objects/Helix/Helix1_resaved.h5"
        volume_GT = BirefringentVolume.init_from_file(
            helix_path, BackEnds.PYTORCH, optical_info
        )
        iteration_params = setup_iteration_parameters(
            "config_settings/iter_config_helix.json"
        )
    else:
        raise ValueError(f"Unknown shape {shape}")

    iteration_params["notes"] = (
        f"Continuation of previous reconstruction {init_volume_path}"
    )
    # visualize_volume(volume_GT, optical_info)
    ret_image_meas = np.load(
        os.path.join("data/forward_images", "ret_" + forward_img_str)
    )
    azim_image_meas = np.load(
        os.path.join("data/forward_images", "azim_" + forward_img_str)
    )

    recon_optical_info = optical_info.copy()
    initial_volume = BirefringentVolume.init_from_file(
        init_volume_path, BackEnds.PYTORCH, recon_optical_info
    )
    visualize_volume(initial_volume, recon_optical_info)

    recon_directory = create_unique_directory("reconstructions")

    # Compute the reconstuction
    recon_config = ReconstructionConfig(
        recon_optical_info,
        ret_image_meas,
        azim_image_meas,
        initial_volume,
        iteration_params,
        gt_vol=volume_GT,
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(
        recon_config, output_dir=recon_directory, omit_rays_based_on_pixels=True
    )
    reconstructor.rays.verbose = False
    reconstructor.reconstruct()
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def resave_volume(old_path, new_path):
    """Resave a volume to a new path."""
    optical_info = setup_optical_parameters("config_settings/optical_config_helix.json")
    volume = BirefringentVolume.init_from_file(old_path, BackEnds.PYTORCH, optical_info)
    my_description = "Helix1 resaved on 2024-01-10"
    volume.save_as_file(new_path, description=my_description)
    print(f"Saved volume to {new_path}")


def simulate_helix_images():
    optical_info = setup_optical_parameters(
        "config_settings/optical_config_helix_z7.json"
    )
    optical_system = {"optical_info": optical_info}
    simulator = ForwardModel(optical_system, backend=BACKEND)
    # Volume creation
    helix_path = "objects/Helix/Helix1_resaved.h5"
    volume_GT = BirefringentVolume.init_from_file(
        helix_path, BackEnds.PYTORCH, optical_info
    )
    # visualize_volume(volume_GT, optical_info)
    simulator.forward_model(volume_GT)
    simulator.view_images()


def recon_helix():
    optical_info = setup_optical_parameters("config_settings/optical_config3.json")
    postfix = get_forward_img_str_postfix(optical_info)
    forward_img_str = "helix1" + postfix + "_7z.npy"
    simulate = False
    if simulate:
        optical_system = {"optical_info": optical_info}
        # Initialize the forward model. Raytracing is performed as part of the initialization.
        simulator = ForwardModel(optical_system, backend=BACKEND)
        # Volume creation
        helix_path = "objects/Helix/Helix1_resaved.h5"
        volume_GT = BirefringentVolume.init_from_file(
            helix_path, BackEnds.PYTORCH, optical_info
        )
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
            np.save("data/forward_images/ret_" + forward_img_str, ret_numpy)
            azim_numpy = azim_image_meas.detach().numpy()
            np.save("data/forward_images/azim_" + forward_img_str, azim_numpy)
    else:
        ret_image_meas = np.load(
            os.path.join("data/forward_images", "ret_" + forward_img_str)
        )
        azim_image_meas = np.load(
            os.path.join("data/forward_images", "azim_" + forward_img_str)
        )

    recon_optical_info = optical_info.copy()
    iteration_params = setup_iteration_parameters(
        "config_settings/iter_config_helix.json"
    )
    initial_volume = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=recon_optical_info,
        volume_creation_args=volume_args.random_args1,
    )
    recon_directory = create_unique_directory("reconstructions")
    if not simulate:
        try:
            helix_path = "objects/Helix/Helix1_resaved.h5"
            volume_GT = BirefringentVolume.init_from_file(
                helix_path, BackEnds.PYTORCH, optical_info
            )
        except:
            volume_GT = initial_volume
    recon_config = ReconstructionConfig(
        recon_optical_info,
        ret_image_meas,
        azim_image_meas,
        initial_volume,
        iteration_params,
        gt_vol=volume_GT,
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(
        recon_config,
        output_dir=recon_directory,
        omit_rays_based_on_pixels=True,
        apply_volume_mask=True,
    )
    reconstructor.rays.verbose = False
    reconstructor.reconstruct()
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def helix():
    """Helix-related functions."""
    # simulate_helix_images()
    # resave_volume("objects/Helix/Helix1.h5", "objects/Helix/Helix1_resaved.h5")
    # recon_helix()
    helix_recon_path = "reconstructions/2024-01-10_21-44-20/volume_ep_300.h5"
    recon_continuation(shape="helix", init_volume_path=helix_recon_path)


def recon_voxel_noise():
    optical_info = setup_optical_parameters("config_settings/optical_config_voxel.json")
    postfix = get_forward_img_str_postfix(optical_info)
    forward_img_str = "voxel_pos" + postfix + ".npy"
    percent_noise = 5
    simulate = True
    if simulate:
        optical_system = {"optical_info": optical_info}
        # Initialize the forward model. Raytracing is performed as part of the initialization.
        simulator = ForwardModel(optical_system, backend=BACKEND)
        # Volume creation
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.voxel_args,
        )
        # visualize_volume(volume_GT, optical_info)

        simulator.forward_model(volume_GT)
        # simulator.view_images()
        ret_image_meas = simulator.ret_img
        azim_image_meas = simulator.azim_img
        add_noise = True
        if add_noise:
            ret_numpy = ret_image_meas.detach().numpy()
            azim_numpy = azim_image_meas.detach().numpy()
            # Generate Gaussian noise
            mean = 0
            std = np.pi * percent_noise / 100
            gaussian_noise = np.random.normal(mean, std, ret_numpy.shape)
            arrays = [ret_numpy, azim_numpy]
            array_names = [
                "ret_noise" + str(percent_noise),
                "azim_noise" + str(percent_noise),
            ]  # Names for saving files
            for i, image in enumerate(arrays):
                # Ensure the image array is a float type for noise addition
                image_float = np.float32(image)
                gaussian_noise = np.random.normal(mean, std, image.shape)
                noisy_array = image_float + gaussian_noise
                noisy_array = np.clip(noisy_array, 0, 255)
                # Convert back to the original data type, e.g., uint8 if it was an image
                # noisy_array = np.uint8(noisy_array)

                save_path = f"data/forward_images/{array_names[i]}_" + forward_img_str
                np.save(save_path, noisy_array)
                print(f"Saved noisy array to {save_path}")

    else:
        pass
        # ret_image_meas = np.load(os.path.join(
        #     'data/forward_images', 'ret_' + forward_img_str))
        # azim_image_meas = np.load(os.path.join(
        #     'data/forward_images', 'azim_' + forward_img_str))
    ret_image_meas = np.load(
        os.path.join(
            "data/forward_images",
            "ret_noise" + str(percent_noise) + "_" + forward_img_str,
        )
    )
    azim_image_meas = np.load(
        os.path.join(
            "data/forward_images",
            "azim_noise" + str(percent_noise) + "_" + forward_img_str,
        )
    )

    recon_optical_info = optical_info.copy()
    # recon_optical_info["n_voxels_per_ml"] = 3
    # recon_optical_info["volume_shape"] = [9, 17, 17]
    iteration_params = setup_iteration_parameters("config_settings/iter_config.json")
    continue_recon = False
    if continue_recon:
        volume_path = "reconstructions/2024-01-04_16-31-34/volume_ep_100.h5"
        initial_volume = BirefringentVolume.init_from_file(
            volume_path, BackEnds.PYTORCH, recon_optical_info
        )
    else:
        initial_volume = BirefringentVolume(
            backend=BACKEND,
            optical_info=recon_optical_info,
            volume_creation_args=volume_args.random_args,
        )
    recon_directory = create_unique_directory("reconstructions")
    if not simulate:
        volume_GT = initial_volume
    recon_config = ReconstructionConfig(
        recon_optical_info,
        ret_image_meas,
        azim_image_meas,
        initial_volume,
        iteration_params,
        gt_vol=volume_GT,
    )
    recon_config.save(recon_directory)

    # Changes made for developing masking process
    reconstructor = Reconstructor(
        recon_config, output_dir=recon_directory, apply_volume_mask=True
    )
    reconstructor.reconstruct()


def recon_sphere_noise():
    optical_info = setup_optical_parameters(
        "config_settings/optical_config_sphere_ss3.json"
    )
    postfix = get_forward_img_str_postfix(optical_info)
    forward_img_str = "sphere3" + postfix + "_ss3.npy"
    percent_noise = 5
    simulate = False
    if simulate:
        optical_system = {"optical_info": optical_info}
        # Initialize the forward model. Raytracing is performed as part of the initialization.
        simulator = ForwardModel(optical_system, backend=BACKEND)
        # Volume creation
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.sphere_args3_ss3,
        )
        # visualize_volume(volume_GT, optical_info)

        simulator.forward_model(volume_GT)
        # simulator.view_images()
        ret_image_meas = simulator.ret_img
        azim_image_meas = simulator.azim_img

        ret_numpy = ret_image_meas.detach().numpy()
        azim_numpy = azim_image_meas.detach().numpy()
        # Generate Gaussian noise
        mean = 0
        std = np.pi * percent_noise / 100
        gaussian_noise = np.random.normal(mean, std, ret_numpy.shape)
        arrays = [ret_numpy, azim_numpy]
        array_names = [
            "ret_noise" + str(percent_noise),
            "azim_noise" + str(percent_noise),
        ]  # Names for saving files
        for i, image in enumerate(arrays):
            # Ensure the image array is a float type for noise addition
            image_float = np.float32(image)
            gaussian_noise = np.random.normal(mean, std, image.shape)
            noisy_array = image_float + gaussian_noise
            noisy_array = np.clip(noisy_array, 0, 255)
            # Convert back to the original data type, e.g., uint8 if it was an image
            # noisy_array = np.uint8(noisy_array)

            save_path = f"data/forward_images/{array_names[i]}_" + forward_img_str
            np.save(save_path, noisy_array)
            print(f"Saved noisy array to {save_path}")
    else:
        pass
    ret_image_meas = np.load(
        os.path.join(
            "data/forward_images",
            "ret_noise" + str(percent_noise) + "_" + forward_img_str,
        )
    )
    azim_image_meas = np.load(
        os.path.join(
            "data/forward_images",
            "azim_noise" + str(percent_noise) + "_" + forward_img_str,
        )
    )

    recon_optical_info = setup_optical_parameters(
        "config_settings/optical_config_sphere.json"
    )
    # recon_optical_info = optical_info.copy()
    # recon_optical_info["n_voxels_per_ml"] = 3
    # recon_optical_info["volume_shape"] = [33, 75, 75]
    iteration_params = setup_iteration_parameters(
        "config_settings/iter_config_sphere.json"
    )
    initial_volume = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=recon_optical_info,
        volume_creation_args=volume_args.random_args,
    )
    recon_directory = create_unique_directory("reconstructions")
    if not simulate:
        volume_GT = initial_volume
    recon_config = ReconstructionConfig(
        recon_optical_info,
        ret_image_meas,
        azim_image_meas,
        initial_volume,
        iteration_params,
        gt_vol=volume_GT,
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(
        recon_config, output_dir=recon_directory, apply_volume_mask=True
    )
    reconstructor.rays.verbose = False
    reconstructor.reconstruct()
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def recon_sphere6_noise():
    optical_info = setup_optical_parameters(
        "config_settings/optical_config_sphere6.json"
    )
    postfix = get_forward_img_str_postfix(optical_info)
    forward_img_str = "sphere6" + postfix + ".npy"
    percent_noise = 5
    simulate = False
    if simulate:
        optical_system = {"optical_info": optical_info}
        # Initialize the forward model. Raytracing is performed as part of the initialization.
        simulator = ForwardModel(optical_system, backend=BACKEND)
        # Volume creation
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.sphere_args6_thick,
        )
        # visualize_volume(volume_GT, optical_info)

        simulator.forward_model(volume_GT)
        simulator.view_images()
        ret_image_meas = simulator.ret_img
        azim_image_meas = simulator.azim_img

        ret_numpy = ret_image_meas.detach().numpy()
        azim_numpy = azim_image_meas.detach().numpy()
        # Generate Gaussian noise
        mean = 0
        std = np.pi * percent_noise / 100
        gaussian_noise = np.random.normal(mean, std, ret_numpy.shape)
        arrays = [ret_numpy, azim_numpy]
        array_names = [
            "ret_noise" + str(percent_noise),
            "azim_noise" + str(percent_noise),
        ]  # Names for saving files
        for i, image in enumerate(arrays):
            # Ensure the image array is a float type for noise addition
            image_float = np.float32(image)
            gaussian_noise = np.random.normal(mean, std, image.shape)
            noisy_array = image_float + gaussian_noise
            noisy_array = np.clip(noisy_array, 0, 255)
            # Convert back to the original data type, e.g., uint8 if it was an image
            # noisy_array = np.uint8(noisy_array)

            save_path = f"data/forward_images/{array_names[i]}_" + forward_img_str
            np.save(save_path, noisy_array)
            print(f"Saved noisy array to {save_path}")
    else:
        pass
    ret_image_meas = np.load(
        os.path.join(
            "data/forward_images",
            "ret_noise" + str(percent_noise) + "_" + forward_img_str,
        )
    )
    azim_image_meas = np.load(
        os.path.join(
            "data/forward_images",
            "azim_noise" + str(percent_noise) + "_" + forward_img_str,
        )
    )

    recon_optical_info = setup_optical_parameters(
        "config_settings/optical_config_sphere6.json"
    )
    # recon_optical_info = optical_info.copy()
    # recon_optical_info["n_voxels_per_ml"] = 3
    # recon_optical_info["volume_shape"] = [33, 75, 75]
    iteration_params = setup_iteration_parameters(
        "config_settings/iter_config_sphere.json"
    )
    initial_volume = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=recon_optical_info,
        volume_creation_args=volume_args.random_args,
    )
    recon_directory = create_unique_directory(
        "reconstructions", postfix="sphere6_noise_cos2"
    )
    if not simulate:
        volume_GT = initial_volume
    recon_config = ReconstructionConfig(
        recon_optical_info,
        ret_image_meas,
        azim_image_meas,
        initial_volume,
        iteration_params,
        gt_vol=volume_GT,
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(
        recon_config,
        output_dir=recon_directory,
        omit_rays_based_on_pixels=False,
        apply_volume_mask=True,
    )
    reconstructor.rays.verbose = False
    reconstructor.reconstruct()
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def main():
    optical_info = setup_optical_parameters(
        "config_settings/optical_config_largemla.json"
    )
    optical_system = {"optical_info": optical_info}
    # Initialize the forward model. Raytracing is performed as part of the initialization.
    simulator = ForwardModel(optical_system, backend=BACKEND)
    # Volume creation
    volume_GT = BirefringentVolume(
        backend=BACKEND,
        optical_info=optical_info,
        volume_creation_args=volume_args.sphere_args5,  # ellipsoid_args2 #voxel_args
    )
    # visualize_volume(volume_GT, optical_info)
    simulator.forward_model(volume_GT)
    simulator.view_images()
    ret_image_meas = simulator.ret_img
    azim_image_meas = simulator.azim_img

    recon_optical_info = optical_info.copy()
    iteration_params = setup_iteration_parameters("config_settings/iter_config.json")
    initial_volume = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=recon_optical_info,
        volume_creation_args=volume_args.random_args,
    )
    recon_directory = create_unique_directory("reconstructions")
    recon_config = ReconstructionConfig(
        recon_optical_info,
        ret_image_meas,
        azim_image_meas,
        initial_volume,
        iteration_params,
        gt_vol=volume_GT,
    )
    recon_config.save(recon_directory)
    # recon_config_recreated = ReconstructionConfig.load(recon_directory)
    reconstructor = Reconstructor(recon_config)
    reconstructor.reconstruct()
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


if __name__ == "__main__":
    # recon()
    # recon_sphere()
    recon_voxel_neg()
    # recon_gpu(postfix='gpu')
    # recon_voxel()
    # recon_voxel_upsampled()
    # recon_gpu()
