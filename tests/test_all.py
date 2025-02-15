import numpy as np
import torch
import pytest
import matplotlib.pyplot as plt
import copy
from tests.fixtures_optical_info import set_optical_info
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import (
    BirefringentVolume,
    BirefringentRaytraceLFM,
)
from VolumeRaytraceLFM.jones.jones_calculus import JonesMatrixGenerators
from VolumeRaytraceLFM.visualization.plotting_intensity import plot_intensity_images
from VolumeRaytraceLFM.jones.intensity import ret_and_azim_from_intensity


@pytest.fixture(scope="module")
def global_data():
    """Create global optic_setting and optical_info containing all the optics and volume information
    The tests can access this by passing the name of this function as an argument for example:
    def test_something(global_data):
        optical_info = global_data['optical_info']
    """
    # Set torch precision to Double to match numpy
    torch.set_default_dtype(torch.float64)

    optical_info = set_optical_info([11, 11, 11], 5, 1)
    optical_info["n_medium"] = 1.52

    return {"optical_info": optical_info}


# Test systems with different number of rays per micro-lens


@pytest.mark.parametrize("pixels_per_ml_init", [3, 5, 10, 17, 33])
def test_rays_computation(global_data, pixels_per_ml_init):

    # Gather global data
    optical_info = copy.deepcopy(global_data["optical_info"])

    optical_info["pixels_per_ml"] = pixels_per_ml_init

    # Create numpy and pytorch raytracer
    BF_raytrace_numpy = BirefringentRaytraceLFM(optical_info=optical_info)
    BF_raytrace_torch = BirefringentRaytraceLFM(
        backend=BackEnds.PYTORCH, optical_info=optical_info
    )

    # Compute ray-volume geometry and Siddon algorithm
    BF_raytrace_numpy.compute_rays_geometry()
    BF_raytrace_torch.compute_rays_geometry()

    # Compare results
    # Remove nan for assertion
    BF_raytrace_numpy.ray_entry[np.isnan(BF_raytrace_numpy.ray_entry)] = -10
    BF_raytrace_torch.ray_entry[torch.isnan(BF_raytrace_torch.ray_entry)] = -10
    BF_raytrace_numpy.ray_exit[np.isnan(BF_raytrace_numpy.ray_exit)] = -10
    BF_raytrace_torch.ray_exit[torch.isnan(BF_raytrace_torch.ray_exit)] = -10
    BF_raytrace_numpy.ray_direction[np.isnan(BF_raytrace_numpy.ray_direction)] = -10
    BF_raytrace_torch.ray_direction[torch.isnan(BF_raytrace_torch.ray_direction)] = -10

    assert np.all(
        np.isclose(BF_raytrace_numpy.ray_entry, BF_raytrace_torch.ray_entry.numpy())
    ), "ray_entry calculation mismatch between Numpy and Pytorch back-end"
    assert np.all(
        np.isclose(BF_raytrace_numpy.ray_exit, BF_raytrace_torch.ray_exit.numpy())
    ), "ray_exit calculation mismatch between Numpy and Pytorch back-end"
    assert np.all(
        np.isclose(
            BF_raytrace_numpy.ray_direction, BF_raytrace_torch.ray_direction.numpy()
        )
    ), "ray_direction calculation mismatch between Numpy and Pytorch back-end"

    # todo: This is hard to compare, as we only store the valid rays in pytorch vs numpy that we store all rays
    # for n_basis in range(3):
    #     for n_ray in range(len(BF_raytrace_numpy.ray_direction_basis)):
    #         # remove nan befor assertion
    #         BF_raytrace_numpy.ray_direction_basis[n_ray][n_basis][np.isnan(BF_raytrace_numpy.ray_direction_basis[n_ray][n_basis])] = -10
    #         BF_raytrace_torch.ray_direction_basis[n_basis][n_ray][torch.isnan(BF_raytrace_torch.ray_direction_basis[n_basis][n_ray])] = -10
    #         assert(np.all(np.isclose(BF_raytrace_numpy.ray_direction_basis[n_ray][n_basis], BF_raytrace_torch.ray_direction_basis[n_basis][n_ray]))), f"ray_direction_basis mismatch for ray: {n_ray}, basis: {n_basis}"


# Test Volume creation with random parameters and an experiment with an microscope align optic


@pytest.mark.parametrize("iteration", range(10))
def test_voxel_array_creation(global_data, iteration):

    delta_n = np.random.rand()  # 0.1
    optic_axis = np.random.rand(3)  # [1.0,3.0,1.0]
    if iteration == 0:
        delta_n = 0.1
        optic_axis = [1.0, 0.0, 0.0]

    # Gather global data
    optical_info = copy.deepcopy(global_data["optical_info"])
    volume_shape = optical_info["volume_shape"]

    # Create voxels in different ways
    # Passing a single value for delta n and optic axis
    voxel_numpy_single_value = BirefringentVolume(
        backend=BackEnds.NUMPY,
        optical_info=optical_info,
        Delta_n=delta_n,
        optic_axis=optic_axis,
    )

    voxel_torch_single_value = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=optical_info,
        Delta_n=delta_n,
        optic_axis=optic_axis,
    )

    # Passing an already build 3D array
    voxel_torch = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=optical_info,
        Delta_n=delta_n * torch.ones(volume_shape),
        optic_axis=torch.tensor(optic_axis)
        .unsqueeze(1)
        .unsqueeze(1)
        .unsqueeze(1)
        .repeat(1, volume_shape[0], volume_shape[1], volume_shape[2]),
    )

    # Passing an already build 3D array
    voxel_numpy = BirefringentVolume(
        backend=BackEnds.NUMPY,
        optical_info=optical_info,
        Delta_n=delta_n * torch.ones(volume_shape).numpy(),
        optic_axis=torch.tensor(optic_axis)
        .unsqueeze(1)
        .unsqueeze(1)
        .unsqueeze(1)
        .repeat(1, volume_shape[0], volume_shape[1], volume_shape[2])
        .numpy(),
    )
    voxel_torch = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=optical_info,
        Delta_n=delta_n * torch.ones(volume_shape),
        optic_axis=torch.tensor(optic_axis)
        .unsqueeze(1)
        .unsqueeze(1)
        .unsqueeze(1)
        .repeat(1, volume_shape[0], volume_shape[1], volume_shape[2]),
    )

    # Check that the initialization and normalization of optical axes are correct
    assert np.all(
        np.isclose(
            voxel_numpy.optic_axis.flatten(),
            voxel_torch.optic_axis.detach().numpy().flatten(),
        )
    ), f"Optic axis mismatch between numpy/pytorch delta_n: {delta_n}, optic_axis: {optic_axis}"
    assert np.all(
        np.isclose(
            voxel_numpy.optic_axis.flatten(),
            voxel_numpy_single_value.optic_axis.flatten(),
        )
    ), f"Optic axis mismatch between numpy/pytorch delta_n: {delta_n}, optic_axis: {optic_axis}"
    assert np.all(
        np.isclose(
            voxel_numpy_single_value.optic_axis.flatten(),
            voxel_torch_single_value.optic_axis.detach().numpy().flatten(),
        )
    ), f"Optic axis mismatch between numpy/pytorch delta_n: {delta_n}, optic_axis: {optic_axis}"
    assert np.all(
        np.isclose(
            voxel_numpy_single_value.Delta_n.flatten(),
            voxel_torch_single_value.Delta_n.detach().numpy().flatten(),
        )
    ), f"Delta_n mismatch between numpy/pytorch delta_n: {delta_n}, optic_axis: {optic_axis}"


@pytest.mark.parametrize(
    "volume_shape_in",
    [
        3 * [1],
        3 * [7],
        3 * [8],
        3 * [11],
        3 * [21],
        3 * [51],
    ],
)
def test_compute_JonesMatrices(global_data, volume_shape_in):
    # Define the voxel parameters
    delta_n = 0.1
    optic_axis = [1.0, 0.0, 0]

    # Gather global data
    local_data = copy.deepcopy(global_data)
    optical_info = local_data["optical_info"]
    optical_info["pixels_per_ml"] = 17

    pixels_per_ml = optical_info["pixels_per_ml"]
    volume_shape = volume_shape_in
    optical_info["volume_shape"] = volume_shape

    # Create numpy and pytorch raytracer
    BF_raytrace_numpy = BirefringentRaytraceLFM(optical_info=optical_info)
    BF_raytrace_torch = BirefringentRaytraceLFM(
        backend=BackEnds.PYTORCH, optical_info=optical_info
    )

    # Compute ray-volume geometry and Siddon algorithm
    BF_raytrace_numpy.compute_rays_geometry()
    BF_raytrace_torch.compute_rays_geometry()

    # Create voxel array in numpy
    voxel_numpy = BirefringentVolume(
        backend=BackEnds.NUMPY,
        optical_info=optical_info,
        Delta_n=delta_n,
        optic_axis=optic_axis,
    )

    # Create a voxel array in torch
    voxel_torch = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=optical_info,
        Delta_n=delta_n,
        optic_axis=optic_axis,
    )

    # Create arrays to store images
    ret_img_numpy = np.zeros([pixels_per_ml, pixels_per_ml])
    azi_img_numpy = np.zeros([pixels_per_ml, pixels_per_ml])
    ret_img_torch = np.zeros([pixels_per_ml, pixels_per_ml])
    azi_img_torch = np.zeros([pixels_per_ml, pixels_per_ml])

    # Compute numpy Jones Matrices, by iterating through all the rays and their interaction with the voxel_numpy
    for ii in range(pixels_per_ml):
        for jj in range(pixels_per_ml):
            # Compute JonesMatrix for this ray and the voxels it collides with
            JM_numpy = BF_raytrace_numpy.calc_cummulative_JM_of_ray_numpy(
                ii, jj, voxel_numpy
            )

            # Compute Retardance and Azimuth
            ret_numpy, azi_numpy = BF_raytrace_numpy.retardance(
                JM_numpy
            ), BF_raytrace_numpy.azimuth(JM_numpy)
            ret_img_numpy[ii, jj] = ret_numpy
            azi_img_numpy[ii, jj] = azi_numpy

    # Compute JM with Pytorch implmentation
    JM_torch = BF_raytrace_torch.calc_cummulative_JM_of_ray_torch(voxel_torch)
    # Compute retardance and azimuth
    ret_torch, azi_torch = BF_raytrace_torch.retardance(
        JM_torch
    ), BF_raytrace_torch.azimuth(JM_torch)

    # Fill in retardance and azimuth of torch into an image,
    # And compare with their corresponding numpy JM
    any_fail = False
    for ray_ix in range(BF_raytrace_torch.ray_valid_indices.shape[1]):
        i = BF_raytrace_torch.ray_valid_indices[0, ray_ix]
        j = BF_raytrace_torch.ray_valid_indices[1, ray_ix]
        ret_img_torch[i, j] = ret_torch[ray_ix].item()
        azi_img_torch[i, j] = azi_torch[ray_ix].item()
        JM_numpy = BF_raytrace_numpy.calc_cummulative_JM_of_ray_numpy(i, j, voxel_numpy)

        # Important, set the tolerance to 1e-5, as numpy computes in float64 and torch in float32
        assert np.isclose(
            JM_numpy.astype(np.complex64), JM_torch[ray_ix].detach().numpy(), atol=1e-7
        ).all(), f"JM mismatch on coord: (i,j)= ({i},{j}):"
        # As we are testing for JM, use try_catch on azimuth and retardance, such that they don't brake the test
        # And report if there was mismatch at the end
        try:
            # Check retardance for this ray
            assert np.isclose(
                ret_img_numpy[i, j], ret_img_torch[i, j], atol=1e-7
            ).all(), f"Retardance mismatch on coord: (i,j)= ({i},{j}):"
        except AssertionError as e:
            print(e)
            any_fail = True
        try:
            # Check azimuth for this ray
            check_azimuth_images(
                np.array([azi_img_numpy[i, j]]),
                np.array(azi_img_torch[i, j]),
                message=f"Azimuth mismatch on coord: (i,j)=({i},{j}):",
            )
        except:
            print(f"Azimuth mismatch on coord: (i,j)=({i},{j}):")
            any_fail = True

    # Use this in debug console to visualize errors
    # plot_ret_azi_image_comparison(ret_img_numpy, azi_img_numpy, ret_img_torch, azi_img_torch)
    assert (
        any_fail == False
    ), "No errors in Jones Matrices, but there were mismatches between Retardance and Azimuth in numpy vs torch"


@pytest.mark.parametrize("iteration", range(0, 4))
def test_compute_retardance_and_azimuth_images(global_data, iteration):
    volume_shapes_to_test = [
        3 * [1],
        3 * [7],
        3 * [8],
        3 * [11],
        3 * [21],
        3 * [50],
    ]
    torch.set_grad_enabled(False)
    # Define the voxel parameters
    # delta_n = 0.1
    # optic_axis = [1.0,0.0,0]

    np.random.seed(42)
    delta_n = np.random.uniform(0.01, 0.25, 1)[0]
    optic_axis = np.random.uniform(0.01, 0.25, 3)

    # Gather global data
    optical_info = copy.deepcopy(global_data["optical_info"])
    volume_shape = optical_info["volume_shape"]

    volume_shape = volume_shapes_to_test[iteration]
    # pixels_per_ml = 17

    optical_info["volume_shape"] = volume_shape
    # optical_info['pixels_per_ml'] = pixels_per_ml

    # Create numpy and pytorch raytracer
    BF_raytrace_numpy = BirefringentRaytraceLFM(optical_info=optical_info)
    BF_raytrace_torch = BirefringentRaytraceLFM(
        backend=BackEnds.PYTORCH, optical_info=optical_info
    )

    # Compute ray-volume geometry and Siddon algorithm
    BF_raytrace_numpy.compute_rays_geometry()
    BF_raytrace_torch.compute_rays_geometry()

    # Create voxel array in numpy
    voxel_numpy = BirefringentVolume(
        backend=BackEnds.NUMPY,
        optical_info=optical_info,
        Delta_n=delta_n,
        optic_axis=optic_axis,
    )

    # Create a voxel array in torch
    voxel_torch = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=optical_info,
        Delta_n=delta_n,
        optic_axis=optic_axis,
    )

    # Compute retardance and azimuth images with both methods
    [ret_img_numpy, azi_img_numpy] = BF_raytrace_numpy.ret_and_azim_images(voxel_numpy)
    [ret_img_torch, azi_img_torch] = BF_raytrace_torch.ret_and_azim_images(voxel_torch)
    # Use this in debug console to visualize errors
    # plot_ret_azi_image_comparison(ret_img_numpy, azi_img_numpy, ret_img_torch, azi_img_torch)

    assert not np.any(np.isnan(ret_img_numpy)), "Error in numpy retardance computations nan found"
    assert not np.any(np.isnan(azi_img_numpy)), "Error in numpy azimuth computations nan found"
    assert not torch.any(torch.isnan(ret_img_torch)), "Error in torch retardance computations nan found"
    assert not torch.any(torch.isnan(azi_img_torch)), "Error in torch azimuth computations nan found"

    assert np.all(
        np.isclose(ret_img_numpy.astype(np.float32), ret_img_torch.numpy(), atol=1e-7)
    ), "Error when comparing retardance computations"
    check_azimuth_images(to_numpy(azi_img_numpy), to_numpy(azi_img_torch))


@pytest.mark.parametrize(
    "volume_shape_in",
    [
        # 3*[1],
        3 * [7],
        3 * [8],
        3 * [11],
        # 3*[21], # todo, debug this two examples
        # 3*[51],
    ],
)
def test_forward_projection_lenslet_grid_random_volumes(global_data, volume_shape_in):
    torch.set_grad_enabled(False)
    # Gather global data
    local_data = copy.deepcopy(global_data)
    optical_info = local_data["optical_info"]
    optical_info["aperture_radius_px"] = optical_info["pixels_per_ml"] / 2

    # Volume shape
    volume_shape = volume_shape_in
    optical_info["volume_shape"] = volume_shape

    # The n_micro_lenses defines the active volume area, and it should be
    #   smaller than the volume_shape.
    # This as some rays go beyond the volume in front of a single micro-lens
    optical_info["n_micro_lenses"] = volume_shape[1] - 6
    optical_info["n_voxels_per_ml"] = 1

    # Create Ray-tracing objects
    BF_raytrace_numpy = BirefringentRaytraceLFM(optical_info=optical_info)
    BF_raytrace_torch = BirefringentRaytraceLFM(
        backend=BackEnds.PYTORCH, optical_info=optical_info
    )

    BF_raytrace_numpy.compute_rays_geometry()
    BF_raytrace_torch.compute_rays_geometry()

    # Generate a volume with random everywhere
    voxel_torch_random = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=optical_info,
        volume_creation_args={"init_mode": "random"},
    )

    # Copy the volume, to have exactly the same things
    voxel_numpy_random = BirefringentVolume(
        backend=BackEnds.NUMPY,
        optical_info=optical_info,
        Delta_n=voxel_torch_random.get_delta_n().numpy(),
        optic_axis=voxel_torch_random.get_optic_axis().numpy(),
    )

    assert (
        BF_raytrace_numpy.optical_info == voxel_numpy_random.optical_info
    ), "Mismatch on RayTracer and volume optical_info numpy"
    assert (
        BF_raytrace_torch.optical_info == voxel_torch_random.optical_info
    ), "Mismatch on RayTracer and volume optical_info torch"

    with np.errstate(divide="raise"):
        [ret_img_numpy, azi_img_numpy] = BF_raytrace_numpy.ray_trace_through_volume(
            voxel_numpy_random
        )
    [ret_img_torch, azi_img_torch] = BF_raytrace_torch.ray_trace_through_volume(
        voxel_torch_random
    )

    plot_ret_azi_image_comparison(
        ret_img_numpy, azi_img_numpy, ret_img_torch, azi_img_torch
    )

    assert not np.any(np.isnan(ret_img_numpy)), "Error in numpy retardance computations nan found"
    assert not np.any(np.isnan(azi_img_numpy)), "Error in numpy azimuth computations nan found"
    assert not torch.any(torch.isnan(ret_img_torch)), "Error in torch retardance computations nan found"
    assert not torch.any(torch.isnan(azi_img_torch)), "Error in torch azimuth computations nan found"

    assert np.all(
        np.isclose(ret_img_numpy.astype(np.float32), ret_img_torch.numpy(), atol=1e-7)
    ), "Error when comparing retardance computations"
    check_azimuth_images(to_numpy(azi_img_numpy), to_numpy(azi_img_torch))


@pytest.mark.parametrize(
    "volume_init_mode", ["single_voxel", "random", "ellipsoid", "1planes", "3planes"]
)
def test_forward_projection_different_volumes(global_data, volume_init_mode):
    torch.set_grad_enabled(False)
    # Gather global data
    local_data = copy.deepcopy(global_data)
    optical_info = local_data["optical_info"]

    # Volume shape
    volume_shape = [7, 7, 7]
    optical_info["volume_shape"] = volume_shape

    # The n_micro_lenses defines the active volume area, and it should be smaller than the volume_shape.
    # This as some rays go beyond the volume in front of a single micro-lens
    optical_info["n_micro_lenses"] = volume_shape[1] - 4
    optical_info["n_voxels_per_ml"] = 1
    optical_info["pixels_per_ml"] = 17

    # Create Ray-tracing objects
    BF_raytrace_numpy = BirefringentRaytraceLFM(optical_info=optical_info)
    BF_raytrace_torch = BirefringentRaytraceLFM(
        backend=BackEnds.PYTORCH, optical_info=optical_info
    )

    BF_raytrace_numpy.compute_rays_geometry()
    BF_raytrace_torch.compute_rays_geometry()

    # Generate a volume with random everywhere
    voxel_torch_random = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=optical_info,
        volume_creation_args={"init_mode": volume_init_mode},
    )
    # Copy the volume, to have exactly the same things
    voxel_numpy_random = BirefringentVolume(
        backend=BackEnds.NUMPY,
        optical_info=optical_info,
        Delta_n=voxel_torch_random.get_delta_n().numpy(),
        optic_axis=voxel_torch_random.get_optic_axis().numpy(),
    )

    assert (
        BF_raytrace_numpy.optical_info == voxel_numpy_random.optical_info
    ), "Mismatch on RayTracer and volume optical_info numpy"
    assert (
        BF_raytrace_torch.optical_info == voxel_torch_random.optical_info
    ), "Mismatch on RayTracer and volume optical_info torch"

    with np.errstate(divide="raise"):
        [ret_img_numpy, azi_img_numpy] = BF_raytrace_numpy.ray_trace_through_volume(
            voxel_numpy_random
        )
    [ret_img_torch, azi_img_torch] = BF_raytrace_torch.ray_trace_through_volume(
        voxel_torch_random
    )

    # plot_ret_azi_image_comparison(
    #     ret_img_numpy, azi_img_numpy, ret_img_torch, azi_img_torch
    # )
    # plt.show(block=True)

    assert not np.any(np.isnan(ret_img_numpy)), "Error in numpy retardance computations nan found"
    assert not np.any(np.isnan(azi_img_numpy)), "Error in numpy azimuth computations nan found"
    assert not torch.any(torch.isnan(ret_img_torch)), "Error in torch retardance computations nan found"
    assert not torch.any(torch.isnan(azi_img_torch)), "Error in torch azimuth computations nan found"

    assert np.all(
        np.isclose(ret_img_numpy.astype(np.float32), ret_img_torch.numpy(), atol=1e-7)
    ), "Error when comparing retardance computations"
    check_azimuth_images(to_numpy(azi_img_numpy), to_numpy(azi_img_torch), atol=1e-5)


@pytest.mark.parametrize("n_voxels_per_ml", [1, 3, 4])
def test_forward_projection_different_super_samplings(global_data, n_voxels_per_ml):
    torch.set_grad_enabled(False)
    # Gather global data
    local_data = copy.deepcopy(global_data)
    optical_info = local_data["optical_info"]

    # Volume shape
    volume_shape = [7, 27, 27]
    optical_info["volume_shape"] = volume_shape

    # The n_micro_lenses defines the active volume area, and it should be smaller than the volume_shape.
    # This as some rays go beyond the volume in front of a single micro-lens
    optical_info["n_micro_lenses"] = 5
    optical_info["n_voxels_per_ml"] = n_voxels_per_ml
    optical_info["pixels_per_ml"] = 17

    # Create Ray-tracing objects
    BF_raytrace_numpy = BirefringentRaytraceLFM(optical_info=optical_info)
    BF_raytrace_torch = BirefringentRaytraceLFM(
        backend=BackEnds.PYTORCH, optical_info=optical_info
    )

    BF_raytrace_numpy.compute_rays_geometry()
    BF_raytrace_torch.compute_rays_geometry()

    # Generate a volume with random everywhere
    voxel_torch_random = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=optical_info,
        volume_creation_args={"init_mode": "ellipsoid"},
    )
    # Copy the volume, to have exactly the same things
    voxel_numpy_random = BirefringentVolume(
        backend=BackEnds.NUMPY,
        optical_info=optical_info,
        Delta_n=voxel_torch_random.get_delta_n().numpy(),
        optic_axis=voxel_torch_random.get_optic_axis().numpy(),
    )

    assert (
        BF_raytrace_numpy.optical_info == voxel_numpy_random.optical_info
    ), "Mismatch on RayTracer and volume optical_info numpy"
    assert (
        BF_raytrace_torch.optical_info == voxel_torch_random.optical_info
    ), "Mismatch on RayTracer and volume optical_info torch"

    with np.errstate(divide="raise"):
        [ret_img_numpy, azi_img_numpy] = BF_raytrace_numpy.ray_trace_through_volume(
            voxel_numpy_random
        )
    [ret_img_torch, azi_img_torch] = BF_raytrace_torch.ray_trace_through_volume(
        voxel_torch_random
    )
    [ret_img_torch, azi_img_torch] = BF_raytrace_torch.ray_trace_through_volume(
        voxel_torch_random
    )

    plot_ret_azi_image_comparison(
        ret_img_numpy, azi_img_numpy, ret_img_torch, azi_img_torch
    )

    assert not np.any(np.isnan(ret_img_numpy)), "Error in numpy retardance computations nan found"
    assert not np.any(np.isnan(azi_img_numpy)), "Error in numpy azimuth computations nan found"
    assert not torch.any(torch.isnan(ret_img_torch)), "Error in torch retardance computations nan found"
    assert not torch.any(torch.isnan(azi_img_torch)), "Error in torch azimuth computations nan found"

    assert np.all(
        np.isclose(ret_img_numpy.astype(np.float32), ret_img_torch.numpy(), atol=1e-7)
    ), "Error when comparing retardance computations"

    check_azimuth_images(to_numpy(azi_img_numpy), to_numpy(azi_img_torch), atol=1e-6)


@pytest.mark.parametrize(
    "volume_init_mode", ["single_voxel", "random", "ellipsoid", "1planes", "3planes"]
)
def test_torch_auto_differentiation(global_data, volume_init_mode):
    # Enable torch gradients
    torch.set_grad_enabled(True)
    # Gather global data
    local_data = copy.deepcopy(global_data)
    optical_info = local_data["optical_info"]

    # Simplify the settings, so it's fast to compute
    volume_shape = [7, 7, 7]
    optical_info["volume_shape"] = volume_shape
    optical_info["pixels_per_ml"] = 17
    optical_info["n_micro_lenses"] = 3

    BF_raytrace_torch = BirefringentRaytraceLFM(
        backend=BackEnds.PYTORCH, optical_info=optical_info
    )
    BF_raytrace_torch.compute_rays_geometry()

    volume_torch_random = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=optical_info,
        volume_creation_args={"init_mode": volume_init_mode},
    )
    # make volume trainable, by telling waveblocks which variables to optimize
    volume_torch_random.members_to_learn.append("Delta_n")
    volume_torch_random.members_to_learn.append("optic_axis")

    # Create optimizer, which will take care of applying the gradients to the volume once computed
    optimizer = torch.optim.Adam(volume_torch_random.get_trainable_variables(), lr=1e-1)
    # Set all gradients to zero
    optimizer.zero_grad()

    # Compute a forward projection
    [ret_image, azim_image] = BF_raytrace_torch.ray_trace_through_volume(
        volume_torch_random
    )
    # Calculate a loss, for example minimizing the mean of both images
    L = ret_image.mean() + azim_image.mean()

    # Compute and propagate the gradients
    L.backward()

    # TODO: check that the none of the gradients are nans
    # Check if the gradients where properly propagated
    assert (
        volume_torch_random.Delta_n.grad is not None
    ), "Gradients were not propagated to the volumes Delta_n correctly"
    assert (
        volume_torch_random.optic_axis.grad is not None
    ), "Gradients were not propagated to the volumes optic_axis correctly"

    # Calculate the volume values before updating the values with the gradients
    Delta_n_sum_initial = volume_torch_random.Delta_n.sum()
    optic_axis_sum_initial = volume_torch_random.optic_axis.sum()

    # Update the volume
    optimizer.step()

    # Calculate the volume values before updating the values with the gradients
    Delta_n_sum_final = volume_torch_random.Delta_n.sum()
    optic_axis_sum_final = volume_torch_random.optic_axis.sum()

    # TODO: check that none of the gradients are nans
    # These should be different
    assert (
        Delta_n_sum_initial != Delta_n_sum_final
    ), "Seems like the volume was not updated correctly. Nothing changed in the volume after the optimization step."
    assert (
        optic_axis_sum_initial != optic_axis_sum_final
    ), "Seems like the volume was not updated correctly. Nothing changed in the volume after the optimization step."


@pytest.mark.parametrize(
    "volume_init_mode", ["single_voxel", "random", "ellipsoid", "1planes", "3planes"]
)
def test_torch_auto_differentiation_subsets(global_data, volume_init_mode):
    # Enable torch gradients
    torch.set_grad_enabled(True)
    # Gather global data
    local_data = copy.deepcopy(global_data)
    optical_info = local_data["optical_info"]

    # Simplify the settings, so it's fast to compute
    volume_shape = [7, 7, 7]
    optical_info["volume_shape"] = volume_shape
    optical_info["pixels_per_ml"] = 17
    optical_info["n_micro_lenses"] = 3

    BF_raytrace_torch = BirefringentRaytraceLFM(
        backend=BackEnds.PYTORCH, optical_info=optical_info
    )
    BF_raytrace_torch.compute_rays_geometry()

    volume_torch_random = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=optical_info,
        volume_creation_args={"init_mode": volume_init_mode},
    )
    num_elements = volume_torch_random.Delta_n.shape[0]
    num_ones = num_elements // 2
    num_ones += num_elements % 2
    mask = torch.zeros(num_elements, dtype=torch.bool)
    mask[:num_ones] = True
    active_indices = torch.where(mask)[0]
    volume_torch_random.indices_active = active_indices
    max_index = mask.size()[0]
    idx_tensor = torch.full((max_index + 1,), -1, dtype=torch.long)
    positions = torch.arange(len(active_indices), dtype=torch.long)
    idx_tensor[active_indices] = positions
    volume_torch_random.active_idx2spatial_idx_tensor = idx_tensor
    volume_torch_random.optic_axis_active = torch.nn.Parameter(
        volume_torch_random.optic_axis[:, active_indices]
    )
    volume_torch_random.birefringence_active = torch.nn.Parameter(
        volume_torch_random.Delta_n[active_indices]
    )
    # make volume trainable, by telling waveblocks which variables to optimize
    volume_torch_random.members_to_learn.append("birefringence_active")
    volume_torch_random.members_to_learn.append("optic_axis_active")

    # Create optimizer, which will take care of applying the gradients to the volume once computed
    optimizer = torch.optim.Adam(volume_torch_random.get_trainable_variables(), lr=1e-1)
    # Set all gradients to zero
    optimizer.zero_grad()

    # Compute a forward projection
    [ret_image, azim_image] = BF_raytrace_torch.ray_trace_through_volume(
        volume_torch_random
    )
    # Calculate a loss, for example minimizing the mean of both images
    L = ret_image.mean() + azim_image.mean()

    # Compute and propagate the gradients
    L.backward()

    # TODO: check that the none of the gradients are nans
    # Check if the gradients where properly propagated
    assert (
        volume_torch_random.birefringence_active.grad is not None
    ), "Gradients were not propagated to the volumes Delta_n correctly"
    assert (
        volume_torch_random.optic_axis_active.grad is not None
    ), "Gradients were not propagated to the volumes optic_axis correctly"

    # Calculate the volume values before updating the values with the gradients
    Delta_n_sum_initial = volume_torch_random.birefringence_active.sum()
    optic_axis_sum_initial = volume_torch_random.optic_axis_active.sum()

    # Update the volume
    optimizer.step()

    # Calculate the volume values before updating the values with the gradients
    Delta_n_sum_final = volume_torch_random.birefringence_active.sum()
    optic_axis_sum_final = volume_torch_random.optic_axis_active.sum()

    # These should be different
    assert (
        Delta_n_sum_initial != Delta_n_sum_final
    ), "Seems like the volume was not updated correctly. Nothing changed in the volume after the optimization step."
    assert (
        optic_axis_sum_initial != optic_axis_sum_final
    ), "Seems like the volume was not updated correctly. Nothing changed in the volume after the optimization step."


def speed_speed(global_data, volume_init_mode):
    # Enable torch gradients
    torch.set_grad_enabled(True)
    # Gather global data
    local_data = copy.deepcopy(global_data)
    optical_info = local_data["optical_info"]

    # Simplify the settings, so it's fast to compute
    volume_shape = [3, 13, 13]
    optical_info["volume_shape"] = volume_shape
    optical_info["pixels_per_ml"] = 17
    optical_info["n_micro_lenses"] = 5

    BF_raytrace_torch = BirefringentRaytraceLFM(
        backend=BackEnds.PYTORCH, optical_info=optical_info
    )
    BF_raytrace_torch.compute_rays_geometry()

    volume_torch_random = BF_raytrace_torch.init_volume(
        volume_shape, init_mode=volume_init_mode
    )
    # make volume trainable, by telling waveblocks which variables to optimize
    volume_torch_random.members_to_learn.append("Delta_n")
    volume_torch_random.members_to_learn.append("optic_axis")

    # Create optimizer, which will take care of applying the gradients to the volume once computed
    optimizer = torch.optim.Adam(volume_torch_random.get_trainable_variables(), lr=1e-1)
    # Set all gradients to zero
    optimizer.zero_grad()

    [ret_image, azim_image] = BF_raytrace_torch.ray_trace_through_volume(
        volume_torch_random
    )

    import torch.autograd.profiler as profiler

    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        # Compute a forward projection
        [ret_image, azim_image] = BF_raytrace_torch.ray_trace_through_volume(
            volume_torch_random
        )
        # Calculate a loss, for example minimizing the mean of both images
        # L = ret_image.mean() + azim_image.mean()

        # Compute and propagate the gradients
        # L.backward()

        # # Check if the gradients where properly propagated
        # assert volume_torch_random.Delta_n.grad.sum() != 0, 'Gradients were not propagated to the volumes Delta_n correctly'
        # assert volume_torch_random.optic_axis.grad.sum() != 0, 'Gradients were not propagated to the volumes optic_axis correctly'

        # # Calculate the volume values before updating the values with the gradients
        # Delta_n_sum_initial = volume_torch_random.Delta_n.sum()
        # optic_axis_sum_initial = volume_torch_random.optic_axis.sum()

        # Update the volume
        # optimizer.step()
    print(
        prof.key_averages(group_by_stack_n=5).table(
            sort_by="self_cpu_time_total", row_limit=10
        )
    )
    prof.export_chrome_trace("forward_trace.trace")
    # # Calculate the volume values before updating the values with the gradients
    # Delta_n_sum_final = volume_torch_random.Delta_n.sum()
    # optic_axis_sum_final = volume_torch_random.optic_axis.sum()

    # # These should be different
    # assert Delta_n_sum_initial != Delta_n_sum_final, 'Seems like the volume wasnt updated correctly, nothing changed in the volume after the optimization step'
    # assert optic_axis_sum_initial != optic_axis_sum_final, 'Seems like the volume wasnt updated correctly, nothing changed in the volume after the optimization step'


def test_azimuth_neg_birefringence(global_data):
    """Verify that the effects of the birefringence being of opposite sign."""
    torch.set_grad_enabled(False)
    # Gather global data
    local_data = copy.deepcopy(global_data)
    optical_info = local_data["optical_info"]

    # Volume shape
    volume_shape = [3, 3, 3]
    optical_info["volume_shape"] = volume_shape

    # The n_micro_lenses defines the active volume area, and it should be smaller than the volume_shape.
    # This as some rays go beyond the volume in front of a single micro-lens
    optical_info["n_micro_lenses"] = 1
    optical_info["n_voxels_per_ml"] = 1
    optical_info["pixels_per_ml"] = 17

    # Create Ray-tracing objects
    BF_raytrace_numpy = BirefringentRaytraceLFM(optical_info=optical_info)
    BF_raytrace_torch = BirefringentRaytraceLFM(
        backend=BackEnds.PYTORCH, optical_info=optical_info
    )

    BF_raytrace_numpy.compute_rays_geometry()
    BF_raytrace_torch.compute_rays_geometry()

    # Create birefringence voxel volumes
    voxel_args_pos = {
        "init_mode": "single_voxel",
        "init_args": {"delta_n": 0.05, "offset": [0, 0, 0]},
    }
    voxel_args_neg = {
        "init_mode": "single_voxel",
        "init_args": {"delta_n": -0.05, "offset": [0, 0, 0]},
    }
    voxel_pos_torch = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=optical_info,
        volume_creation_args=voxel_args_pos,
    )
    voxel_neg_torch = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=optical_info,
        volume_creation_args=voxel_args_neg,
    )
    voxel_pos_numpy = BirefringentVolume(
        backend=BackEnds.NUMPY,
        optical_info=optical_info,
        volume_creation_args=voxel_args_pos,
    )
    voxel_neg_numpy = BirefringentVolume(
        backend=BackEnds.NUMPY,
        optical_info=optical_info,
        volume_creation_args=voxel_args_neg,
    )

    # Verify wtih pytorch
    [ret_img_pos, azi_img_pos] = BF_raytrace_torch.ray_trace_through_volume(
        voxel_pos_torch
    )
    [ret_img_neg, azi_img_neg] = BF_raytrace_torch.ray_trace_through_volume(
        voxel_neg_torch
    )
    rotated_azi_img_neg = torch.flip(azi_img_neg.permute(1, 0), [1])
    plot_ret_azi_flipsign_image_comparison(
        ret_img_pos.numpy(),
        azi_img_pos.numpy(),
        ret_img_neg.numpy(),
        rotated_azi_img_neg.numpy(),
    )
    assert np.all(
        np.isclose(ret_img_pos.numpy(), ret_img_neg.numpy(), atol=1e-7)
    ), "Retardance depends on the sign of the birefrigence."
    check_azimuth_images(
        to_numpy(azi_img_pos),
        to_numpy(rotated_azi_img_neg),
        atol=1e-6,
        message="Flipping the sign of the birefringence does not simply rotate the azimuth.",
    )

    # Verify with numpy
    with np.errstate(divide="raise"):
        [ret_img_pos, azi_img_pos] = BF_raytrace_numpy.ray_trace_through_volume(
            voxel_pos_numpy
        )
        [ret_img_neg, azi_img_neg] = BF_raytrace_numpy.ray_trace_through_volume(
            voxel_neg_numpy
        )
    rotated_azi_img_neg = np.flip(azi_img_neg.T, axis=1)
    plot_ret_azi_flipsign_image_comparison(
        ret_img_pos, azi_img_pos, ret_img_neg, rotated_azi_img_neg
    )
    assert np.all(
        np.isclose(
            ret_img_pos.astype(np.float32), ret_img_neg.astype(np.float32), atol=1e-7
        )
    ), "Retardance depends on the sign of the birefrigence."
    check_azimuth_images(
        to_numpy(azi_img_pos),
        to_numpy(rotated_azi_img_neg),
        atol=1e-6,
        message="Flipping the sign of the birefringence does not simply rotate the azimuth.",
    )


def test_intensity_with_both_methods(global_data):
    """Verify that the effects of the birefringence being of opposite sign."""
    torch.set_grad_enabled(False)
    # Gather global data
    local_data = copy.deepcopy(global_data)

    # Select backend method
    backends = [BackEnds.NUMPY, BackEnds.PYTORCH]
    results = []
    for backend in backends:
        optical_info = local_data["optical_info"]
        optical_info['aperture_radius_ml'] = 3
        # LC-PolScope setup
        optical_info["analyzer"] = JonesMatrixGenerators.left_circular_polarizer()
        optical_info["polarizer_swing"] = 0.03

        # Create a Birefringent Raytracer!
        rays = BirefringentRaytraceLFM(backend=backend, optical_info=optical_info)

        # Compute the rays and use the Siddon's algorithm to compute the intersections with voxels.
        # If a filepath is passed as argument, the object with all its calculations
        #   get stored/loaded from a file
        rays.compute_rays_geometry()

        # Load volume from a file
        loaded_volume = BirefringentVolume.init_from_file(
            "data/objects/shell.h5", backend, optical_info
        )
        my_volume = loaded_volume

        image_list = rays.ray_trace_through_volume(my_volume, intensity=True)
        # plot_intensity_images(image_list)
        # plt.show(block=True)

        if backend == BackEnds.PYTORCH:
            image_list = [img.detach().cpu().numpy() for img in image_list]
        results.append(image_list)
    for nSetting in range(5):
        check_azimuth_images(results[0][nSetting], results[1][nSetting])


def test_retardance_and_azimuth_from_intensity(global_data):
    """Test that image lists transformed into retardance and azimuth using ret_and_azim_from_intensity 
       match the output of ray_trace_through_volume with intensity=False."""
    
    # Disable gradient tracking for efficiency during tests
    torch.set_grad_enabled(False)

    # Gather global data
    local_data = copy.deepcopy(global_data)

    # Select backend methods to compare
    backends = [BackEnds.NUMPY, BackEnds.PYTORCH]
    results = []
    swing = 0.03  # Polarizer swing value used in the LC-PolScope setup

    for backend in backends:
        optical_info = local_data["optical_info"]
        optical_info['aperture_radius_ml'] = 3
        optical_info["analyzer"] = JonesMatrixGenerators.left_circular_polarizer()
        optical_info["polarizer_swing"] = swing

        # Create a Birefringent Raytracer
        rays = BirefringentRaytraceLFM(backend=backend, optical_info=optical_info)

        # Compute rays geometry and voxel intersections using Siddon's algorithm
        rays.compute_rays_geometry()

        # Load volume data
        loaded_volume = BirefringentVolume.init_from_file(
            "data/objects/shell.h5", backend, optical_info
        )
        my_volume = loaded_volume

        # Trace rays through the volume and gather intensity images
        image_list = rays.ray_trace_through_volume(my_volume, intensity=True)

        if backend == BackEnds.PYTORCH:
            # Convert PyTorch tensors to numpy arrays for comparison
            image_list = [img.detach().cpu().numpy() for img in image_list]
        
        # Transform the intensity images into retardance and azimuth images
        ret_intensity, azim_intensity = ret_and_azim_from_intensity(image_list, swing)

        # Trace rays through the volume with intensity=False to get direct retardance and azimuth
        ret_direct, azim_direct = rays.ray_trace_through_volume(my_volume, intensity=False)

        # Store results for both methods
        results.append((ret_intensity, azim_intensity, ret_direct, azim_direct))

    # Verify that the transformed intensity results match the direct method results
    for ret_intensity, azim_intensity, ret_direct, azim_direct in results:
        np.testing.assert_allclose(ret_intensity, ret_direct, rtol=1e-5, atol=1e-7, 
                                   err_msg="Retardance images do not match")
        azim_intensity[np.abs(azim_intensity) < 0.094] = 0
        check_azimuth_images(azim_intensity, azim_direct, atol=1e-7, 
                            message="Azimuth images do not match")


def main():
    # test_torch_auto_differentiation(global_data(), '1planes')
    # test_torch_auto_differentiation(global_data(), 'ellipsoid')
    test_intensity_with_both_methods(global_data())
    # speed_speed(global_data(), 'ellipsoid')
    # test_forward_projection_lenslet_grid_random_volumes(global_data(), 3*[8])
    # Multi lenslet example
    # test_forward_projection_different_volumes(global_data(), 'ellipsoid')
    # test_forward_projection_lenslet_grid_random_volumes(global_data(), 3*[51])
    # test_rays_computation(global_data(), 17)
    # test_compute_JonesMatrices(global_data(), 3*[11])
    import sys

    sys.exit()
    # test_compute_JonesMatrices(global_data(), 3*[1])
    # test_voxel_array_creation(global_data(),1)
    # Objective configuration


def to_numpy(x):
    # Convert torch tensors to numpy arrays if needed
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def check_azimuth_images(
    img1, img2, atol=1e-7, message="Error when comparing azimuth computations"
):
    """Compares two azimuth images, taking into account that atan2 output
    of 0 and pi is equivalent"""
    img1 = to_numpy(img1)
    img2 = to_numpy(img2)
    if not np.all(np.isclose(img1, img2, atol=atol)):
        diff = np.abs(img1 - img2)
        # Check if the difference is a multiple of pi
        # Only consider differences that are not close to zero
        non_zero_diff = diff[~np.isclose(diff, 0.0, atol=atol)]
        # plot_azimuth(diff)
        
        # If there are non-zero differences, check if they are close to pi
        if non_zero_diff.size > 0:
            np.testing.assert_allclose(non_zero_diff, np.pi, atol=atol, 
                                       err_msg="Azimuth differences are not close to pi")


def plot_azimuth(img):
    """Plot azimuth image, with polar coordinates"""
    ctr = [(img.shape[0] - 1) / 2, (img.shape[1] - 1) / 2]
    i = np.linspace(0, img.shape[0] - 1, img.shape[0])
    j = np.linspace(0, img.shape[0] - 1, img.shape[0])
    jv, iv = np.meshgrid(i, j)
    dist_from_ctr = np.sqrt((iv - ctr[0]) ** 2 + (jv - ctr[1]) ** 2)

    fig = plt.figure(figsize=(13, 5))
    fig.subplots_adjust(bottom=0, left=0.025, top=0.925, right=0.975)

    plt.rcParams["image.origin"] = "upper"
    plt.clf()
    sub1 = plt.subplot(1, 3, 1)
    sub1.imshow(dist_from_ctr)
    plt.title("Distance from center")

    sub2 = plt.subplot(1, 3, 2)
    cax = sub2.imshow(img * 180 / np.pi)
    plt.colorbar()
    plt.title("Azimuth (degrees)")
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    # vertically oriented colorbar
    # cbar.ax.set_yticklabels(['-pi', '-pi/2', '-pi/4', '-pi/8', '0', 'pi/8', 'pi/4', 'pi/2', 'pi'])
    cbar = fig.colorbar(
        cax,
        ticks=np.rad2deg(
            [
                -np.pi,
                -np.pi / 2,
                -np.pi / 3,
                -np.pi / 4,
                -np.pi / 6,
                0,
                np.pi / 6,
                np.pi / 4,
                np.pi / 3,
                np.pi / 2,
                np.pi,
            ]
        ),
    )
    cbar.ax.set_yticklabels(
        ["-180", "-90", "-60", "-45", "-30", "0", "30", "45", "60", "90", "180"]
    )

    # plt.subplot(1, 3, 3)
    # plt.polar(dist_from_ctr, img / np.pi)
    sub3 = plt.subplot(1, 3, 3)
    sub3.scatter(dist_from_ctr, img * 180 / np.pi)
    # plt.colorbar()
    plt.title("Azimuth in polar coordinates")
    plt.xlabel("Distance from center pixel")
    plt.ylabel("Azimuth")

    plt.pause(0.05)
    plt.show(block=True)


def plot_ret_azi_image_comparison(
    ret_img_numpy, azi_img_numpy, ret_img_torch, azi_img_torch
):
    # Check if we are running in pytest
    import os

    if "PYTEST_CURRENT_TEST" in os.environ:
        return
    plt.rcParams["image.origin"] = "upper"
    plt.clf()
    plt.subplot(3, 2, 1)
    plt.imshow(ret_img_numpy)
    plt.title("Ret. numpy")
    plt.subplot(3, 2, 2)
    plt.imshow(azi_img_numpy)
    plt.title("Azi. numpy")

    plt.subplot(3, 2, 3)
    plt.imshow(ret_img_torch)
    plt.title("Ret. torch")
    plt.subplot(3, 2, 4)
    plt.imshow(azi_img_torch)
    plt.title("Azi. torch")

    plt.subplot(3, 2, 5)
    diff = np.abs(ret_img_torch - ret_img_numpy)
    plt.imshow(diff)
    plt.title(f"Ret. Diff Sum: {diff.sum():.3f}")

    plt.subplot(3, 2, 6)
    diff = np.abs(azi_img_torch - azi_img_numpy)
    plt.imshow(diff)
    plt.title(f"Azi. Diff Sum: {diff.sum():.3f}")
    
    plt.tight_layout()
    plt.pause(0.05)
    plt.show(block=True)


def plot_ret_azi_flipsign_image_comparison(
    ret_img_pos, azi_img_pos, ret_img_neg, azi_img_neg
):
    # Check if we are running in pytest
    import os

    if "PYTEST_CURRENT_TEST" in os.environ:
        return
    plt.rcParams["image.origin"] = "upper"
    plt.clf()
    plt.subplot(3, 2, 1)
    plt.imshow(ret_img_pos)
    plt.title("Ret. pos bir")
    plt.subplot(3, 2, 2)
    plt.imshow(azi_img_pos, cmap="twilight")
    plt.title("Azi. pos bir")

    plt.subplot(3, 2, 3)
    plt.imshow(ret_img_neg)
    plt.title("Ret. neg bir")
    plt.subplot(3, 2, 4)
    plt.imshow(azi_img_neg, cmap="twilight")
    plt.title("Azi. neg bir")

    plt.subplot(3, 2, 5)
    diff = np.abs(ret_img_pos - ret_img_neg)
    plt.imshow(diff)
    plt.title(f"Ret. Diff: {diff.sum():.3f}")

    plt.subplot(3, 2, 6)
    diff = np.abs(azi_img_pos - azi_img_neg)
    plt.imshow(diff)
    plt.title(f"Azi. Diff: {diff.sum():.3f}")
    plt.tight_layout()
    plt.pause(0.05)
    plt.show(block=True)


if __name__ == "__main__":
    main()
