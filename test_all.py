import pytest
from VolumeRaytraceLFM.birefringence_implementations import *
import matplotlib.pyplot as plt
import copy
import os


@pytest.fixture(scope = 'module')
def global_data():
    '''Create global optic_setting and optical_info containing all the optics and volume information
        The tests can access this by passing the name of this function as an argument for example:
        def test_something(global_data):
            optical_info = global_data['optical_info]
    '''
    
    # Set torch precision to Double to match numpy 
    torch.set_default_tensor_type(torch.DoubleTensor)
    
    # Fetch default optical info
    optical_info = OpticalElement.get_optical_info_template()

    optical_info['volume_shape'] = [11, 11, 11]
    optical_info['axial_voxel_size_um'] = 1.0
    optical_info['pixels_per_ml'] = 5
    optical_info['na_obj'] = 1.2
    optical_info['n_medium'] = 1.52
    optical_info['wavelength'] = 0.550
    optical_info['n_micro_lenses'] = 1
    optical_info['n_voxels_per_ml'] = 1
    optical_info['polarizer'] = np.array([[1, 0], [0, 1]])
    optical_info['analyzer'] = np.array([[1, 0], [0, 1]])

    return {'optical_info' : optical_info}

# Test systems with different number of rays per micro-lens
@pytest.mark.parametrize('pixels_per_ml_init', [3,5,10,17,33])
def test_rays_computation(global_data, pixels_per_ml_init):
    
    # Gather global data
    optical_info = copy.deepcopy(global_data['optical_info'])

    optical_info['pixels_per_ml'] = pixels_per_ml_init

    # Create numpy and pytorch raytracer
    BF_raytrace_numpy = BirefringentRaytraceLFM(optical_info=optical_info)
    BF_raytrace_torch = BirefringentRaytraceLFM(backend=BackEnds.PYTORCH, optical_info=optical_info)
    
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

    assert np.all(np.isclose(BF_raytrace_numpy.ray_entry,BF_raytrace_torch.ray_entry.numpy())),         "ray_entry calculation mismatch between Numpy and Pytorch back-end"
    assert np.all(np.isclose(BF_raytrace_numpy.ray_exit,BF_raytrace_torch.ray_exit.numpy())),           "ray_exit calculation mismatch between Numpy and Pytorch back-end"
    assert np.all(np.isclose(BF_raytrace_numpy.ray_direction,BF_raytrace_torch.ray_direction.numpy())), "ray_direction calculation mismatch between Numpy and Pytorch back-end"
    
    # todo: This is hard to compare, as we only store the valid rays in pytorch vs numpy that we store all rays
    # for n_basis in range(3):
    #     for n_ray in range(len(BF_raytrace_numpy.ray_direction_basis)):
    #         # remove nan befor assertion
    #         BF_raytrace_numpy.ray_direction_basis[n_ray][n_basis][np.isnan(BF_raytrace_numpy.ray_direction_basis[n_ray][n_basis])] = -10
    #         BF_raytrace_torch.ray_direction_basis[n_basis][n_ray][torch.isnan(BF_raytrace_torch.ray_direction_basis[n_basis][n_ray])] = -10
    #         assert(np.all(np.isclose(BF_raytrace_numpy.ray_direction_basis[n_ray][n_basis], BF_raytrace_torch.ray_direction_basis[n_basis][n_ray]))), f"ray_direction_basis mismatch for ray: {n_ray}, basis: {n_basis}"

# Test Volume creation with random parameters and an experiment with an microscope align optic 
@pytest.mark.parametrize('iteration', range(10))
def test_voxel_array_creation(global_data, iteration):
    
    delta_n = np.random.rand()# 0.1
    optic_axis = np.random.rand(3) #[1.0,3.0,1.0]
    if iteration==0:
        delta_n = 0.1
        optic_axis = [1.0,0.0,0.0]

    # Gather global data
    optical_info = copy.deepcopy(global_data['optical_info'])
    volume_shape = optical_info['volume_shape']
    
    # Create voxels in different ways
    # Passing a single value for delta n and optic axis
    voxel_numpy_single_value = BirefringentVolume(backend=BackEnds.NUMPY, optical_info=optical_info, 
                                    Delta_n=delta_n, optic_axis=optic_axis)

    voxel_torch_single_value = BirefringentVolume(backend=BackEnds.PYTORCH, optical_info=optical_info,
                                    Delta_n=delta_n, optic_axis=optic_axis)
                                    
    # Passing an already build 3D array                            
    voxel_torch = BirefringentVolume(backend=BackEnds.PYTORCH, optical_info=optical_info,
                                    Delta_n=delta_n*torch.ones(volume_shape), 
                                    optic_axis=torch.tensor(optic_axis).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, volume_shape[0], volume_shape[1], volume_shape[2])
                                    )
                                    
    # Passing an already build 3D array                            
    voxel_numpy = BirefringentVolume(backend=BackEnds.NUMPY, optical_info=optical_info,
                                    Delta_n=delta_n*torch.ones(volume_shape).numpy(), 
                                    optic_axis=torch.tensor(optic_axis).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, volume_shape[0], volume_shape[1], volume_shape[2]).numpy()
                                    )
    voxel_torch = BirefringentVolume(backend=BackEnds.PYTORCH, optical_info=optical_info,
                                    Delta_n=delta_n*torch.ones(volume_shape), optic_axis=torch.tensor(optic_axis).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, volume_shape[0], volume_shape[1], volume_shape[2]))
    
    # Check that the initialization and normalization of optical axes are correct
    assert np.all(np.isclose(voxel_numpy.optic_axis.flatten(), voxel_torch.optic_axis.detach().numpy().flatten())), f"Optic axis mismatch between numpy/pytorch delta_n: {delta_n}, optic_axis: {optic_axis}"
    assert np.all(np.isclose(voxel_numpy.optic_axis.flatten(), voxel_numpy_single_value.optic_axis.flatten())), f"Optic axis mismatch between numpy/pytorch delta_n: {delta_n}, optic_axis: {optic_axis}"                                  
    assert np.all(np.isclose(voxel_numpy_single_value.optic_axis.flatten(), voxel_torch_single_value.optic_axis.detach().numpy().flatten())), f"Optic axis mismatch between numpy/pytorch delta_n: {delta_n}, optic_axis: {optic_axis}"
    assert np.all(np.isclose(voxel_numpy_single_value.Delta_n.flatten(), voxel_torch_single_value.Delta_n.detach().numpy().flatten())), f"Delta_n mismatch between numpy/pytorch delta_n: {delta_n}, optic_axis: {optic_axis}"

@pytest.mark.parametrize('volume_shape_in', [
        3*[1],
        3*[7],
        3*[8],
        3*[11],
        3*[21],
        3*[51],
    ])
def test_compute_JonesMatrices(global_data, volume_shape_in):
    # Define the voxel parameters
    delta_n = 0.1
    optic_axis = [1.0,0.0,0]

    # Gather global data
    local_data = copy.deepcopy(global_data)
    optical_info = local_data['optical_info']
    optical_info['pixels_per_ml'] = 17

    pixels_per_ml = optical_info['pixels_per_ml']
    volume_shape = volume_shape_in
    optical_info['volume_shape'] = volume_shape

    # Create numpy and pytorch raytracer
    BF_raytrace_numpy = BirefringentRaytraceLFM(optical_info=optical_info)
    BF_raytrace_torch = BirefringentRaytraceLFM(backend=BackEnds.PYTORCH, optical_info=optical_info)
    
    # Compute ray-volume geometry and Siddon algorithm
    BF_raytrace_numpy.compute_rays_geometry()
    BF_raytrace_torch.compute_rays_geometry()

    # Create voxel array in numpy
    voxel_numpy = BirefringentVolume(backend=BackEnds.NUMPY, optical_info=optical_info,
                                    Delta_n=delta_n, optic_axis=optic_axis)

    # Create a voxel array in torch                          
    voxel_torch = BirefringentVolume(backend=BackEnds.PYTORCH, optical_info=optical_info,
                                    Delta_n=delta_n, optic_axis=optic_axis)
       

    # Create arrays to store images
    ret_img_numpy = np.zeros([pixels_per_ml,pixels_per_ml])
    azi_img_numpy = np.zeros([pixels_per_ml,pixels_per_ml])
    ret_img_torch = np.zeros([pixels_per_ml,pixels_per_ml])
    azi_img_torch = np.zeros([pixels_per_ml,pixels_per_ml])


    # Compute numpy Jones Matrices, by iterating through all the rays and their interaction with the voxel_numpy
    for ii in range(pixels_per_ml):
        for jj in range(pixels_per_ml):
            # Compute JonesMatrix for this ray and the voxels it collides with
            JM_numpy = BF_raytrace_numpy.calc_cummulative_JM_of_ray_numpy(ii, jj, voxel_numpy)

            # Compute Retardance and Azimuth
            ret_numpy,azi_numpy = BF_raytrace_numpy.retardance(JM_numpy), BF_raytrace_numpy.azimuth(JM_numpy)
            ret_img_numpy[ii,jj] = ret_numpy
            azi_img_numpy[ii,jj] = azi_numpy


    # Compute JM with Pytorch implmentation
    JM_torch = BF_raytrace_torch.calc_cummulative_JM_of_ray_torch(voxel_torch)
    # Compute retardance and azimuth
    ret_torch,azi_torch = BF_raytrace_torch.retardance(JM_torch), BF_raytrace_torch.azimuth(JM_torch)


    # Fill in retardance and azimuth of torch into an image,
    # And compare with their corresponding numpy JM
    any_fail = False
    for ray_ix in range(BF_raytrace_torch.ray_valid_indices.shape[1]):
        i = BF_raytrace_torch.ray_valid_indices[0,ray_ix]
        j = BF_raytrace_torch.ray_valid_indices[1,ray_ix]
        ret_img_torch[i, j] = ret_torch[ray_ix].item()
        azi_img_torch[i, j] = azi_torch[ray_ix].item()
        JM_numpy = BF_raytrace_numpy.calc_cummulative_JM_of_ray_numpy(i, j, voxel_numpy)

        # Important, set the tolerance to 1e-5, as numpy computes in float64 and torch in float32
        assert np.isclose(JM_numpy.astype(np.complex64), JM_torch[ray_ix].detach().numpy(), atol=1e-5).all(), f'JM mismatch on coord: (i,j)= ({i},{j}):'
        # As we are testing for JM, use try_catch on azimuth and retardance, such that they don't brake the test
        # And report if there was mismatch at the end
        try:
            # Check retardance for this ray
            assert np.isclose(ret_img_numpy[i, j], ret_img_torch[i, j], atol=1e-5).all(), f'Retardance mismatch on coord: (i,j)= ({i},{j}):'
        except:
            print(f'Retardance mismatch on coord: (i,j)= ({i},{j}):')
            any_fail = True
        try:
            # Check azimuth for this ray
            check_azimuth_images(np.array([azi_img_numpy[i, j]]), np.array(azi_img_torch[i, j]), message=f'Azimuth mismatch on coord: (i,j)=({i},{j}):')
        except:
            print(f'Azimuth mismatch on coord: (i,j)=({i},{j}):')
            any_fail = True
    
    # Use this in debug console to visualize errors
    # plot_ret_azi_image_comparison(ret_img_numpy, azi_img_numpy, ret_img_torch, azi_img_torch)
    assert any_fail==False, 'No errors in Jones Matrices, but there were mismatches between Retardance and Azimuth in numpy vs torch'

@pytest.mark.parametrize('iteration', range(0, 4))
def test_compute_retardance_and_azimuth_images(global_data, iteration):
    volume_shapes_to_test = [
        3*[1],
        3*[7],
        3*[8],
        3*[11],
        3*[21],
        3*[50],
    ]
    torch.set_grad_enabled(False)
    # Define the voxel parameters
    # delta_n = 0.1
    # optic_axis = [1.0,0.0,0]
    
    delta_n = np.random.uniform(0.01,0.25,1)[0]
    optic_axis = np.random.uniform(0.01,0.25,3)

    # Gather global data
    optical_info = copy.deepcopy(global_data['optical_info'])
    volume_shape = optical_info['volume_shape']

    volume_shape = volume_shapes_to_test[iteration]
    # pixels_per_ml = 17
    
    optical_info['volume_shape'] = volume_shape
    # optical_info['pixels_per_ml'] = pixels_per_ml

    # Create numpy and pytorch raytracer
    BF_raytrace_numpy = BirefringentRaytraceLFM(optical_info=optical_info)
    BF_raytrace_torch = BirefringentRaytraceLFM(backend=BackEnds.PYTORCH, optical_info=optical_info)
    
    # Compute ray-volume geometry and Siddon algorithm
    BF_raytrace_numpy.compute_rays_geometry()
    BF_raytrace_torch.compute_rays_geometry()

    # Create voxel array in numpy
    voxel_numpy = BirefringentVolume(backend=BackEnds.NUMPY,  optical_info=optical_info,
                                    Delta_n=delta_n, optic_axis=optic_axis)

    # Create a voxel array in torch                          
    voxel_torch = BirefringentVolume(backend=BackEnds.PYTORCH, optical_info=optical_info,
                                    Delta_n=delta_n, optic_axis=optic_axis)
    
    # Compute retardance and azimuth images with both methods
    ret_img_numpy, azi_img_numpy = BF_raytrace_numpy.ret_and_azim_images(voxel_numpy)
    ret_img_torch, azi_img_torch = BF_raytrace_torch.ret_and_azim_images(voxel_torch)
    # Use this in debug console to visualize errors
    # plot_ret_azi_image_comparison(ret_img_numpy, azi_img_numpy, ret_img_torch, azi_img_torch)

    assert np.all(np.isnan(ret_img_numpy)==False), "Error in numpy retardance computations nan found"
    assert np.all(np.isnan(azi_img_numpy)==False), "Error in numpy azimuth computations nan found"
    assert torch.all(torch.isnan(ret_img_torch)==False), "Error in torch retardance computations nan found"
    assert torch.all(torch.isnan(azi_img_torch)==False), "Error in torch azimuth computations nan found"

    assert np.all(np.isclose(ret_img_numpy.astype(np.float32), ret_img_torch.numpy(), atol=1e-5)), "Error when comparing retardance computations"
    check_azimuth_images(azi_img_numpy.astype(np.float32), azi_img_torch.numpy())

@pytest.mark.parametrize('volume_shape_in', [
        # 3*[1],
        3*[7],
        3*[8],
        3*[11],
        # 3*[21], # todo, debug this two examples
        # 3*[51],
    ])
def test_forward_projection_lenslet_grid_random_volumes(global_data, volume_shape_in):
    torch.set_grad_enabled(False)
    # Gather global data
    local_data = copy.deepcopy(global_data)
    optical_info = local_data['optical_info']

    # Volume shape
    volume_shape = volume_shape_in
    optical_info['volume_shape'] = volume_shape

    # The n_micro_lenses defines the active volume area, and it should be smaller than the volume_shape.
    # This as some rays go beyond the volume in front of a single micro-lens
    optical_info['n_micro_lenses']  = volume_shape[1] - 4
    optical_info['n_voxels_per_ml'] = 1

    
    # Create Ray-tracing objects
    BF_raytrace_numpy = BirefringentRaytraceLFM(optical_info=optical_info)
    BF_raytrace_torch = BirefringentRaytraceLFM(backend=BackEnds.PYTORCH, optical_info=optical_info)
    
    BF_raytrace_numpy.compute_rays_geometry()
    BF_raytrace_torch.compute_rays_geometry()

    
    # Generate a volume with random everywhere
    # voxel_torch_random = BF_raytrace_torch.init_volume(volume_shape, init_mode='ellipsoid')
    # voxel_numpy_random = BF_raytrace_numpy.init_volume(volume_shape, init_mode='ellipsoid')
    voxel_torch_random = BF_raytrace_torch.init_volume(volume_shape, init_mode='random')
    # voxel_numpy_random = BF_raytrace_numpy.init_volume(volume_shape, init_mode='random')
    # Copy the volume, to have exactly the same things
    voxel_numpy_random = BirefringentVolume(backend=BackEnds.NUMPY,  optical_info=optical_info,
                                    Delta_n=voxel_torch_random.get_delta_n().numpy(), optic_axis=voxel_torch_random.get_optic_axis().numpy())


    
    assert BF_raytrace_numpy.optical_info == voxel_numpy_random.optical_info, 'Mismatch on RayTracer and volume optical_info numpy'
    assert BF_raytrace_torch.optical_info == voxel_torch_random.optical_info, 'Mismatch on RayTracer and volume optical_info torch'
    
    with np.errstate(divide='raise'):
        ret_img_numpy, azi_img_numpy = BF_raytrace_numpy.ray_trace_through_volume(voxel_numpy_random)
    ret_img_torch, azi_img_torch = BF_raytrace_torch.ray_trace_through_volume(voxel_torch_random)
    
    plot_ret_azi_image_comparison(ret_img_numpy, azi_img_numpy, ret_img_torch, azi_img_torch)

    assert np.all(np.isnan(ret_img_numpy)==False), "Error in numpy retardance computations nan found"
    assert np.all(np.isnan(azi_img_numpy)==False), "Error in numpy azimuth computations nan found"
    assert torch.all(torch.isnan(ret_img_torch)==False), "Error in torch retardance computations nan found"
    assert torch.all(torch.isnan(azi_img_torch)==False), "Error in torch azimuth computations nan found"

    assert np.all(np.isclose(ret_img_numpy.astype(np.float32), ret_img_torch.numpy(), atol=1e-5)), "Error when comparing retardance computations"
    
    check_azimuth_images(azi_img_numpy.astype(np.float32), azi_img_torch.numpy())

@pytest.mark.parametrize('volume_init_mode', [
        'random',
        'ellipsoid',
        '1planes',
        '3planes'
    ])
def test_forward_projection_different_volumes(global_data, volume_init_mode):
    torch.set_grad_enabled(False)
    # Gather global data
    local_data = copy.deepcopy(global_data)
    optical_info = local_data['optical_info']

    # Volume shape
    volume_shape = [7,7,7]
    optical_info['volume_shape'] = volume_shape

    # The n_micro_lenses defines the active volume area, and it should be smaller than the volume_shape.
    # This as some rays go beyond the volume in front of a single micro-lens
    optical_info['n_micro_lenses']  = volume_shape[1] - 4
    optical_info['n_voxels_per_ml'] = 1
    optical_info['pixels_per_ml'] = 17

    
    # Create Ray-tracing objects
    BF_raytrace_numpy = BirefringentRaytraceLFM(optical_info=optical_info)
    BF_raytrace_torch = BirefringentRaytraceLFM(backend=BackEnds.PYTORCH, optical_info=optical_info)
    
    BF_raytrace_numpy.compute_rays_geometry()
    BF_raytrace_torch.compute_rays_geometry()

    
    # Generate a volume with random everywhere
    # voxel_torch_random = BF_raytrace_torch.init_volume(volume_shape, init_mode='ellipsoid')
    # voxel_numpy_random = BF_raytrace_numpy.init_volume(volume_shape, init_mode='ellipsoid')
    voxel_torch_random = BF_raytrace_torch.init_volume(volume_shape, init_mode=volume_init_mode)
    # voxel_numpy_random = BF_raytrace_numpy.init_volume(volume_shape, init_mode='random')
    # Copy the volume, to have exactly the same things
    voxel_numpy_random = BirefringentVolume(backend=BackEnds.NUMPY,  optical_info=optical_info,
                                    Delta_n=voxel_torch_random.get_delta_n().numpy(), optic_axis=voxel_torch_random.get_optic_axis().numpy())


    
    assert BF_raytrace_numpy.optical_info == voxel_numpy_random.optical_info, 'Mismatch on RayTracer and volume optical_info numpy'
    assert BF_raytrace_torch.optical_info == voxel_torch_random.optical_info, 'Mismatch on RayTracer and volume optical_info torch'
    
    with np.errstate(divide='raise'):
        ret_img_numpy, azi_img_numpy = BF_raytrace_numpy.ray_trace_through_volume(voxel_numpy_random)
    ret_img_torch, azi_img_torch = BF_raytrace_torch.ray_trace_through_volume(voxel_torch_random)
    ret_img_torch, azi_img_torch = BF_raytrace_torch.ray_trace_through_volume(voxel_torch_random)
    
    plot_ret_azi_image_comparison(ret_img_numpy, azi_img_numpy, ret_img_torch, azi_img_torch)

    assert np.all(np.isnan(ret_img_numpy)==False), "Error in numpy retardance computations nan found"
    assert np.all(np.isnan(azi_img_numpy)==False), "Error in numpy azimuth computations nan found"
    assert torch.all(torch.isnan(ret_img_torch)==False), "Error in torch retardance computations nan found"
    assert torch.all(torch.isnan(azi_img_torch)==False), "Error in torch azimuth computations nan found"

    assert np.all(np.isclose(ret_img_numpy.astype(np.float32), ret_img_torch.numpy(), atol=1e-5)), "Error when comparing retardance computations"
    
    check_azimuth_images(azi_img_numpy.astype(np.float32), azi_img_torch.numpy())



@pytest.mark.parametrize('volume_init_mode', [
        'random',
        'ellipsoid',
        '1planes',
        '3planes'
    ])
def test_torch_auto_differentiation(global_data, volume_init_mode):
    # Enable torch gradients
    torch.set_grad_enabled(True)
    # Gather global data
    local_data = copy.deepcopy(global_data)
    optical_info = local_data['optical_info']

    # Simplify the settings, so it's fast to compute
    volume_shape = [3,3,3]
    optical_info['volume_shape'] = volume_shape
    optical_info['pixels_per_ml'] = 17

    BF_raytrace_torch = BirefringentRaytraceLFM(backend=BackEnds.PYTORCH, optical_info=optical_info)
    BF_raytrace_torch.compute_rays_geometry()

    volume_torch_random = BF_raytrace_torch.init_volume(volume_shape, init_mode=volume_init_mode)
    # make volume trainable, by telling waveblocks which variables to optimize
    volume_torch_random.members_to_learn.append('Delta_n')
    volume_torch_random.members_to_learn.append('optic_axis')

    # Create optimizer, which will take care of applying the gradients to the volume once computed
    optimizer = torch.optim.Adam(volume_torch_random.get_trainable_variables(), lr=1e-1)
    # Set all gradients to zero
    optimizer.zero_grad()


    # Compute a forward projection
    ret_image, azim_image = BF_raytrace_torch.ray_trace_through_volume(volume_torch_random)
    # Calculate a loss, for example minimizing the mean of both images
    L = ret_image.mean() + azim_image.mean()

    # Compute and propagate the gradients
    L.backward()

    # Check if the gradients where properly propagated
    assert volume_torch_random.Delta_n.grad.sum() != 0, 'Gradients were not propagated to the volumes Delta_n correctly'
    assert volume_torch_random.optic_axis.grad.sum() != 0, 'Gradients were not propagated to the volumes optic_axis correctly'

    # Calculate the volume values before updating the values with the gradients
    Delta_n_sum_initial = volume_torch_random.Delta_n.sum()
    optic_axis_sum_initial = volume_torch_random.optic_axis.sum()

    # Update the volume
    optimizer.step()

    # Calculate the volume values before updating the values with the gradients
    Delta_n_sum_final = volume_torch_random.Delta_n.sum()
    optic_axis_sum_final = volume_torch_random.optic_axis.sum()

    # These should be different
    assert Delta_n_sum_initial != Delta_n_sum_final, 'Seems like the volume wasnt updated correctly, nothing changed in the volume after the optimization step'
    assert optic_axis_sum_initial != optic_axis_sum_final, 'Seems like the volume wasnt updated correctly, nothing changed in the volume after the optimization step'



# @pytest.mark.parametrize('volume_init_mode', [
#         'random',
#         'ellipsoid',
#         '1planes',
#         '3planes'
#     ])
def speed_speed(global_data, volume_init_mode):
    # Enable torch gradients
    torch.set_grad_enabled(True)
    # Gather global data
    local_data = copy.deepcopy(global_data)
    optical_info = local_data['optical_info']

    # Simplify the settings, so it's fast to compute
    volume_shape = [3,13,13]
    optical_info['volume_shape'] = volume_shape
    optical_info['pixels_per_ml'] = 17
    optical_info['n_micro_lenses'] = 5

    BF_raytrace_torch = BirefringentRaytraceLFM(backend=BackEnds.PYTORCH, optical_info=optical_info)
    BF_raytrace_torch.compute_rays_geometry()

    volume_torch_random = BF_raytrace_torch.init_volume(volume_shape, init_mode=volume_init_mode)
    # make volume trainable, by telling waveblocks which variables to optimize
    volume_torch_random.members_to_learn.append('Delta_n')
    volume_torch_random.members_to_learn.append('optic_axis')

    # Create optimizer, which will take care of applying the gradients to the volume once computed
    optimizer = torch.optim.Adam(volume_torch_random.get_trainable_variables(), lr=1e-1)
    # Set all gradients to zero
    optimizer.zero_grad()

    ret_image, azim_image = BF_raytrace_torch.ray_trace_through_volume(volume_torch_random)

    import torch.autograd.profiler as profiler
    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        # Compute a forward projection
        ret_image, azim_image = BF_raytrace_torch.ray_trace_through_volume(volume_torch_random)
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
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=10))
    prof.export_chrome_trace('forward_trace.trace')
        # # Calculate the volume values before updating the values with the gradients
        # Delta_n_sum_final = volume_torch_random.Delta_n.sum()
        # optic_axis_sum_final = volume_torch_random.optic_axis.sum()

        # # These should be different
        # assert Delta_n_sum_initial != Delta_n_sum_final, 'Seems like the volume wasnt updated correctly, nothing changed in the volume after the optimization step'
        # assert optic_axis_sum_initial != optic_axis_sum_final, 'Seems like the volume wasnt updated correctly, nothing changed in the volume after the optimization step'



def main():
    speed_speed(global_data(), 'ellipsoid')
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
    
 # Check azimuth images
def check_azimuth_images(img1, img2, message="Error when comparing azimuth computations"):
    ''' Compares two azimuth images, taking into account that atan2 output of 0 and pi is equivalent'''
    if not np.all(np.isclose(img1, img2, atol=1e-5)):
        # Check if the difference is a multiple of pi
        diff = np.abs(img1 - img2)
        assert np.all(np.isclose( diff[~np.isclose(diff,0.0, atol=1e-5)], np.pi, atol=1e-5)), message
    
def plot_azimuth(img):
    ctr = [(img.shape[0] - 1)/ 2, (img.shape[1] - 1)/ 2]
    i = np.linspace(0, img.shape[0] - 1, img.shape[0])
    j = np.linspace(0, img.shape[0] - 1, img.shape[0])
    jv, iv = np.meshgrid(i, j)
    dist_from_ctr = np.sqrt((iv - ctr[0]) ** 2 + (jv - ctr[1]) ** 2)


    fig = plt.figure(figsize=(13, 4))
    fig.subplots_adjust(bottom=0.025, left=0.025, top = 0.975, right=0.975)

    plt.rcParams['image.origin'] = 'lower'
    plt.clf()
    sub1 = plt.subplot(1, 3, 1)
    sub1.imshow(dist_from_ctr)
    plt.title("Distance from center")

    sub2 = plt.subplot(1, 3, 2)
    cax = sub2.imshow(img * 180 / np.pi)
    # plt.colorbar()
    plt.title('Azimuth (degrees)')
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    # cbar.ax.set_yticklabels(['-pi', '-pi/2', '-pi/4', '-pi/8', '0', 'pi/8', 'pi/4', 'pi/2', 'pi'])  # vertically oriented colorbar
    cbar = fig.colorbar(cax, ticks=np.rad2deg([-np.pi, -np.pi/2, -np.pi/3, -np.pi/4, -np.pi/6, 0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi]))
    cbar.ax.set_yticklabels(['-180', '-90', '-60', '-45', '-30', '0', '30', '45', '60', '90', '180'])

    # plt.subplot(1, 3, 3)
    # plt.polar(dist_from_ctr, img / np.pi)
    sub3 = plt.subplot(1, 3, 3)
    sub3.scatter(dist_from_ctr, img * 180 / np.pi)
    # plt.colorbar()
    plt.title('Azimuth in polar coordinates')
    plt.xlabel('Distance from center pixel')
    plt.ylabel('Azimuth')

    plt.pause(0.05)
    plt.show()


def plot_ret_azi_image_comparison(ret_img_numpy, azi_img_numpy, ret_img_torch, azi_img_torch):
    # Check if we are running in pytest
    import os
    if "PYTEST_CURRENT_TEST" in os.environ:
        return
    plt.rcParams['image.origin'] = 'lower'
    plt.clf()
    plt.subplot(3,2,1)
    plt.imshow(ret_img_numpy)
    plt.title('Ret. numpy')
    plt.subplot(3,2,2)
    plt.imshow(azi_img_numpy)
    plt.title('Azi. numpy')

    plt.subplot(3,2,3)
    plt.imshow(ret_img_torch)
    plt.title('Ret. torch')
    plt.subplot(3,2,4)
    plt.imshow(azi_img_torch)
    plt.title('Azi. torch')

    plt.subplot(3,2,5)
    diff = np.abs(ret_img_torch-ret_img_numpy)
    plt.imshow(diff)
    plt.title(f'Ret. Diff: {diff.sum()}')

    plt.subplot(3,2,6)
    diff = np.abs(azi_img_torch-azi_img_numpy)
    plt.imshow(diff)
    plt.title(f'Azi. Diff: {diff.sum()}')
    plt.pause(0.05)
    plt.show(block=True)


if __name__ == '__main__':
    main()