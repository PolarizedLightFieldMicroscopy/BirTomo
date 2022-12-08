import pytest
from VolumeRaytraceLFM.birefringence_implementations import *
import matplotlib.pyplot as plt

@pytest.fixture(scope = 'module')
def global_data():
    '''Create global optic_setting and optical_info containing all the optics and volume information
        The tests can access this by passing the name of this function as an argument for example:
        def test_something(global_data):
            optical_info = global_data['optical_info]
    '''
    # Objective configuration
    magnObj = 60
    wavelength = 0.550
    naObj = 1.2
    nMedium = 1.52
    # Camera and volume configuration
    camPixPitch = 6.5
    # MLA configuration
    pixels_per_ml = 5 # num pixels behind lenslet
    microLensPitch = pixels_per_ml * camPixPitch / magnObj
    # voxPitch is the width of each voxel in um (dividing by 5 to supersample)
    voxPitch = microLensPitch / 1
    axialPitch = voxPitch
    voxel_size_um = [axialPitch, voxPitch, voxPitch]
    # Volume shape
    volume_shape = [11, 11, 11]

    optic_config = OpticConfig()
    # Set objective info
    optic_config.PSF_config.M = magnObj      # Objective magnification
    optic_config.PSF_config.NA = naObj    # Objective NA
    optic_config.PSF_config.ni = nMedium   # Refractive index of sample (experimental)
    optic_config.PSF_config.ni0 = nMedium  # Refractive index of sample (design value)
    optic_config.PSF_config.wvl = wavelength
    optic_config.mla_config.n_pixels_per_mla = pixels_per_ml
    optic_config.mla_config.n_micro_lenses = volume_shape[1]
    optic_config.camera_config.sensor_pitch = camPixPitch
    optic_config.mla_config.pitch = optic_config.mla_config.n_pixels_per_mla * optic_config.camera_config.sensor_pitch

    optic_config.volume_config.volume_shape = volume_shape
    optic_config.volume_config.voxel_size_um = voxel_size_um
    optic_config.volume_config.volume_size_um = np.array(optic_config.volume_config.volume_shape) * np.array(optic_config.volume_config.voxel_size_um)


    # Prepate settings for numpy
    optical_info={
                'volume_shape' : volume_shape, 
                'voxel_size_um' : voxel_size_um, 
                'pixels_per_ml' : pixels_per_ml, 
                'na_obj' : naObj, 
                'n_medium' : nMedium,
                'wavelength' : wavelength}


    return {'optic_config': optic_config, 'optical_info' : optical_info}

# todo: run with different volume sizes
# @pytest.mark.parametrize('iteration', range(1, 10))
def test_rays_computation(global_data):
    
    # Gather global data
    optical_info = global_data['optical_info']
    optic_config = global_data['optic_config']

    # Create numpy and pytorch raytracer
    BF_raytrace_numpy = BirefringentRaytraceLFM(optical_info=optical_info)
    BF_raytrace_torch = BirefringentRaytraceLFM(back_end=BackEnds.PYTORCH, torch_args={'optic_config':optic_config})
    
    # Compute ray-volume geometry and Siddon algorithm
    BF_raytrace_numpy.compute_rays_geometry()
    BF_raytrace_torch.compute_rays_geometry()

    # Compare results
    assert np.all(np.isclose(BF_raytrace_numpy.ray_entry,BF_raytrace_torch.ray_entry.numpy())),         "ray_entry calculation mismatch between Numpy and Pytorch back-end"
    assert np.all(np.isclose(BF_raytrace_numpy.ray_exit,BF_raytrace_torch.ray_exit.numpy())),           "ray_exit calculation mismatch between Numpy and Pytorch back-end"
    assert np.all(np.isclose(BF_raytrace_numpy.ray_direction,BF_raytrace_torch.ray_direction.numpy())), "ray_direction calculation mismatch between Numpy and Pytorch back-end"
    
    for n_basis in range(3):
        for n_ray in range(len(BF_raytrace_numpy.ray_direction_basis)):
            assert(np.all(np.isclose(BF_raytrace_numpy.ray_direction_basis[n_ray][n_basis], BF_raytrace_torch.ray_direction_basis[n_basis][n_ray]))), f"ray_direction_basis mismatch for ray: {n_ray}, basis: {n_basis}"

@pytest.mark.parametrize('iteration', range(1, 10))
def test_voxel_array_creation(global_data, iteration):
    
    delta_n = np.random.rand()# 0.1
    optic_axis = np.random.rand(3) #[1.0,3.0,1.0]

    # Gather global data
    optical_info = global_data['optical_info']
    optic_config = global_data['optic_config']
    volume_shape = optical_info['volume_shape']
    
    # Create voxels in different ways
    # Passing a single value for delta n and optic axis
    voxel_numpy_single_value = BirefringentVolume(back_end=BackEnds.NUMPY, 
                                    Delta_n=delta_n, optic_axis=optic_axis)

    voxel_torch_single_value = BirefringentVolume(back_end=BackEnds.PYTORCH, torch_args={'optic_config' : optic_config},
                                    Delta_n=delta_n, optic_axis=optic_axis)
                                    
    # Passing an already build 3D array                            
    voxel_torch = BirefringentVolume(back_end=BackEnds.PYTORCH, torch_args={'optic_config':optic_config},
                                    Delta_n=delta_n*torch.ones(volume_shape), 
                                    optic_axis=torch.tensor(optic_axis).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, volume_shape[0], volume_shape[1], volume_shape[2])
                                    )
                                    
    # Passing an already build 3D array                            
    voxel_numpy = BirefringentVolume(back_end=BackEnds.NUMPY,
                                    Delta_n=delta_n*torch.ones(volume_shape).numpy(), 
                                    optic_axis=torch.tensor(optic_axis).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, volume_shape[0], volume_shape[1], volume_shape[2]).numpy()
                                    )
    voxel_torch = BirefringentVolume(back_end=BackEnds.PYTORCH, torch_args={'optic_config':optic_config},
                                    Delta_n=delta_n*torch.ones(volume_shape), optic_axis=torch.tensor(optic_axis).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, volume_shape[0], volume_shape[1], volume_shape[2]))
    
    # Check that the initialization and normalization of optical axes are correct
    assert np.all(np.isclose(voxel_numpy.optic_axis.flatten(), voxel_torch.optic_axis.detach().numpy().flatten()))
    assert np.all(np.isclose(voxel_numpy.optic_axis.flatten(), voxel_numpy_single_value.optic_axis.flatten()))
    assert np.all(np.isclose(voxel_numpy_single_value.optic_axis.flatten(), voxel_torch_single_value.optic_axis.detach().numpy().flatten()))
    assert np.all(np.isclose(voxel_numpy_single_value.Delta_n.flatten(), voxel_torch_single_value.Delta_n.detach().numpy().flatten()))

# todo: failing with pixels_per_ml = 5
@pytest.mark.parametrize('volume_shape_in', [
        3*[1],
        3*[7],
        3*[8],
        3*[11],
        # 3*[21],
        # 3*[50],
    ])
def test_compute_JonesMatrices(global_data, volume_shape_in):
    # Define the voxel parameters
    delta_n = 0.1
    optic_axis = [1.0,0.0,0]

    # Gather global data
    optical_info = global_data['optical_info']
    optic_config = global_data['optic_config']
    volume_shape = optical_info['volume_shape']
    pixels_per_ml = optical_info['pixels_per_ml']

    volume_shape = volume_shape_in#[1, 6, 6]
    # pixels_per_ml = 17
    optic_config.volume_config.volume_shape = volume_shape
    optic_config.mla_config.n_micro_lenses = volume_shape[1]
    
    optical_info['volume_shape'] = volume_shape
    # optical_info['pixels_per_ml'] = pixels_per_ml

    # Create numpy and pytorch raytracer
    BF_raytrace_numpy = BirefringentRaytraceLFM(optical_info=optical_info)
    BF_raytrace_torch = BirefringentRaytraceLFM(back_end=BackEnds.PYTORCH, torch_args={'optic_config':optic_config})
    
    # Compute ray-volume geometry and Siddon algorithm
    BF_raytrace_numpy.compute_rays_geometry()
    BF_raytrace_torch.compute_rays_geometry()

    # Create voxel array in numpy
    voxel_numpy = BirefringentVolume(back_end=BackEnds.NUMPY, 
                                    Delta_n=delta_n, optic_axis=optic_axis)

    # Create a voxel array in torch                          
    voxel_torch = BirefringentVolume(back_end=BackEnds.PYTORCH, torch_args={'optic_config':optic_config},
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
    for ray_ix, (i,j) in enumerate(BF_raytrace_torch.ray_valid_indices):
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
            assert np.isclose(azi_img_numpy[i, j], azi_img_torch[i, j], atol=1e-5).all(), f'Azimuth mismatch on coord: (i,j)=({i},{j}):'
        except:
            f'Azimuth mismatch on coord: (i,j)=({i},{j}):'
            any_fail = True
    
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
    # Define the voxel parameters
    delta_n = 0.1
    optic_axis = [1.0,0.0,0]

    # Gather global data
    optical_info = global_data['optical_info']
    optic_config = global_data['optic_config']
    volume_shape = optical_info['volume_shape']
    pixels_per_ml = optical_info['pixels_per_ml']

    volume_shape = volume_shapes_to_test[iteration]#[1, 6, 6]
    # pixels_per_ml = 17
    optic_config.volume_config.volume_shape = volume_shape
    optic_config.mla_config.n_micro_lenses = volume_shape[1]
    
    optical_info['volume_shape'] = volume_shape
    # optical_info['pixels_per_ml'] = pixels_per_ml

    # Create numpy and pytorch raytracer
    BF_raytrace_numpy = BirefringentRaytraceLFM(optical_info=optical_info)
    BF_raytrace_torch = BirefringentRaytraceLFM(back_end=BackEnds.PYTORCH, torch_args={'optic_config':optic_config})
    
    # Compute ray-volume geometry and Siddon algorithm
    BF_raytrace_numpy.compute_rays_geometry()
    BF_raytrace_torch.compute_rays_geometry()

    # Create voxel array in numpy
    voxel_numpy = BirefringentVolume(back_end=BackEnds.NUMPY, 
                                    Delta_n=delta_n, optic_axis=optic_axis)

    # Create a voxel array in torch                          
    voxel_torch = BirefringentVolume(back_end=BackEnds.PYTORCH, torch_args={'optic_config':optic_config},
                                    Delta_n=delta_n, optic_axis=optic_axis)
    
    # Compute retardance and azimuth images with both methods
    ret_img_numpy, azi_img_numpy = BF_raytrace_numpy.ret_and_azim_images(voxel_numpy)
    with torch.no_grad():
        ret_img_torch, azi_img_torch = BF_raytrace_torch.ret_and_azim_images(voxel_torch)
    # Use this in debug console to visualize errors
    # plot_ret_azi_image_comparison(ret_img_numpy, azi_img_numpy, ret_img_torch, azi_img_torch)

    assert np.all(np.isnan(ret_img_numpy)==False), "Error in numpy retardance computations nan found"
    assert np.all(np.isnan(azi_img_numpy)==False), "Error in numpy azimuth computations nan found"
    assert torch.all(torch.isnan(ret_img_torch)==False), "Error in torch retardance computations nan found"
    assert torch.all(torch.isnan(azi_img_torch)==False), "Error in torch azimuth computations nan found"

    assert np.all(np.isclose(ret_img_numpy.astype(np.float32), ret_img_torch.numpy(), atol=1e-5)), "Error when comparing retardance computations"
    assert np.all(np.isclose(azi_img_numpy.astype(np.float32), azi_img_torch.numpy(), atol=1e-5)), "Error when comparing azimuth computations"


def main():
    # test_compute_retardance_and_azimuth_images(global_data(),0)
    # test_voxel_array_creation(global_data(),1)
    # torch.set_default_tensor_type(torch.DoubleTensor)
    # Objective configuration
    magnObj = 60
    wavelength = 0.550
    naObj = 1.2
    nMedium = 1.52
    # Camera and volume configuration
    camPixPitch = 6.5
    # MLA configuration
    pixels_per_ml = 17 # num pixels behind lenslet
    microLensPitch = pixels_per_ml * camPixPitch / magnObj
    # voxPitch is the width of each voxel in um (dividing by 5 to supersample)
    voxPitch = microLensPitch / 1
    axialPitch = voxPitch
    voxel_size_um = [axialPitch, voxPitch, voxPitch]
    # Volume shape
    volume_shape = [1, 11, 11] # [11, 11, 11]


    optic_config = OpticConfig()
    # Set objective info
    optic_config.PSF_config.M = magnObj      # Objective magnification
    optic_config.PSF_config.NA = naObj    # Objective NA
    optic_config.PSF_config.ni = nMedium   # Refractive index of sample (experimental)
    optic_config.PSF_config.ni0 = nMedium  # Refractive index of sample (design value)
    optic_config.PSF_config.wvl = wavelength
    optic_config.mla_config.n_pixels_per_mla = pixels_per_ml
    optic_config.mla_config.n_micro_lenses = volume_shape[1]
    optic_config.camera_config.sensor_pitch = camPixPitch
    optic_config.mla_config.pitch = optic_config.mla_config.n_pixels_per_mla * optic_config.camera_config.sensor_pitch

    optic_config.volume_config.volume_shape = volume_shape
    optic_config.volume_config.voxel_size_um = voxel_size_um
    optic_config.volume_config.volume_size_um = np.array(optic_config.volume_config.volume_shape) * np.array(optic_config.volume_config.voxel_size_um)


    # Prepate settings for numpy
    optical_info={'volume_shape' : volume_shape, 'voxel_size_um' : voxel_size_um, 'pixels_per_ml' : pixels_per_ml, 'na_obj' : naObj, 'n_medium' : nMedium, 'wavelength' : wavelength}

        
    BF_raytrace_numpy = BirefringentRaytraceLFM(optical_info=optical_info)
    BF_raytrace_torch = BirefringentRaytraceLFM(back_end=BackEnds.PYTORCH, torch_args={'optic_config':optic_config})
    
    BF_raytrace_numpy.compute_rays_geometry()
    BF_raytrace_torch.compute_rays_geometry()

    # assert np.all(np.isclose(BF_raytrace_numpy.ray_entry,BF_raytrace_torch.ray_entry.numpy()))
    # assert np.all(np.isclose(BF_raytrace_numpy.ray_exit,BF_raytrace_torch.ray_exit.numpy()))
    # assert np.all(np.isclose(BF_raytrace_numpy.ray_direction,BF_raytrace_torch.ray_direction.numpy()))
    # for n_basis in range(3):
    #     for n_ray in range(len(BF_raytrace_numpy.ray_direction_basis)):
    #         assert(np.all(np.isclose(BF_raytrace_numpy.ray_direction_basis[n_ray][n_basis], BF_raytrace_torch.ray_direction_basis[n_basis][n_ray])))
    
    delta_n = 0.1
    optic_axis = [1.0,0,0]
    
    # Create voxels in different ways

    # Passing a single value for delta n and optic axis
    voxel_numpy = BirefringentVolume(back_end=BackEnds.NUMPY, 
                                    Delta_n=delta_n, optic_axis=optic_axis, optical_info=optical_info)

    # Passing an already build 3D array                            
    voxel_torch = BirefringentVolume(back_end=BackEnds.PYTORCH, torch_args={'optic_config':optic_config}, 
                                    Delta_n=delta_n*torch.ones(volume_shape), optic_axis=torch.tensor(optic_axis).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, volume_shape[0], volume_shape[1], volume_shape[2]))
       

    ret_img_numpy, azi_img_numpy = BF_raytrace_numpy.ret_and_azim_images_numpy(voxel_numpy)
    with torch.no_grad():
        ret_img_torch, azi_img_torch = BF_raytrace_torch.ret_and_azim_images_torch(voxel_torch)
    
    # plot_azimuth(azi_img_numpy)
    plot_ret_azi_image_comparison(ret_img_numpy, azi_img_numpy, ret_img_torch, azi_img_torch)

    
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
    plt.rcParams['image.origin'] = 'lower'
    plt.clf()
    plt.subplot(3,2,1)
    plt.imshow(ret_img_numpy)
    plt.title('Ret. numpy')
    plt.subplot(3,2,2)
    plt.imshow(azi_img_numpy / np.pi)
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
    plt.show()


if __name__ == '__main__':
    main()