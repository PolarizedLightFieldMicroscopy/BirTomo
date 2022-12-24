import pytest

from VolumeRaytraceLFM.birefringence_implementations import *
import matplotlib.pyplot as plt
import copy
import os

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

def speed_speed(global_data, volume_init_mode):
    # Enable torch gradients
    torch.set_grad_enabled(True)
    # Gather global data
    local_data = copy.deepcopy(global_data)
    optical_info = local_data['optical_info']

    # Simplify the settings, so it's fast to compute
    volume_shape = [3,21,21]
    optical_info['volume_shape'] = volume_shape
    optical_info['pixels_per_ml'] = 17
    optical_info['n_micro_lenses'] = 15

    BF_raytrace_torch = BirefringentRaytraceLFM(backend=BackEnds.PYTORCH, optical_info=optical_info)
    BF_raytrace_torch.compute_rays_geometry()


    BF_raytrace_torch2 = BirefringentRaytraceLFM(backend=BackEnds.PYTORCH, optical_info=optical_info)
    BF_raytrace_torch2.compute_rays_geometry()

    volume_torch_random = BirefringentVolume.create_dummy_volume( backend=BackEnds.PYTORCH, optical_info=optical_info, \
                                                    vol_type='ellipsoid')
    # make volume trainable, by telling waveblocks which variables to optimize
    volume_torch_random.members_to_learn.append('Delta_n')
    volume_torch_random.members_to_learn.append('optic_axis')

    # Create optimizer, which will take care of applying the gradients to the volume once computed
    optimizer = torch.optim.Adam(volume_torch_random.get_trainable_variables(), lr=1e-1)
    # Set all gradients to zero
    optimizer.zero_grad()

    ret_image, azim_image = BF_raytrace_torch.ray_trace_through_volume(volume_torch_random, True,)
    ret_image2, azim_image2 = BF_raytrace_torch2.ray_trace_through_volume(volume_torch_random, False,)



    import time
    import sys
    startTime = time.time()
    ret_image, azim_image = BF_raytrace_torch.ray_trace_through_volume(volume_torch_random, True,)
    executionTime = (time.time() - startTime)
    print('Warmup time in seconds with Torch: ' + str(executionTime))

    startTime = time.time()
    ret_image2, azim_image2 = BF_raytrace_torch2.ray_trace_through_volume(volume_torch_random, False,)
    executionTime = (time.time() - startTime)
    print('Warmup time in seconds with Torch: ' + str(executionTime))

    plt.subplot(1,3,1)
    plt.imshow(ret_image.detach().numpy())
    plt.subplot(1,3,2)
    plt.imshow(ret_image2.detach().numpy())
    plt.subplot(1,3,3)
    plt.imshow((ret_image-ret_image2).detach().numpy())
    plt.show(block=True)
    sys.exit(0)
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