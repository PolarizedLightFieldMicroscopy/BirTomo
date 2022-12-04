import torch
from VolumeRaytraceLFM.birefringence_implementations import *
from waveblocks.utils.misc_utils import *
from ray import *
from jones import *
import time
from tqdm import tqdm

# Set objective info
optic_config = OpticConfig()
optic_config.PSF_config.M = 60      # Objective magnification
optic_config.PSF_config.NA = 1.2    # Objective NA
optic_config.PSF_config.ni = 1.52   # Refractive index of sample (experimental)
optic_config.PSF_config.ni0 = 1.52  # Refractive index of sample (design value)
optic_config.PSF_config.wvl = 0.550
optic_config.mla_config.n_pixels_per_mla = 17
optic_config.camera_config.sensor_pitch = 6.5
optic_config.mla_config.pitch = optic_config.mla_config.n_pixels_per_mla * optic_config.camera_config.sensor_pitch
optic_config.mla_config.n_micro_lenses = 11

optic_config.volume_config.volume_shape = [21, 111, 111]
optic_config.volume_config.voxel_size_um = [1,] + 2*[optic_config.mla_config.pitch / optic_config.PSF_config.M]
optic_config.volume_config.volume_size_um = np.array(optic_config.volume_config.volume_shape) * np.array(optic_config.volume_config.voxel_size_um)

# Create a Birefringent Raytracer
BF_raytrace = BirefringentRaytraceLFM(optic_config=optic_config, members_to_learn=[])

# Compute the rays and use the Siddon's algorithm to compute the intersections with voxels.
# If a filepath is passed as argument, the object with all its calculations get stored/loaded from a file
startTime = time.time()
BF_raytrace = BF_raytrace.compute_rays_geometry() #'test_ray_geometry'
executionTime = (time.time() - startTime)
print('Ray-tracing time in seconds: ' + str(executionTime))

# Create a Birefringent volume, with random 
volume = BF_raytrace.init_volume(init_mode='random')


# Single voxel
if True:
    my_volume = BF_raytrace.init_volume(init_mode='zeros')
    voxel_birefringence = 0.1
    voxel_birefringence_axis = torch.tensor([-0.5,0.5,0])
    voxel_birefringence_axis /= voxel_birefringence_axis.norm()
    offset = 0
    # Disable gradients in volume, as in-place assignment on tensors with gradients is not allowed in Pytorch
    my_volume.voxel_parameters.requires_grad = False
    my_volume.voxel_parameters[:,   BF_raytrace.vox_ctr_idx[0]-2, 
                                    BF_raytrace.vox_ctr_idx[1]+offset, 
                                    BF_raytrace.vox_ctr_idx[2]+offset] = torch.tensor([
                                                                                        voxel_birefringence, 
                                                                                        voxel_birefringence_axis[0], 
                                                                                        voxel_birefringence_axis[1], 
                                                                                        voxel_birefringence_axis[2]])
    # Re-enable gradients                                                                                           
    my_volume.voxel_parameters.requires_grad = True

else: # whole plane
    my_volume = BF_raytrace.init_volume(init_mode='1planes')
    volume = BF_raytrace.generate_ellipsoid_volume(volume_shape=optic_config.volume_config.volume_shape, radius=[5,7,7], delta_n=0.1)
    my_volume.voxel_parameters = volume
# my_volume.plot_volume_plotly(opacity=0.1)



# Perform same calculation with torch
startTime = time.time()
ret_image_measured, azim_image_measured = BF_raytrace.ray_trace_through_volume(my_volume) #BF_raytrace.ret_and_azim_images(my_volume)
executionTime = (time.time() - startTime)
print('Execution time in seconds with Torch: ' + str(executionTime))

# Store GT images
volume_GT = my_volume.voxel_parameters[0].detach().clone()
ret_image_measured = ret_image_measured.detach()
azim_image_measured = azim_image_measured.detach()


############# Torch
# Let's create an optimizer
# Initial guess
my_volume = BF_raytrace.init_volume(init_mode='random')
optimizer = torch.optim.Adam([my_volume.voxel_parameters], lr=0.0001)
n_epochs = 5000
sparse_lambda = 0.1
loss_function = torch.nn.L1Loss()

# To test differentiability let's define a loss function L = |ret_image_torch|, and minimize it
losses = []
plt.ion()
figure = plt.figure(figsize=(15,15))

co_gt, ca_gt = ret_image_measured*torch.cos(azim_image_measured), ret_image_measured*torch.sin(azim_image_measured)
for ep in tqdm(range(n_epochs), "Minimizing"):
    optimizer.zero_grad()
    ret_image_current, azim_image_current = BF_raytrace.ray_trace_through_volume(my_volume)
    # Vector difference
    # co_pred, ca_pred = ret_image_current*torch.cos(azim_image_current), ret_image_current*torch.sin(azim_image_current)
    # L = ((co_gt-co_pred)**2 + (ca_gt-ca_pred)**2).sqrt().mean()
    L = loss_function(ret_image_measured, ret_image_current) + \
        0.1*(torch.cos(azim_image_measured) - torch.cos(azim_image_current)).abs().mean() #+ 0.1*(torch.sin(azim_image_measured) - torch.sin(azim_image_current)).abs().mean()
    #     (torch.cos(azim_image_measured-azim_image_current)**2 + torch.sin(azim_image_measured-azim_image_current)**2).abs().mean()
        # cos + sine
        # 0.1*(torch.cos(azim_image_measured) - torch.cos(azim_image_current)).abs().mean() + 0.1*(torch.sin(azim_image_measured) - torch.sin(azim_image_current)).abs().mean()
        # (torch.atan2(torch.sin(azim_image_measured - azim_image_current), torch.cos(azim_image_measured - azim_image_current))).abs().mean()
        # (2 * (1 - torch.cos(azim_image_measured - azim_image_current))).mean() 
        # loss_function(azim_image_measured, azim_image_current)
    
    # Calculate update of the my_volume (Compute gradients of the L with respect to my_volume)
    L.backward()
    # Apply gradient updates to the volume
    optimizer.step()
    # print(f'Ep:{ep} loss: {L.item()}')
    losses.append(L.item())


    if ep%10==0:
        plt.subplot(3,3,1)
        plt.imshow(ret_image_measured.detach().numpy())
        plt.colorbar()
        plt.title('Initial Retardance')
        plt.subplot(3,3,2)
        plt.imshow(azim_image_measured.detach().numpy())
        plt.colorbar()
        plt.title('Initial Azimuth')
        plt.subplot(3,3,3)
        plt.imshow(volume_2_projections(volume_GT.unsqueeze(0))[0,0].detach().numpy())
        plt.colorbar()
        plt.title('Initial volume MIP')

        plt.subplot(3,3,4)
        plt.imshow(ret_image_current.detach().numpy())
        plt.colorbar()
        plt.title('Final Retardance')
        plt.subplot(3,3,5)
        plt.imshow(azim_image_current.detach().numpy())
        plt.colorbar()
        plt.title('Final Azimuth')
        plt.subplot(3,3,6)
        plt.imshow(volume_2_projections(my_volume.voxel_parameters[0].unsqueeze(0))[0,0].detach().numpy())
        plt.colorbar()
        plt.title('Final Volume MIP')
        plt.subplot(3,1,3)
        plt.plot(list(range(len(losses))),losses)

        figure.canvas.draw()
        figure.canvas.flush_events()
        time.sleep(0.1)


# Display
plt.savefig("Optimization_ellipse_cosine_diff.pdf")
plt.show()
