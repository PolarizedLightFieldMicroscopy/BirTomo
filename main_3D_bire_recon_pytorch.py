import torch
from VolumeRaytraceLFM.birefringence_implementations import *
from waveblocks.utils.misc_utils import *
import time
from tqdm import tqdm

camPixPitch = 6.5
magnObj = 60
voxel_size_um = 3*[camPixPitch / magnObj]
n_micro_lenses = 5

optical_info={
            'volume_shape' : 3*[11], 
            'voxel_size_um' : voxel_size_um, 
            'pixels_per_ml' : 17, 
            'na_obj' : 1.2, 
            'n_medium' : 1.52,
            'wavelength' : 0.55,
            'n_micro_lenses' : n_micro_lenses,
            'n_voxels_per_ml' : 1}

back_end = BackEnds.PYTORCH

# Create a Birefringent Raytracer
BF_raytrace = BirefringentRaytraceLFM(back_end=back_end, optical_info=optical_info)

# Compute the rays and use the Siddon's algorithm to compute the intersections with voxels.
# If a filepath is passed as argument, the object with all its calculations get stored/loaded from a file
startTime = time.time()
BF_raytrace.compute_rays_geometry() 
executionTime = (time.time() - startTime)
print('Ray-tracing time in seconds: ' + str(executionTime))

# Move ray tracer to GPU
device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
print(f'Using computing device: {device}')
BF_raytrace = BF_raytrace.to(device)


# Single voxel
if False:
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
    my_volume = BF_raytrace.init_volume(volume_shape=optical_info['volume_shape'], init_mode='ellipsoid')

# my_volume.plot_volume_plotly(opacity=0.1)



with torch.no_grad():
    # Perform same calculation with torch
    startTime = time.time()
    ret_image_measured, azim_image_measured = BF_raytrace.ray_trace_through_volume(my_volume) 
    executionTime = (time.time() - startTime)
    print('Execution time in seconds with Torch: ' + str(executionTime))

    # Store GT images
    Delta_n_GT = my_volume.Delta_n.detach().clone()
    optic_axis_GT = my_volume.optic_axis.detach().clone()
    ret_image_measured = ret_image_measured.detach()
    azim_image_measured = azim_image_measured.detach()


############# Torch
# Let's create an optimizer
# Initial guess
my_volume = BF_raytrace.init_volume(volume_shape=optical_info['volume_shape'], init_mode='random')
my_volume.members_to_learn.append('Delta_n')
my_volume.members_to_learn.append('optic_axis')

optimizer = torch.optim.Adam(my_volume.get_trainable_variables(), lr=0.01)
n_epochs = 500
sparse_lambda = 0.1
loss_function = torch.nn.L1Loss()

# To test differentiability let's define a loss function L = |ret_image_torch|, and minimize it
losses = []
plt.ion()
figure = plt.figure(figsize=(10,5))

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
        plt.subplot(2,4,1)
        plt.imshow(ret_image_measured.detach().cpu().numpy())
        plt.colorbar()
        plt.title('Initial Retardance')
        plt.subplot(2,4,2)
        plt.imshow(azim_image_measured.detach().cpu().numpy())
        plt.colorbar()
        plt.title('Initial Azimuth')
        plt.subplot(2,4,3)
        plt.imshow(volume_2_projections(Delta_n_GT.unsqueeze(0))[0,0].detach().cpu().numpy())
        plt.colorbar()
        plt.title('Initial volume MIP')

        plt.subplot(2,4,5)
        plt.imshow(ret_image_current.detach().cpu().numpy())
        plt.colorbar()
        plt.title('Final Retardance')
        plt.subplot(2,4,6)
        plt.imshow(azim_image_current.detach().cpu().numpy())
        plt.colorbar()
        plt.title('Final Azimuth')
        plt.subplot(2,4,7)
        plt.imshow(volume_2_projections(my_volume.Delta_n.unsqueeze(0))[0,0].detach().cpu().numpy())
        plt.colorbar()
        plt.title('Final Volume MIP')
        plt.subplot(2,4,8)
        plt.plot(list(range(len(losses))),losses)

        figure.canvas.draw()
        figure.canvas.flush_events()
        time.sleep(0.1)


# Display
plt.savefig("Optimization_ellipse_cosine_diff_new_implementation.pdf")
plt.show()
