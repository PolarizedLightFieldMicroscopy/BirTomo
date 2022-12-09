import torch
from VolumeRaytraceLFM.birefringence_implementations import *
from waveblocks.utils.misc_utils import *
import time
from tqdm import tqdm
import os


# Select back end
back_end = BackEnds.PYTORCH
torch.set_default_tensor_type(torch.DoubleTensor)

camera_pix_pitch = 6.5
objective_M = 60
optical_info={
            'volume_shape' : [11,51,51], 
            'voxel_size_um' : 3*[camera_pix_pitch / objective_M], 
            'pixels_per_ml' : 17, 
            'na_obj' : 1.2, 
            'n_medium' : 1.52,
            'wavelength' : 0.55,
            'n_micro_lenses' : 15,
            'n_voxels_per_ml' : 1}


training_params = {
    'n_epochs' : 5000,
    'azimuth_weight' : 0.1,
    'lr' : 1e-2,
    'output_posfix' : '11ml_atan2loss'
}



# Volume type
# number is the shift from the end of the volume, change it as you wish, do single_voxel{volume_shape[0]//2} for a voxel in the center
# for shift in range(-5,6):
shift_from_center = 0
volume_axial_offset = optical_info['volume_shape'][0]//2+shift_from_center #for center
# volume_type = 'ellipsoid'
volume_type = 'shell'
# volume_type = 'single_voxel' 

# Plot azimuth
# azimuth_plot_type = 'lines'
azimuth_plot_type = 'hsv'

# Create output dir
output_dir = f'reconstructions/recons_{volume_type}_{optical_info["volume_shape"][0]}x{optical_info["volume_shape"][1]}x{optical_info["volume_shape"][2]}__{training_params["output_posfix"]}'
os.makedirs(output_dir, exist_ok=True)
torch.save({'optical_info' : optical_info,
            'training_params' : training_params,
            'volume_type' : volume_type}, f'{output_dir}/parameters.pt')

if volume_type == 'single_voxel':
    optical_info['n_micro_lenses'] = 1
    azimuth_plot_type = 'lines'



# Create a Birefringent Raytracer
BF_raytrace = BirefringentRaytraceLFM(back_end=back_end, optical_info=optical_info)

# Compute the rays and use the Siddon's algorithm to compute the intersections with voxels.
# If a filepath is passed as argument, the object with all its calculations get stored/loaded from a file
startTime = time.time()
BF_raytrace.compute_rays_geometry() 
executionTime = (time.time() - startTime)
print('Ray-tracing time in seconds: ' + str(executionTime))

# Move ray tracer to GPU
if back_end == BackEnds.PYTORCH:
    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    device = "cpu"
    print(f'Using computing device: {device}')
    BF_raytrace = BF_raytrace.to(device)


# Single voxel
if volume_type == 'single_voxel':
    voxel_delta_n = 0.1
    voxel_birefringence_axis = torch.tensor([1,0.0,0])
    voxel_birefringence_axis /= voxel_birefringence_axis.norm()

    # Create empty volume
    my_volume = BF_raytrace.init_volume(optical_info['volume_shape'], init_mode='zeros')
    # Set delta_n
    my_volume.Delta_n.requires_grad = False
    my_volume.optic_axis.requires_grad = False
    my_volume.Delta_n[volume_axial_offset, 
                                    BF_raytrace.vox_ctr_idx[1], 
                                    BF_raytrace.vox_ctr_idx[2]] = voxel_delta_n
    # set optical_axis
    my_volume.optic_axis[:, volume_axial_offset, 
                            BF_raytrace.vox_ctr_idx[1], 
                            BF_raytrace.vox_ctr_idx[2]] = torch.tensor([voxel_birefringence_axis[0], 
                                                                            voxel_birefringence_axis[1], 
                                                                            voxel_birefringence_axis[2]]) if back_end == BackEnds.PYTORCH else voxel_birefringence_axis

    my_volume.Delta_n.requires_grad = True
    my_volume.optic_axis.requires_grad = True

elif volume_type == 'shell' or volume_type == 'ellipsoid': # whole plane
    ellipsoid_args = {  'radius' : [3.5, 4.5, 3.5],
                        'center' : [volume_axial_offset/optical_info['volume_shape'][0], 0.48, 0.51],   # from 0 to 1
                        'delta_n' : -0.1,
                        'border_thickness' : 0.3}

    my_volume = BF_raytrace.init_volume(volume_shape=optical_info['volume_shape'], init_mode='ellipsoid', init_args=ellipsoid_args)

    my_volume.Delta_n.requires_grad = False
    my_volume.optic_axis.requires_grad = False

    # Do we want a shell? let's remove some of the volume
    if volume_type == 'shell':
        my_volume.Delta_n[:optical_info['volume_shape'][0]//2+1,...] = 0
        
    my_volume.Delta_n.requires_grad = True
    my_volume.optic_axis.requires_grad = True

with torch.no_grad():
    my_volume.plot_volume_plotly(optical_info, voxels_in=my_volume.Delta_n, opacity=0.1)





# Create a Birefringent Raytracer
BF_raytrace = BirefringentRaytraceLFM(back_end=back_end, optical_info=optical_info)

# Compute the rays and use the Siddon's algorithm to compute the intersections with voxels.
# If a filepath is passed as argument, the object with all its calculations get stored/loaded from a file
startTime = time.time()
BF_raytrace.compute_rays_geometry() 
executionTime = (time.time() - startTime)
print('Ray-tracing time in seconds: ' + str(executionTime))

print(f'Using computing device: {device}')
BF_raytrace = BF_raytrace.to(device)


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
    azim_comp_measured = torch.arctan2(torch.sin(azim_image_measured), torch.cos(azim_image_measured)).detach()


############# Torch
# Let's create an optimizer
# Initial guess
my_volume = BF_raytrace.init_volume(volume_shape=optical_info['volume_shape'], init_mode='random')
my_volume.members_to_learn.append('Delta_n')
my_volume.members_to_learn.append('optic_axis')

optimizer = torch.optim.Adam(my_volume.get_trainable_variables(), lr=training_params['lr'])
loss_function = torch.nn.L1Loss()

# To test differentiability let's define a loss function L = |ret_image_torch|, and minimize it
losses = []
plt.ion()
figure = plt.figure(figsize=(18,6))
plt.rcParams['image.origin'] = 'lower'

co_gt, ca_gt = ret_image_measured*torch.cos(azim_image_measured), ret_image_measured*torch.sin(azim_image_measured)
for ep in tqdm(range(training_params['n_epochs']), "Minimizing"):
    optimizer.zero_grad()
    ret_image_current, azim_image_current = BF_raytrace.ray_trace_through_volume(my_volume)
    # Vector difference
    # co_pred, ca_pred = ret_image_current*torch.cos(azim_image_current), ret_image_current*torch.sin(azim_image_current)
    # L = ((co_gt-co_pred)**2 + (ca_gt-ca_pred)**2).sqrt().mean()
    azim_diff = azim_comp_measured - torch.arctan2(torch.sin(azim_image_current), torch.cos(azim_image_current))
    L = loss_function(ret_image_measured, ret_image_current) + \
        training_params['azimuth_weight']*(azim_diff).abs().mean() #+ 0.1*(torch.sin(azim_image_measured) - torch.sin(azim_image_current)).abs().mean()
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
        plt.clf()
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
        plt.imshow(np.rad2deg(azim_image_current.detach().cpu().numpy()))
        plt.colorbar()
        plt.title('Final Azimuth')
        plt.subplot(2,4,7)
        plt.imshow(volume_2_projections(my_volume.Delta_n.unsqueeze(0))[0,0].detach().cpu().numpy())
        plt.colorbar()
        plt.title('Final Volume MIP')
        plt.subplot(2,4,8)
        plt.plot(list(range(len(losses))),losses)
        
        plt.gca().yaxis.set_label_position("right")
        plt.gca().yaxis.tick_right()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        figure.canvas.draw()
        figure.canvas.flush_events()
        time.sleep(0.1)
        plt.savefig(f"{output_dir}/Optimization_ep_{'{:02d}'.format(ep)}.pdf")
        time.sleep(0.1)


# Display
plt.savefig(f"{output_dir}/Optimization_final.pdf")
plt.show()
