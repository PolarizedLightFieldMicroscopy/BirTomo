import streamlit as st

st.set_page_config(
    page_title="Reconstructions",
    page_icon="👋",
    layout="wide",
)

st.title("Reconstructions")
st.write("Let's try to reconstruct a volume based on our images!")

import time
import os
import torch
from tqdm import tqdm
from waveblocks.utils.misc_utils import *
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume, BirefringentRaytraceLFM

st.header("Choose our parameters")

columns = st.columns(2)
#first Column
with columns[0]:
############ Optical Params #################
    # Get optical parameters template
    optical_info = BirefringentVolume.get_optical_info_template()
    # Alter some of the optical parameters
    st.subheader('Optical')
    optical_info['n_micro_lenses'] = st.slider('Number of microlenses', min_value=1, max_value=25, value=5)
    optical_info['pixels_per_ml'] = st.slider('Pixels per microlens', min_value=1, max_value=33, value=17, step=2)
    optical_info['n_voxels_per_ml'] = st.slider('Number of voxels per microlens', min_value=1, max_value=3, value=1)


############ Other #################
    st.subheader('Other')
    backend_choice = st.radio('Backend', ['torch'])
    st.write("Backend needs to be torch for the reconstructions")
    backend = BackEnds.PYTORCH

# Second Column
with columns[1]:
############ Volume #################
    st.subheader('Volume')
    volume_container = st.container() # set up a home for other volume selections to go
    optical_info['volume_shape'][0] = st.slider('Axial volume dimension', min_value=1, max_value=50, value=15)
    # y will follow x if x is changed. x will not follow y if y is changed
    optical_info['volume_shape'][1] = st.slider('X volume dimension', min_value=1, max_value=100, value=51)
    optical_info['volume_shape'][2] = st.slider('Y volume dimension', min_value=1, max_value=100, value=optical_info['volume_shape'][1])
    shift_from_center = st.slider('Axial shift from center [voxels]', \
                                    min_value = -int(optical_info['volume_shape'][0]/2), \
                                    max_value = int(optical_info['volume_shape'][0]/2),value = 0)
    volume_axial_offset = optical_info['volume_shape'][0] // 2 + shift_from_center # for center
############ To be continued... #################

############ Volume continued... #################    

    with volume_container: # now that we know backend and shift, we can fill in the rest of the volume params
        how_get_vol = st.radio("Volume can be created or uploaded as an h5 file", \
                                ['h5 upload', 'Create a new volume'], index=1)
        if how_get_vol == 'h5 upload':
            h5file = st.file_uploader("Upload Volume h5 Here", type=['h5'])
            if h5file is not None:
                st.session_state['my_recon_volume'] = BirefringentVolume.init_from_file(h5file, backend=backend, \
                                                        optical_info=optical_info)
        else:
            volume_type = st.selectbox('Volume type',['ellipsoid','shell','2ellipsoids','single_voxel'],1)
            st.session_state['my_recon_volume'] = BirefringentVolume.create_dummy_volume(backend=backend, optical_info=optical_info, \
                                        vol_type=volume_type, volume_axial_offset=volume_axial_offset)

st.subheader("Volume viewing")
st.write("See Forward Projection page for plotting")
# if st.button("Plot volume!"):
#     st.write("Scroll over image to zoom in and out.")
#     with torch.no_grad():
#         my_fig = st.session_state['my_recon_volume'].plot_volume_plotly(optical_info, 
#                                 voxels_in=st.session_state['my_recon_volume'].Delta_n, opacity=0.1)
#     st.plotly_chart(my_fig)
######################################################################

st.subheader("Training parameters")
n_epochs = st.slider('Number of iterations', min_value=1, max_value=30, value=10)
# want learning rate to be multiple choice
# lr = st.slider('Learning rate', min_value=1, max_value=5, value=3) 
filename_message = st.text_input('Message to add to the filename (not currently saving anyway..)')
training_params = {
    'n_epochs' : n_epochs,
    'azimuth_weight' : 1,
    'lr' : 1e-2,
    'output_posfix' : '11ml_atan2loss'
}


if st.button("Reconstruct!"):
    my_volume = st.session_state['my_recon_volume']
    # output_dir = f'reconstructions/recons_{volume_type}_{optical_info["volume_shape"][0]} \
    #                 x{optical_info["volume_shape"][1]}x{optical_info["volume_shape"][2]}__{training_params["output_posfix"]}'
    # os.makedirs(output_dir, exist_ok=True)
    # torch.save({'optical_info' : optical_info,
    #             'training_params' : training_params,
    #             'volume_type' : volume_type}, f'{output_dir}/parameters.pt')


    # Create a Birefringent Raytracer
    rays = BirefringentRaytraceLFM(backend=backend, optical_info=optical_info)
    rays.compute_rays_geometry()
    if backend == BackEnds.PYTORCH:
        device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        # Force cpu, as for now cpu is faster
        device = "cpu"
        print(f'Using computing device: {device}')
        rays = rays.to(device)


    with torch.no_grad():
        # Perform same calculation with torch
        startTime = time.time()
        ret_image_measured, azim_image_measured = rays.ray_trace_through_volume(my_volume)
        executionTime = (time.time() - startTime)
        print('Warmup time in seconds with Torch: ' + str(executionTime))

        # Store GT images
        Delta_n_GT = my_volume.get_delta_n().detach().clone()
        optic_axis_GT = my_volume.get_optic_axis().detach().clone()
        ret_image_measured = ret_image_measured.detach()
        azim_image_measured = azim_image_measured.detach()
        azim_comp_measured = torch.arctan2(torch.sin(azim_image_measured), \
                                torch.cos(azim_image_measured)).detach()


    ############# 
    # Let's create an optimizer
    # Initial guess
    my_volume = BirefringentVolume( backend=backend, optical_info=optical_info, \
                                    volume_creation_args = {'init_mode' : 'random'})
    my_volume.members_to_learn.append('Delta_n')
    my_volume.members_to_learn.append('optic_axis')
    my_volume = my_volume.to(device)

    optimizer = torch.optim.Adam(my_volume.get_trainable_variables(), lr=training_params['lr'])
    loss_function = torch.nn.L1Loss()

    # To test differentiability let's define a loss function L = |ret_image_torch|, and minimize it
    losses = []
    plt.ion()
    figure = plt.figure(figsize=(18,6))
    plt.rcParams['image.origin'] = 'lower'

    st.write("Working on these ", n_epochs, "iterations...")
    my_bar = st.progress(0)
    width = st.sidebar.slider("Plot width", 1, 25, 15)
    height = st.sidebar.slider("Plot height", 1, 25, 8)
    co_gt, ca_gt = ret_image_measured*torch.cos(azim_image_measured), ret_image_measured*torch.sin(azim_image_measured)

    my_plot = st.empty() # set up a place holder for the plot

    for ep in tqdm(range(training_params['n_epochs']), "Minimizing"):
        optimizer.zero_grad()
        ret_image_current, azim_image_current = rays.ray_trace_through_volume(my_volume)

        azim_diff = azim_comp_measured - torch.arctan2(torch.sin(azim_image_current), torch.cos(azim_image_current))
        L = loss_function(ret_image_measured, ret_image_current) + \
            training_params['azimuth_weight'] * (2 * (1 - torch.cos(azim_image_measured - azim_image_current))).mean()

        # Calculate update of the my_volume (Compute gradients of the L with respect to my_volume)
        L.backward()
        # Apply gradient updates to the volume
        optimizer.step()
        # print(f'Ep:{ep} loss: {L.item()}')
        losses.append(L.item())

        percent_complete = int(ep / training_params['n_epochs'] * 100)
        my_bar.progress(percent_complete + 1)

        if ep%4==0:
            plt.clf()
            # fig = plt.figure()
            
            fig, ax = plt.subplots(figsize=(width, height))
            plt.subplot(2,4,1)
            plt.imshow(ret_image_measured.detach().cpu().numpy())
            plt.colorbar()
            plt.title('Initial Retardance')
            plt.subplot(2,4,2)
            plt.imshow(azim_image_measured.detach().cpu().numpy())
            plt.colorbar()
            plt.title('Initial Azimuth')
            plt.subplot(2,4,3)
            plt.imshow(volume_2_projections(Delta_n_GT.unsqueeze(0))[0,0] \
                                            .detach().cpu().numpy())
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
            plt.imshow(volume_2_projections(my_volume.get_delta_n().unsqueeze(0))[0,0] \
                                            .detach().cpu().numpy())
            plt.colorbar()
            plt.title('Final Volume MIP')
            plt.subplot(2,4,8)
            plt.plot(list(range(len(losses))),losses)
            plt.gca().yaxis.set_label_position("right")
            plt.gca().yaxis.tick_right()
            plt.xlabel('Epoch')
            plt.ylabel('Loss')

            # figure.canvas.draw()
            # figure.canvas.flush_events()
            # time.sleep(0.1)
            # plt.savefig(f"{output_dir}/Optimization_ep_{'{:02d}'.format(ep)}.pdf")
            # time.sleep(0.1)
            my_plot.pyplot(fig)
            # st.image(fig)


    # st.pyplot(fig)
    # Display
    # plt.savefig(f"{output_dir}/g_Optimization_final.pdf")
    # plt.show()

    st.success("Done reconstructing! How does it look?", icon="✅")