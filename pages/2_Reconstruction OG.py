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
import io
import json
import copy
import numpy as np
import torch
from PIL import Image
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib 
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume, BirefringentRaytraceLFM
from VolumeRaytraceLFM.visualization.plotting_volume import volume_2_projections
from VolumeRaytraceLFM.visualization.plotting_iterations import plot_iteration_update
from VolumeRaytraceLFM.loss_functions import *

st.header("Choose our parameters")

columns = st.columns(2)
#first Column
with columns[0]:
############ Optical Params #################
    # Get optical parameters template
    optical_info = BirefringentVolume.get_optical_info_template()
    # Alter some of the optical parameters
    st.subheader('Optical')
    optical_info['n_micro_lenses'] = st.slider('Number of microlenses', min_value=1, max_value=51, value=9)
    optical_info['pixels_per_ml'] = st.slider('Pixels per microlens', min_value=1, max_value=33, value=17, step=2)
    # GT volume simulation
    optical_info['n_voxels_per_ml'] = st.slider('Number of voxels per microlens (volume sampling)', min_value=1, max_value=7, value=1)

############ Reconstruction settings #################
    backend = BackEnds.PYTORCH
    st.subheader("Iterative reconstruction parameters")
    n_epochs = st.slider('Number of iterations', min_value=1, max_value=500, value=500)
    # See loss_functions.py for more details
    loss_function = st.selectbox('Loss function',
                                ['vonMisses', 'vector', 'L1_cos', 'L1all'], 1)
    regularization_function1 = st.selectbox('Volume regularization function 1',
                                ['L1', 'L2', 'unit', 'TV', 'none'], 2)
    regularization_function2 = st.selectbox('Volume regularization function 2',
                                ['L1', 'L2', 'unit', 'TV', 'none'], 4)
    reg_weight1 = st.number_input('Regularization weight 1', min_value=0., max_value=0.5, value=0.5)
    # st.write('The current regularization weight 1 is ', reg_weight1)
    reg_weight2 = st.number_input('Regularization weight 2', min_value=0., max_value=0.5, value=0.5)
    # st.write('The current regularization weight 2 is ', reg_weight2)
    ret_azim_weight = st.number_input('Retardance-Orientation weight', min_value=0., max_value=1., value=0.5)
    # st.write('The current retardance/orientation weight is ', ret_azim_weight)
    st.subheader("Initial estimated volume")
    volume_init_type = st.selectbox('Initial volume type',
                                       ['random', 'upload'], 0)
    if volume_init_type == 'upload':
        h5file_init = st.file_uploader("Upload the initial volume h5 Here", type=['h5'])  
        delta_n_init_magnitude = 1
        mask_bool = False      
    else:
        mask_bool = st.checkbox('Mask out area unreachable by light rays')
        delta_n_init_magnitude = st.number_input('Volume Delta_n initial magnitude', min_value=0., max_value=1., value=0.0001, format="%0.5f")
        # st.write('The current Volume Delta_n initial magnitude is ', delta_n_init_magnitude)
    
    st.subheader('Learning rate')
    learning_rate_delta_n = st.number_input('Learning rate for Delta_n', min_value=0.0, max_value=10.0, value=0.001, format="%0.5f")
    # st.write('The current LR is ', learning_rate_delta_n)
    learning_rate_optic_axis = st.number_input('Learning rate for optic_axis', min_value=0.0, max_value=10.0, value=0.001, format="%0.5f")
    # st.write('The current optic axis LR is ', learning_rate_optic_axis)

    
def key_investigator(key_home, my_str='', prefix='- '):
    if hasattr(key_home, 'keys'):
        for my_key in key_home.keys():
            my_str = my_str + prefix + my_key +'\n'
            my_str = key_investigator(key_home[my_key], my_str, '\t'+prefix)
    return my_str

# Second Column
with columns[1]:
############ Volume #################
    st.subheader('Ground truth volume')
    volume_container = st.container() # set up a home for other volume selections to go
    with volume_container:
        how_get_vol = st.radio("Volume can be created or uploaded as an h5 file",
                                ['h5 upload', 'Create a new volume', 'Upload experimental images'], index=1)
        if how_get_vol == 'h5 upload':
            h5file = st.file_uploader("Upload Volume h5 Here", type=['h5'])
            optical_info['n_voxels_per_ml_volume'] = st.slider('Number of voxels per microlens in volume', min_value=1, max_value=21, value=1)
            if h5file is not None:
                with h5py.File(h5file) as file:
                    try:
                        vol_shape = file['optical_info']['volume_shape'][()]
                    except KeyError:
                        st.error('This file does specify the volume shape.')
                    except Exception as e:
                        st.error(e)
                vol_shape_default = [int(v) for v in vol_shape]
                optical_info['volume_shape'] = vol_shape_default
                st.markdown(f"Using a cube volume shape with the dimension of the"
                            + f" loaded volume: {vol_shape_default}.")

                display_h5 = st.checkbox("Display h5 file contents")                
                if display_h5:
                    with h5py.File(h5file) as file:
                        st.markdown('**File Structure:**\n' + key_investigator(file))
                        try:
                            st.markdown('**Description:** '+str(file['optical_info']['description'][()])[2:-1])
                        except KeyError:
                            st.error('This file does not have a description.')
                        except Exception as e:
                            st.error(e)
                        try:
                            vol_shape = file['optical_info']['volume_shape'][()]
                            # optical_info['volume_shape'] = vol_shape
                            st.markdown(f"**Volume Shape:** {vol_shape}")
                        except KeyError:
                            st.error('This file does specify the volume shape.')
                        except Exception as e:
                            st.error(e)
                        try:
                            voxel_size = file['optical_info']['voxel_size_um'][()]
                            st.markdown(f"**Voxel Size (um):** {voxel_size}")
                        except KeyError:
                            st.error('This file does specify the voxel size. Voxels are likely to be cubes.')
                        except Exception as e:
                            st.error(e)
        elif how_get_vol == 'Upload experimental images':
            retardance_path = st.file_uploader("Upload retardance tif", type=['png', 'tif', 'tiff'])
            azimuth_path = st.file_uploader("Upload orientation tif", type=['png', 'tif', 'tiff'])
            metadata_file = st.file_uploader("Upload metadata from Napari-LF", type=['txt'])
            plot_cropped_imgs = st.empty() # set up a place holder for the plot

            # Load files
            if retardance_path is not None:
                ret_img_raw = torch.from_numpy(np.array(Image.open(retardance_path)).astype(np.float32))
            if azimuth_path is not None:
                azim_img_raw = torch.from_numpy(np.array(Image.open(azimuth_path)).astype(np.float32))

            if metadata_file is not None:
                # Lets load metadata
                metadata = metadata_file.read()
                metadata =  json.loads(metadata)
                # MLA data
                optical_info['pixels_per_ml'] = metadata['calibrate']['pixels_per_ml'] if 'pixels_per_ml' in metadata['calibrate'].keys() else optical_info['pixels_per_ml']
                # optical_info['n_micro_lenses']      = 11
                # optical_info['n_voxels_per_ml']     = 1
                # optical_info['axial_voxel_size_um'] = 1
                # Optics data
                optical_info['M_obj']           = metadata['calibrate']['objective_magnification']
                optical_info['na_obj']          = metadata['calibrate']['objective_na']
                optical_info['n_medium']        = metadata['calibrate']['medium_index']
                optical_info['wavelength']      = metadata['calibrate']['center_wavelength']
                optical_info['camera_pix_pitch']= metadata['calibrate']['pixel_size']

            ### Which part to crop from the images?

            if retardance_path and azimuth_path and metadata_file:
                # Crop data based on n_micro_lenses and n_voxels_per_ml
                n_mls_y = ret_img_raw.shape[0] // optical_info['pixels_per_ml']
                n_mls_x = ret_img_raw.shape[1] // optical_info['pixels_per_ml']
 
                crop_pos_y = st.slider('Image region center Y', min_value=1, max_value=n_mls_y, value=55)
                crop_pos_x = st.slider('Image region center X', min_value=1, max_value=n_mls_x, value=54)
                start_ml = [crop_pos_y,crop_pos_x]
                start_coords = [sc * optical_info['pixels_per_ml'] for sc in start_ml]
                end_coords = [sc + optical_info['n_micro_lenses'] * optical_info['pixels_per_ml'] for sc in start_coords ]

                st.session_state['ret_image_measured'] = ret_img_raw[start_coords[0]:end_coords[0], start_coords[1] : end_coords[1]]
                st.session_state['azim_image_measured'] = azim_img_raw[start_coords[0]:end_coords[0], start_coords[1] : end_coords[1]]

                # Plot images
                fig = plt.figure(figsize=(6,3))
                plt.rcParams['image.origin'] = 'lower'
                plt.subplot(1,2,1)
                plt.imshow(st.session_state['ret_image_measured'].numpy(), cmap='gray')
                plt.subplot(1,2,2)
                plt.imshow(st.session_state['azim_image_measured'].numpy(), cmap='gray')
                plot_cropped_imgs.pyplot(fig)

            st.subheader('Volume shape')
            optical_info['volume_shape'][0] = st.slider('Axial volume dimension',
                                                        min_value=1, max_value=50, value=15)
            # y will follow x if x is changed. x will not follow y if y is changed
            optical_info['volume_shape'][1] = st.slider('Y-Z volume dimension',
                                                        min_value=1, max_value=100, value=51)
            optical_info['volume_shape'][2] = optical_info['volume_shape'][1]
        else:
            optical_info['n_voxels_per_ml_volume'] = st.slider('Number of voxels per microlens in volume space', min_value=1, max_value=21, value=1)
            volume_type = st.selectbox('Volume type',
                                       ['ellipsoid','shell','2ellipsoids','single_voxel'], 3)
            st.subheader('Volume shape')
            optical_info['volume_shape'][0] = st.slider('Axial volume dimension',
                                                        min_value=1, max_value=50, value=5)
            # y will follow x if x is changed. x will not follow y if y is changed
            optical_info['volume_shape'][1] = st.slider('Y-Z volume dimension',
                                                        min_value=1, max_value=100, value=51)
            optical_info['volume_shape'][2] = optical_info['volume_shape'][1]
            shift_from_center = st.slider('Axial shift from center [voxels]',
                                        min_value = -int(optical_info['volume_shape'][0] / 2),
                                        max_value = int(optical_info['volume_shape'][0] / 2), value = -1)
            volume_axial_offset = optical_info['volume_shape'][0] // 2 + shift_from_center # for center
    # Create the volume based on the selections.
    with volume_container:
        if how_get_vol == 'h5 upload':
            # Upload ground truth volume from an h5 file
            if h5file is not None:
                # Lets create a new optical info for volume space, as the sampling might be higher than the reconstruction
                # DEBUG: might not be using the 'n_voxels_per_ml_volume' selected above
                # deepcopy may disregard the selection made above int he ground truth volume
                optical_info_volume = copy.deepcopy(optical_info)
                optical_info_volume['n_voxels_per_ml'] = optical_info_volume['n_voxels_per_ml_volume']
                # optical_info_volume['n_voxels_per_ml'] = 3
                # optical_info_volume['volume_shape'][1] = 501
                # optical_info_volume['volume_shape'][2] = 501
                st.session_state['my_volume'] = BirefringentVolume.init_from_file(
                                                        h5file,
                                                        backend=backend,
                                                        optical_info=optical_info_volume
                                                        )
                test_vol = st.session_state['my_volume'] = BirefringentVolume.init_from_file(
                            h5file,
                            backend=backend,
                            optical_info=optical_info_volume
                            )
        elif how_get_vol == 'Upload experimental images':
            with torch.no_grad():
                st.session_state['my_volume'] = BirefringentVolume.create_dummy_volume(
                                                    backend=backend,
                                                    optical_info=optical_info,
                                                    vol_type='zeros',
                                                    volume_axial_offset=0
                                                    )
        else:
            # Lets create a new optical info for volume space, as the sampling might be higher than the reconstruction
            optical_info_volume = copy.deepcopy(optical_info)
            optical_info_volume['n_voxels_per_ml'] = optical_info_volume['n_voxels_per_ml_volume']
            # optical_info_volume['volume_shape'][1] = 501
            # optical_info_volume['volume_shape'][2] = 501
            with torch.no_grad():
                st.session_state['my_volume'] = BirefringentVolume.create_dummy_volume(
                                                    backend=backend,
                                                    optical_info=optical_info_volume,
                                                    vol_type=volume_type,
                                                    volume_axial_offset=volume_axial_offset
                                                    )

######################################################################

# want learning rate to be multiple choice
# lr = st.slider('Learning rate', min_value=1, max_value=5, value=3) 
# filename_message = st.text_input('Message to add to the filename (not currently saving anyway..)')
training_params = {
    'n_epochs' : n_epochs,                          # How long to train for
    'azimuth_weight' : ret_azim_weight,             # Azimuth loss weight
    'regularization_weight' : [reg_weight1, reg_weight2],           # Regularization weight
    'lr' : learning_rate_delta_n,                   # Learning rate for delta_n
    'lr_optic_axis' : learning_rate_optic_axis,     # Learning rate for optic axis
    'output_posfix' : '',                           # Output file name posfix
    'loss' : loss_function,                         # Loss function
    'reg' : [regularization_function1, regularization_function2]                 # Regularization function
}

if st.button("Reconstruct!"):
    my_volume = st.session_state['my_volume']

    # Create a Birefringent Raytracer
    # DEBUG
    # Force the volume shape to be smaller for the reconstructed volume
    st.subheader('Volume shape for the estimated volume')
    estimated_volume_shape = optical_info['volume_shape'].copy()
    estimated_volume_shape[0] = st.slider('Axial volume dimension for estimated volume',
                                                min_value=1, max_value=50, value=5)
    # y will follow x if x is changed. x will not follow y if y is changed
    estimated_volume_shape[1] = st.slider('Y-Z volume dimension for estimated volume',
                                                min_value=1, max_value=100, value=31)
    estimated_volume_shape[2] = estimated_volume_shape[1]
    optical_info['volume_shape'] = estimated_volume_shape
    rays = BirefringentRaytraceLFM(backend=backend, optical_info=optical_info)
    rays.compute_rays_geometry()
    if backend == BackEnds.PYTORCH:
        device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        # Force cpu, as for now cpu is faster
        device = "cpu"
        print(f'Using computing device: {device}')
        rays = rays.to_device(device)

    # Generate images using the forward model
    with torch.no_grad():
        if how_get_vol == 'Upload experimental images':
            ret_image_measured  =  st.session_state['ret_image_measured']
            azim_image_measured =  st.session_state['azim_image_measured']
            # Normalize data
            ret_image_measured /= ret_image_measured.max()
            ret_image_measured *= 0.01
            azim_image_measured *= torch.pi / azim_image_measured.max()
        else:
            # We need a raytracer with different number of voxels per ml for higher sampling measurements
            rays_higher_sampling = BirefringentRaytraceLFM(backend=backend, optical_info=optical_info_volume)
            rays_higher_sampling.compute_rays_geometry()
            # Perform same calculation with torch
            start_time = time.time()
            [ret_image_measured, azim_image_measured] = rays_higher_sampling.ray_trace_through_volume(my_volume)
            execution_time = (time.time() - start_time)
            print('Warmup time in seconds with Torch: ' + str(execution_time))

        # Store GT images
        Delta_n_GT = my_volume.get_delta_n().detach().clone()
        optic_axis_GT = my_volume.get_optic_axis().detach().clone()
        ret_image_measured = ret_image_measured.detach()
        azim_image_measured = azim_image_measured.detach()

    ############# 
    # Let's create an optimizer
    # Initial guess
    if volume_init_type == 'upload' and h5file_init is not None:
        volume_estimation = BirefringentVolume.init_from_file(
                                        h5file_init,
                                        backend=backend,
                                        optical_info=optical_info
                                        )
    else:
        volume_estimation = BirefringentVolume(backend=backend, optical_info=optical_info, \
                                        volume_creation_args = {'init_mode' : 'random'})
        # Let's rescale the random to initialize the volume
        volume_estimation.Delta_n.requires_grad = False
        volume_estimation.optic_axis.requires_grad = False
        volume_estimation.Delta_n *= delta_n_init_magnitude
        if mask_bool:
            # And mask out volume that is outside FOV of the microscope
            mask = rays.get_volume_reachable_region()
            volume_estimation.Delta_n[mask.view(-1)==0] = 0
        volume_estimation.Delta_n.requires_grad = True
        volume_estimation.optic_axis.requires_grad = True

    # Indicate to this object that we are going to optimize Delta_n and optic_axis
    volume_estimation.members_to_learn.append('Delta_n')
    volume_estimation.members_to_learn.append('optic_axis')
    volume_estimation = volume_estimation.to(device)

    trainable_parameters = volume_estimation.get_trainable_variables()

    # As delta_n has much lower values than optic_axis, we might need 2 different learning rates
    parameters = [{'params': trainable_parameters[0], 'lr': training_params['lr_optic_axis']},  # Optic axis
                {'params': trainable_parameters[1], 'lr': training_params['lr']}]               # Delta_n

    # Create optimizer 
    optimizer = torch.optim.Adam(parameters, lr=training_params['lr'])

    # To test differentiability let's define a loss function L = |ret_image_torch|, and minimize it
    losses = []
    data_term_losses = []
    regularization_term_losses = []

    # Create weight mask for the azimuth
    # as the azimuth is irrelevant when the retardance is low, lets scale error with a mask
    azimuth_damp_mask = (ret_image_measured / ret_image_measured.max()).detach()

    # width = st.sidebar.slider("Plot width", 1, 25, 15)
    # height = st.sidebar.slider("Plot height", 1, 25, 8)

    my_plot = st.empty() # set up a place holder for the plot
    my_3D_plot = st.empty() # set up a place holder for the 3D plot

    st.write("Working on these ", n_epochs, "iterations...")
    my_bar = st.progress(0)
    for ep in tqdm(range(training_params['n_epochs']), "Minimizing"):
        optimizer.zero_grad()

        # Forward projection
        [ret_image_current, azim_image_current] = rays.ray_trace_through_volume(volume_estimation)

        # Conpute loss and regularization        
        L, data_term, regularization_term = apply_loss_function_and_reg(training_params['loss'], training_params['reg'], ret_image_measured, azim_image_measured, 
                                                ret_image_current, azim_image_current, 
                                                training_params['azimuth_weight'], volume_estimation, training_params['regularization_weight'])

        # Calculate update of the my_volume (Compute gradients of the L with respect to my_volume)
        L.backward()

        # Apply gradient updates to the volume
        optimizer.step()
        with torch.no_grad():
            num_nan_vecs = torch.sum(torch.isnan(volume_estimation.optic_axis[0, :]))
            replacement_vecs = torch.nn.functional.normalize(torch.rand(3, int(num_nan_vecs)), p=2, dim=0)
            volume_estimation.optic_axis[:, torch.isnan(volume_estimation.optic_axis[0, :])] = replacement_vecs
            if ep == 0 and num_nan_vecs != 0:
                st.write(f"Replaced {num_nan_vecs} NaN optic axis vectors with random unit vectors, " +
                         "likely on every iteration.")
        # print(f'Ep:{ep} loss: {L.item()}')
        losses.append(L.item())
        data_term_losses.append(data_term.item())
        regularization_term_losses.append(regularization_term.item())

        azim_image_out = azim_image_current.detach()
        azim_image_out[azimuth_damp_mask==0] = 0

        percent_complete = int(ep / training_params['n_epochs'] * 100)
        my_bar.progress(percent_complete + 1)

        if ep%2==0:
            matplotlib.pyplot.close()
            fig = plot_iteration_update(
                volume_2_projections(Delta_n_GT.unsqueeze(0))[0,0].detach().cpu().numpy(),
                ret_image_measured.detach().cpu().numpy(),
                azim_image_measured.detach().cpu().numpy(),
                volume_2_projections(volume_estimation.get_delta_n().unsqueeze(0))[0,0].detach().cpu().numpy(),
                ret_image_current.detach().cpu().numpy(),
                azim_image_current.detach().cpu().numpy(),
                losses,
                data_term_losses,
                regularization_term_losses,
                streamlit_purpose=True
                )

            my_plot.pyplot(fig)

    st.success("Done reconstructing! How does it look?", icon="✅")
    st.session_state['my_volume'] = volume_estimation
    st.write("Scroll over image to zoom in and out.")
    # Todo: use a slider to filter the volume
    volume_ths = 0.05 #st.slider('volume ths', min_value=0., max_value=1., value=0.1)
    matplotlib.pyplot.close()
    my_fig = st.session_state['my_volume'].plot_lines_plotly(delta_n_ths=volume_ths)
    st.plotly_chart(my_fig, use_container_width=True)

    st.subheader('Download results')
    # print(ret_image_current.detach().cpu().numpy().shape)
    # st.download_button('Download estimated retardance', ret_image_current.detach().cpu().numpy().tobytes(), mime="image/jpeg")
    # st.download_button('Download estimated orientation', azim_image_current.detach().cpu().numpy().tobytes(), mime="image/jpeg")

    # Save volume to h5
    h5_file = io.BytesIO()
    st.download_button('Download estimated volume as HDF5 file', volume_estimation.save_as_file(h5_file), mime='application/x-hdf5')
