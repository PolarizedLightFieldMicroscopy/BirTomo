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
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume, BirefringentRaytraceLFM
from VolumeRaytraceLFM.optic_config import volume_2_projections
from plotting_tools import plot_iteration_update

st.header("Choose our parameters")

columns = st.columns(2)
#first Column
with columns[0]:
############ Optical Params #################
    # Get optical parameters template
    optical_info = BirefringentVolume.get_optical_info_template()
    # Alter some of the optical parameters
    st.subheader('Optical')
    optical_info['n_micro_lenses'] = st.slider('Number of microlenses', min_value=1, max_value=51, value=5)
    optical_info['pixels_per_ml'] = st.slider('Pixels per microlens', min_value=1, max_value=33, value=17, step=2)
    optical_info['n_voxels_per_ml'] = st.slider('Number of voxels per microlens', min_value=1, max_value=7, value=1)

############ Other #################
    st.subheader('Other')
    backend_choice = st.radio('Backend', ['torch'])
    st.write("Backend needs to be torch for the reconstructions")
    backend = BackEnds.PYTORCH

def key_investigator(key_home, my_str='', prefix='- '):
    if hasattr(key_home, 'keys'):
        for my_key in key_home.keys():
            my_str = my_str + prefix + my_key +'\n'
            my_str = key_investigator(key_home[my_key], my_str, '\t'+prefix)
    return my_str

# Second Column
with columns[1]:
############ Volume #################
    st.subheader('Volume')
    volume_container = st.container() # set up a home for other volume selections to go
    with volume_container:
        how_get_vol = st.radio("Volume can be created or uploaded as an h5 file",
                                ['h5 upload', 'Create a new volume', 'Upload experimental images'], index=1)
        if how_get_vol == 'h5 upload':
            h5file = st.file_uploader("Upload Volume h5 Here", type=['h5'])
            optical_info['n_voxels_per_ml_volume'] = st.slider('Number of voxels per microlens in volume space', min_value=1, max_value=21, value=3)
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
            retardance_path = st.file_uploader("Upload retardance tif", type=['png'])
            azimuth_path = st.file_uploader("Upload orientation tif", type=['png'])
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
            optical_info['volume_shape'][1] = st.slider('Y volume dimension',
                                                        min_value=1, max_value=100, value=51)
            optical_info['volume_shape'][2] = st.slider('Z volume dimension',
                                                        min_value=1, max_value=100,
                                                        value=optical_info['volume_shape'][1])
            
        else:
            optical_info['n_voxels_per_ml_volume'] = st.slider('Number of voxels per microlens in volume space', min_value=1, max_value=21, value=3)
            volume_type = st.selectbox('Volume type',
                                       ['ellipsoid','shell','2ellipsoids','single_voxel'], 1)
            st.subheader('Volume shape')
            optical_info['volume_shape'][0] = st.slider('Axial volume dimension',
                                                        min_value=1, max_value=50, value=15)
            # y will follow x if x is changed. x will not follow y if y is changed
            optical_info['volume_shape'][1] = st.slider('Y volume dimension',
                                                        min_value=1, max_value=100, value=51)
            optical_info['volume_shape'][2] = st.slider('Z volume dimension',
                                                        min_value=1, max_value=100,
                                                        value=optical_info['volume_shape'][1])
            shift_from_center = st.slider('Axial shift from center [voxels]',
                                        min_value = -int(optical_info['volume_shape'][0] / 2),
                                        max_value = int(optical_info['volume_shape'][0] / 2), value = 0)
            volume_axial_offset = optical_info['volume_shape'][0] // 2 + shift_from_center # for center
    # Create the volume based on the selections.
    with volume_container:
        if how_get_vol == 'h5 upload':
            if h5file is not None:
                # Lets create a new optical info for volume space, as the sampling might be higher than the reconstruction
                optical_info_volume = copy.deepcopy(optical_info)
                optical_info_volume['n_voxels_per_ml'] = optical_info_volume['n_voxels_per_ml_volume']
                optical_info_volume['volume_shape'][1] = 501
                optical_info_volume['volume_shape'][2] = 501

                st.session_state['my_volume'] = BirefringentVolume.init_from_file(
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
            optical_info_volume['volume_shape'][1] = 501
            optical_info_volume['volume_shape'][2] = 501
            with torch.no_grad():
                st.session_state['my_volume'] = BirefringentVolume.create_dummy_volume(
                                                    backend=backend,
                                                    optical_info=optical_info_volume,
                                                    vol_type=volume_type,
                                                    volume_axial_offset=volume_axial_offset
                                                    )

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
n_epochs = st.slider('Number of iterations', min_value=1, max_value=100, value=10)
# want learning rate to be multiple choice
# lr = st.slider('Learning rate', min_value=1, max_value=5, value=3) 
filename_message = st.text_input('Message to add to the filename (not currently saving anyway..)')
training_params = {
    'n_epochs' : n_epochs,                      # How long to train for
    'azimuth_weight' : .5,                   # Azimuth loss weight
    'regularization_weight' : 1.0,          # Regularization weight
    'lr' : 1e-3,                            # Learning rate
    'output_posfix' : '15ml_bundleX_E_vector_unit_reg'     # Output file name posfix
}


if st.button("Reconstruct!"):
    my_volume = st.session_state['my_volume']
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
            ret_image_measured, azim_image_measured = rays_higher_sampling.ray_trace_through_volume(my_volume)
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
    volume_estimation = BirefringentVolume(backend=backend, optical_info=optical_info, \
                                    volume_creation_args = {'init_mode' : 'random'})

    # Let's rescale the random to initialize the volume
    volume_estimation.Delta_n.requires_grad = False
    volume_estimation.optic_axis.requires_grad = False
    volume_estimation.Delta_n *= 0.0001
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

    # Create an optimizer
    optimizer = torch.optim.Adam(trainable_parameters, lr=training_params['lr'])
    
    # To test differentiability let's define a loss function L = |ret_image_torch|, and minimize it
    losses = []
    data_term_losses = []
    regularization_term_losses = []
    
    # Create weight mask for the azimuth
    # as the azimuth is irrelevant when the retardance is low, lets scale error with a mask
    azimuth_damp_mask = (ret_image_measured / ret_image_measured.max()).detach()


    # width = st.sidebar.slider("Plot width", 1, 25, 15)
    # height = st.sidebar.slider("Plot height", 1, 25, 8)
    co_gt, ca_gt = ret_image_measured*torch.cos(azim_image_measured), ret_image_measured*torch.sin(azim_image_measured)

    my_plot = st.empty() # set up a place holder for the plot
    
    st.write("Working on these ", n_epochs, "iterations...")
    my_bar = st.progress(0)
    for ep in tqdm(range(training_params['n_epochs']), "Minimizing"):
        optimizer.zero_grad()
        
        # Forward projection
        ret_image_current, azim_image_current = rays.ray_trace_through_volume(volume_estimation)

        # Vector difference
        co_pred, ca_pred = ret_image_current*torch.cos(azim_image_current), ret_image_current*torch.sin(azim_image_current)
        data_term = ((co_gt-co_pred)**2 + (ca_gt-ca_pred)**2).mean()
        regularization_term  = (1-(volume_estimation.optic_axis[0,...]**2+volume_estimation.optic_axis[1,...]**2+volume_estimation.optic_axis[2,...]**2)).abs().mean()
        L = data_term + training_params['regularization_weight'] * regularization_term
        # Calculate update of the my_volume (Compute gradients of the L with respect to my_volume)
        L.backward()
        
        # Apply gradient updates to the volume
        optimizer.step()

        # print(f'Ep:{ep} loss: {L.item()}')
        losses.append(L.item())
        data_term_losses.append(data_term.item())
        regularization_term_losses.append(regularization_term.item())

        azim_image_out = azim_image_current.detach()
        azim_image_out[azimuth_damp_mask==0] = 0

        percent_complete = int(ep / training_params['n_epochs'] * 100)
        my_bar.progress(percent_complete + 1)

        if ep%2==0:
            fig = plot_iteration_update(
                volume_2_projections(Delta_n_GT.unsqueeze(0))[0,0].detach().cpu().numpy(),
                ret_image_measured.detach().cpu().numpy(),
                azim_image_measured.detach().cpu().numpy(),
                volume_2_projections(volume_estimation.get_delta_n().unsqueeze(0))[0,0].detach().cpu().numpy(),
                ret_image_current.detach().cpu().numpy(),
                np.rad2deg(azim_image_current.detach().cpu().numpy()),
                losses,
                data_term_losses,
                regularization_term_losses,
                streamlit_purpose=True
                )
            
            my_plot.pyplot(fig)

    st.success("Done reconstructing! How does it look?", icon="✅")


    st.subheader('Download results')
    # print(ret_image_current.detach().cpu().numpy().shape)
    # st.download_button('Download estimated retardance', ret_image_current.detach().cpu().numpy().tobytes(), mime="image/jpeg")
    # st.download_button('Download estimated orientation', azim_image_current.detach().cpu().numpy().tobytes(), mime="image/jpeg")

    # Save volume to h5
    h5_file = io.BytesIO()
    st.download_button('Download estimated volume as HDF5 file', volume_estimation.save_as_file(h5_file), mime='application/x-hdf5')