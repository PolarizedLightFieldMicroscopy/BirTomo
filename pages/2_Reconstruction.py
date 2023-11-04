import streamlit as st

st.set_page_config(
    page_title="Reconstructions",
    page_icon="👋",
    layout="wide",
)

st.title("Reconstructions")
st.write("Let's try to reconstruct a volume based on our images!")
st.markdown('''
        1. Select the microscope optical parameters and the optimization parameters.
        1. Capture experimental or synethic polarized light field microscopy images.
        1. Reconstruct the birefringent volume based on the images.
    ''')

import io
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
import pandas as pd
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume, BirefringentRaytraceLFM
from VolumeRaytraceLFM.loss_functions import *
from utils.parameters import (
        forward_propagate,
        dataframe_to_dict,
        extract_scalar_params,
        get_vol_shape_from_h5,
        display_h5_metadata
    )
from VolumeRaytraceLFM.visualization.plotting_ret_azim import plot_retardance_orientation

# st.sidebar.selectbox('Side bar variable', options=['Rudolf', 'Grant'], index=0)

# st.session_state['optical_info'] = BirefringentVolume.get_optical_info_template()
# optical_info = st.session_state['optical_info']

# 'n_voxels_per_ml'
# 'Number of voxels per microlens'

tabs = st.tabs(["Optical", "Optimization"])
for i, tab in enumerate(tabs):
    pass

####### Optical parameters ##################
tabs[0].subheader('Microscope configuration')
optical_info = BirefringentVolume.get_optical_info_template()
df_microscope = extract_scalar_params(optical_info)
df_microscope.loc[0] = ['Number of microlenses', 3]
edited_df_microscope = tabs[0].data_editor(df_microscope)
values_from_df = dataframe_to_dict(edited_df_microscope)
optical_info.update(values_from_df)

# st.write(st.session_state)

#################################################

############## Optimization parameters ###################
tabs[1].subheader("Iterative reconstruction parameters")
n_epochs = tabs[1].slider('Number of iterations', min_value=1, max_value=500, value=200)
optim_cols = tabs[1].columns(2)
# See loss_functions.py for more details
optim_cols[0].markdown('**Loss function**')
loss_function = optim_cols[0].selectbox('Loss function',
                            ['vonMisses', 'vector', 'L1_cos', 'L1all'], 1)
regularization_function1 = optim_cols[0].selectbox('Volume regularization function 1',
                            ['L1', 'L2', 'unit', 'TV', 'none'], 2)
regularization_function2 = optim_cols[0].selectbox('Volume regularization function 2',
                            ['L1', 'L2', 'unit', 'TV', 'none'], 4)
reg_weight1 = optim_cols[0].number_input('Regularization weight 1', min_value=0., max_value=0.5, value=0.5)
reg_weight2 = optim_cols[0].number_input('Regularization weight 2', min_value=0., max_value=0.5, value=0.5)
ret_azim_weight = optim_cols[0].number_input('Retardance-Orientation weight', min_value=0., max_value=1., value=0.5)
optim_cols[1].markdown('**Learning rate**')
learning_rate_delta_n = optim_cols[1].number_input('Learning rate for the birefringence $\Delta n$', min_value=0.0, max_value=10.0, value=0.001, format="%0.5f")
learning_rate_optic_axis = optim_cols[1].number_input('Learning rate for optic axis $\mathbf{a}$', min_value=0.0, max_value=10.0, value=0.001, format="%0.5f")

optim_cols[1].markdown('**Initial estimated volume**')
est_volume_init_type = optim_cols[1].selectbox('Initial volume type',
                                    ['random', 'upload'], 0)
if est_volume_init_type == 'upload':
    est_h5file_init = optim_cols[1].file_uploader(
            "Upload the initial volume h5 Here", type=['h5']
        )
    delta_n_init_magnitude = 1
    mask_bool = False
    optim_cols[1].write('TODO: add volume shape')
else:
    # Force the volume shape to be smaller for the reconstructed volume
    optim_cols[1].write('Estimated volume shape')
    # estimated_volume_shape = optical_info['volume_shape'].copy()
    est_vol_shape = np.array([0, 0, 0])
    est_vol_shape[0] = optim_cols[1].slider('Axial volume dimension',
                                                min_value=1, max_value=50, value=5)
    # y will follow x if x is changed. x will not follow y if y is changed
    est_vol_shape[1] = optim_cols[1].slider('Y-Z volume dimension',
                                                min_value=1, max_value=100, value=31)
    est_vol_shape[2] = est_vol_shape[1]
    mask_bool = optim_cols[1].checkbox('Mask out area unreachable by light rays')
    delta_n_init_magnitude = optim_cols[1].number_input(
            'Volume $$\Delta n$$ initial magnitude',
            min_value=0., max_value=1., value=0.0001, format="%0.5f"
        )

st.markdown("""---""")
st.subheader('Capture microscopy images')
forw_cols = st.columns(2)
# forw_cols[0].markdown('**Method of acquiring images**')


backend = BackEnds.PYTORCH


############ Volume #################
# st.subheader('Ground truth volume')
volume_container = st.container()
with volume_container:
    how_acquire = forw_cols[0].radio(
            "**Method of acquiring images**",
            ['External: experimental or synthetical', 'Internal'],
            index=1
        )
    how_synthetic = how_acquire
    if how_acquire == 'External: experimental or synthetical':
        forw_cols[1].write('Check back later :)')
    elif how_acquire == 'Internal':
        forw_cols[0].write('*Internal imaging method*')
        how_synthetic = forw_cols[0].radio("Method of acquiring birefringent volume to image",
                                ['Generate a new volume', 'Upload h5 file'], index=1)
        optical_info['n_voxels_per_ml_volume'] = forw_cols[0].slider(
                'Supersampling: number of voxels per microlens in volume',
                min_value=1, max_value=21, value=1
            )
        # optical_info_GT = copy.deepcopy(optical_info)
        # optical_info_GT['n_voxels_per_ml'] = optical_info_GT['n_voxels_per_ml_volume']
        if how_synthetic == 'Generate a new volume':
            forw_cols[1].write('Birefringent volume creation options')
            volume_type = forw_cols[1].selectbox('Volume type',
                                        ['ellipsoid','shell','single_voxel'], 2) #2ellipsoid not working
            forw_cols[1].write('*Volume shape*')
            optical_info['volume_shape'][0] = forw_cols[1].slider('Axial volume dimension of GT',
                                                        min_value=1, max_value=50, value=5)
            # y will follow x if x is changed. x will not follow y if y is changed
            optical_info['volume_shape'][1] = forw_cols[1].slider('Y-Z volume dimension of GT',
                                                        min_value=1, max_value=100, value=51)
            optical_info['volume_shape'][2] = optical_info['volume_shape'][1]
            shift_from_center = forw_cols[1].slider('Axial shift from center [voxels]',
                                        min_value = -int(optical_info['volume_shape'][0] / 2),
                                        max_value = int(optical_info['volume_shape'][0] / 2), value = 0)
            volume_axial_offset = optical_info['volume_shape'][0] // 2 + shift_from_center # for center
            with torch.no_grad():
                st.session_state.GT = BirefringentVolume.create_dummy_volume(
                                                backend=backend,
                                                optical_info=optical_info,
                                                vol_type=volume_type,
                                                volume_axial_offset=volume_axial_offset
                                            )
        elif how_synthetic == 'Upload h5 file':
            h5file = forw_cols[1].file_uploader("Upload a ground truth volume", type=['h5'])
            if h5file is not None:
                optical_info['volume_shape'] = get_vol_shape_from_h5(h5file)
                with forw_cols[1].expander("Attributes and metadata", False):
                    display_h5_metadata(h5file)
                # Create a birefringent volume object
                with torch.no_grad():
                    st.session_state.GT = BirefringentVolume.init_from_file(
                                            h5file,
                                            backend=backend,
                                            optical_info=optical_info
                    )

######################################################################
##### forward propagate


if st.button('**Calculate retardance and orientation images!**', key='calc1'):
    with torch.no_grad():
        rays = BirefringentRaytraceLFM(backend=backend, optical_info=optical_info)
        [ret_image, azim_image], execution_time = forward_propagate(rays, st.session_state.GT)
        # st.text(f'Execution time in seconds with backend {backend}: ' + str(execution_time))
        st.success(f"Geometric ray tracing was successful in {execution_time:.2f} secs!", icon="✅")
        st.session_state['ret_image'] = ret_image
        st.session_state['azim_image'] = azim_image
  
if "ret_image" in st.session_state:
    # Plot with streamlit
    azimuth_plot_type = st.selectbox('Azmiuth Plot Type', ['lines', 'hsv'], index = 1)
    output_ret_image = st.session_state['ret_image']
    output_azim_image = st.session_state['azim_image']
    my_fig = plot_retardance_orientation(output_ret_image, output_azim_image, azimuth_plot_type)
    st.pyplot(my_fig)
    # st.success("Images were successfully created!", icon="✅")


st.markdown("""---""")

##########################################################
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

def forward_one_iter(rays, volume):
    # Forward projection
    [ret_image_current, azim_image_current] = rays.ray_trace_through_volume(volume)
    return ret_image_current, azim_image_current

def loss_one_iter():
    # Conpute loss and regularization        
    L, data_term, regularization_term = apply_loss_function_and_reg(
            training_params['loss'],
            training_params['reg'],
            ret_image_measured,
            azim_image_measured,
            ret_image_current,
            azim_image_current,
            training_params['azimuth_weight'],
            volume_estimation,
            training_params['regularization_weight']
        )

    return L, data_term, regularization_term

def generate_random_vol(mask=False):
    '''Generate a random initial estimated volume.'''
    volume = BirefringentVolume(backend=backend, optical_info=optical_info, \
                                    volume_creation_args = {'init_mode' : 'random'})
    # Let's rescale the random to initialize the volume
    volume.Delta_n.requires_grad = False
    volume.optic_axis.requires_grad = False
    volume.Delta_n *= delta_n_init_magnitude
    if mask:
        # And mask out volume that is outside FOV of the microscope
        mask = rays_est.get_volume_reachable_region()
        volume.Delta_n[mask.view(-1)==0] = 0
    volume.Delta_n.requires_grad = True
    volume.optic_axis.requires_grad = True

    return volume

def create_optimizer(volume):
    '''Create optimizer for the variables that we want to estimate.'''
    trainable_parameters = volume.get_trainable_variables()

    # As delta_n has much lower values than optic_axis, we might need 2 different learning rates
    parameters = [{'params': trainable_parameters[0], 'lr': training_params['lr_optic_axis']},  # Optic axis
                {'params': trainable_parameters[1], 'lr': training_params['lr']}]               # Delta_n

    # Create optimizer
    optim = torch.optim.Adam(parameters, lr=training_params['lr'])

    return optim

if st.button("**Reconstruct birefringent volume!**"):
    assert backend == BackEnds.PYTORCH, 'backend must be torch'
    if 'ret_image' not in st.session_state:
        st.error("We need images before we can reconstruct a volume.")
    try:
        my_volume = st.session_state.GT
        Delta_n_GT = my_volume.get_delta_n().detach().clone()
        optic_axis_GT = my_volume.get_optic_axis().detach().clone()
    except NameError:
        st.error("Ground truth volume is unknown.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ret_image_measured = torch.tensor(output_ret_image, device=device)
    azim_image_measured = torch.tensor(output_azim_image, device=device)
    optical_info['volume_shape'] = list(est_vol_shape)
    # Compute ray geometry
    rays_est = BirefringentRaytraceLFM(backend=backend, optical_info=optical_info)
    rays_est.compute_rays_geometry()

    # Initial guess
    if est_volume_init_type == 'upload' and est_h5file_init is not None:
        volume_estimation = BirefringentVolume.init_from_file(
                                        est_h5file_init,
                                        backend=backend,
                                        optical_info=optical_info
                                        )
    else:
        volume_estimation = generate_random_vol(mask=mask_bool)

    # Move variables to the gpu if available
    volume_estimation = volume_estimation.to(device)
    my_volume = my_volume.to(device)
    rays_est = rays_est.to(device)

    # Indicate to this object that we are going to optimize Delta_n and optic_axis
    volume_estimation.members_to_learn.append('Delta_n')
    volume_estimation.members_to_learn.append('optic_axis')

    optimizer = create_optimizer(volume_estimation)

    # To test differentiability let's define a loss function L = |ret_image_torch|, and minimize it
    losses = []
    data_term_losses = []
    regularization_term_losses = []

    # Create weight mask for the azimuth
    # as the azimuth is irrelevant when the retardance is low, lets scale error with a mask
    azimuth_damp_mask = (ret_image_measured / ret_image_measured.max()).detach()

    my_recon_img_plot = st.empty()
    my_loss = st.empty()
    my_plot = st.empty() # set up a place holder for the plot
    my_3D_plot = st.empty() # set up a place holder for the 3D plot

    st.write("Working on these ", n_epochs, "iterations...")
    my_bar = st.progress(0)
    for ep in tqdm(range(training_params['n_epochs']), "Minimizing"):
        optimizer.zero_grad()
        ret_image_current, azim_image_current = forward_one_iter(rays_est, volume_estimation)
        L, data_term, regularization_term = loss_one_iter()
        # Calculate update of the my_volume (Compute gradients of the L with respect to my_volume)
        L.backward()
        # Apply gradient updates to the volume
        optimizer.step()
        with torch.no_grad():
            num_nan_vecs = torch.sum(torch.isnan(volume_estimation.optic_axis[0, :]))
            replacement_vecs = torch.nn.functional.normalize(torch.rand(3, int(num_nan_vecs)), p=2, dim=0).to(device)
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
            recon_img_fig = plot_retardance_orientation(
                    ret_image_current.detach().cpu().numpy(),
                    azim_image_current.detach().cpu().numpy(),
                    'hsv'
                )
            my_recon_img_plot.pyplot(recon_img_fig)
            df_loss = pd.DataFrame(
                    {'Total loss': losses,
                    'Data fidelity': data_term_losses,
                    'Regularization': regularization_term_losses
                    })
            my_loss.line_chart(df_loss)

    matplotlib.pyplot.close()
    st.success("Done reconstructing! How does it look?", icon="✅")
    st.session_state['vol_est'] = volume_estimation
    st.session_state['ret_est'] = ret_image_current.detach().cpu().numpy()
    st.session_state['azim_est'] = azim_image_current.detach().cpu().numpy()

st.markdown("""---""")

if 'vol_est' in st.session_state:
    st.subheader("Analyze reconstructed volume")
    # st.session_state['vol_threshold'] = 0.1
    st.session_state['vol_threshold'] = st.slider(
        'Display optic axis vectors for the voxel where the birefringence is above the threshold:',
        min_value=0., max_value=1., value=0.5
        )
    st.session_state['fig_est'] = st.session_state['vol_est'].plot_lines_plotly(delta_n_ths=st.session_state['vol_threshold'])
    # starts off more pan'ed in than desired
    st.write(":blue[Scroll over image to zoom in and out.]")
    st.plotly_chart(st.session_state['fig_est'], use_container_width=True)


    st.subheader('Download results')
    # Save retardance and orientation images of estimated volume
    st.write("Downloaded estimated retardance and orientation images may not be \
             saved in a way that another program can read the images.")
    st.download_button('Download estimated retardance',
                       st.session_state['ret_est'].tobytes(),
                       mime="image/jpeg",
                       file_name='retardance_estimated.jpeg',
                       )
    st.download_button('Download estimated orientation',
                       st.session_state['azim_est'].tobytes(),
                       mime="image/jpeg",
                       file_name='azimuth_estimated.jpeg',
                       )
    # Save estimated volume to h5
    h5_file = io.BytesIO()
    st.download_button(
            'Download estimated volume as HDF5 file',
            st.session_state['vol_est'].save_as_file(h5_file),
            mime='application/x-hdf5',
            file_name='volume_estimated.h5',
        )

# st.subheader("Volume viewing")
# if st.button("Plot volume!"):
#     st.markdown("Scroll over image to zoom in and out.")
#     my_fig = st.session_state['my_volume'].plot_volume_plotly(
#                 optical_info,
#                 voxels_in=st.session_state['my_volume'].Delta_n,
#                 opacity=0.1
#                 )
#     st.plotly_chart(my_fig)

# if st.button("Plot volume with optic axis!"):
#     st.write("Scroll over image to zoom in and out.")
#     my_fig = st.session_state['my_volume'].plot_lines_plotly()
#     st.session_state['my_volume'].plot_volume_plotly(
#                 optical_info,
#                 voxels_in=st.session_state['my_volume'].Delta_n,
#                 opacity=0.1
#                 )
#     st.plotly_chart(my_fig, use_container_width=True)
