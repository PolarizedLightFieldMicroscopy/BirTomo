'''User interface using the Streamlit package'''
# Enter the following into the command line the refresh browser to see updates:
# pip install streamlit
# streamlit run User_Interface.py


######################################################################
import streamlit as st

st.set_page_config(
    page_title="Hello Rudolf",
    page_icon="👋",
    layout="wide", # centered
)

st.title("Polarization Simulations")

st.markdown("*Click around to explore*")

st.title("Forward Projection")

import time         # to measure ray tracing time
import numpy as np  # to convert radians to degrees for plots
import matplotlib.pyplot as plt
from plotting_tools import plot_birefringence_lines, plot_birefringence_colorized
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume, BirefringentRaytraceLFM
try:
    import torch
except:
    pass

st.header("Choose our parameters")

st.subheader("Parameters currently have the following values:")
optical_info = BirefringentVolume.get_optical_info_template()
st.write(optical_info)

columns = st.columns(2)
# First Column
with columns[0]:
############ Optical Params #################
    # Get optical parameters template
    optical_info = BirefringentVolume.get_optical_info_template()
    # Alter some of the optical parameters
    st.subheader('Optical')
    optical_info['n_micro_lenses'] = st.slider('Number of microlenses', min_value=1, max_value=25, value=5)
    optical_info['pixels_per_ml'] = st.slider('Pixels per microlens', min_value=1, max_value=33, value=17, step=2)
    # optical_info['axial_voxel_size_um'] = st.slider('Axial voxel size [um]', min_value=.1, max_value=10., value = 1.0)
    optical_info['n_voxels_per_ml'] = st.slider('Number of voxels per microlens (supersampling)', min_value=1, max_value=3, value=1)
    optical_info['M_obj'] = st.slider('Magnification', min_value=1, max_value=100, value=60, step=10)
    optical_info['na_obj'] = st.slider('NA of objective', min_value=0.5, max_value=1.75, value=1.2)

    # st.write("Computed voxel size [um]:", optical_info['voxel_size_um'])

    # microlens size is 6.5*17 = 110.5 (then divided by mag 60)
############ Other #################
    st.subheader('Other')
    backend_choice = st.radio('Backend', ['numpy', 'torch'])

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
    if backend_choice == 'torch':
        backend = BackEnds.PYTORCH
    else:
        backend = BackEnds.NUMPY

    with volume_container: # now that we know backend and shift, we can fill in the rest of the volume params
        how_get_vol = st.radio("Volume can be created or uploaded as an h5 file", \
                                ['h5 upload', 'Create a new volume'], index=1)
        if how_get_vol == 'h5 upload':
            h5file = st.file_uploader("Upload Volume h5 Here", type=['h5'])
            if h5file is not None:
                st.session_state['my_volume'] = BirefringentVolume.init_from_file(h5file, backend=backend, \
                                                        optical_info=optical_info)
        else:
            volume_type = st.selectbox('Volume type',['ellipsoid','shell','2ellipsoids','single_voxel'],1)
            st.session_state['my_volume'] = BirefringentVolume.create_dummy_volume(backend=backend, optical_info=optical_info, \
                                        vol_type=volume_type, volume_axial_offset=volume_axial_offset)

st.subheader("Volume viewing")
if st.button("Plot volume!"):
    st.write("Scroll over image to zoom in and out.")
    my_fig = st.session_state['my_volume'].plot_volume_plotly_streamlit(optical_info, 
                            voxels_in=st.session_state['my_volume'].Delta_n, opacity=0.1)
    st.plotly_chart(my_fig)
######################################################################
# Create a function for doing the forward propagation math
def forwardPropagate():
    try:
        rays = BirefringentRaytraceLFM(backend=backend, optical_info=optical_info)
        startTime = time.time()
        rays.compute_rays_geometry()
        executionTime = (time.time() - startTime)
        st.text('Ray-tracing time in seconds: ' + str(executionTime))

        # Move ray tracer to GPU
        if backend == BackEnds.PYTORCH:
            # Disable gradients
            torch.set_grad_enabled(False)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            st.text(f'Using computing device: {device}')
            rays = rays.to(device)

        startTime = time.time()
        ret_image, azim_image = rays.ray_trace_through_volume(st.session_state['my_volume'])
        executionTime = (time.time() - startTime)
        st.text(f'Execution time in seconds with backend {backend}: ' + str(executionTime))

        if backend == BackEnds.PYTORCH:
            ret_image, azim_image = ret_image.numpy(), azim_image.numpy()

        st.session_state['ret_image'] = ret_image
        st.session_state['azim_image'] = azim_image

        st.success("Geometric ray tracing was successful!", icon="✅")
    except KeyError:
        st.error('Please chose a volume first!')

    return

########################################################################
# Now we calculate based on the selected inputs
st.header("Retardance and azimuth images")

# st.write(st.session_state)
if st.button('Calculate!'):
    forwardPropagate()


if "ret_image" in st.session_state:
    # Plot with streamlit
    azimuth_plot_type = st.selectbox('Azmiuth Plot Type', ['lines','hsv'], index = 1)
    colormap = 'viridis'
    plt.rcParams['image.origin'] = 'lower'
    fig = plt.figure(figsize=(12,2.5))
    plt.subplot(1,3,1)
    plt.imshow(st.session_state['ret_image'], cmap=colormap)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(F'Retardance {backend}')
    plt.subplot(1,3,2)
    plt.imshow(np.rad2deg(st.session_state['azim_image']), cmap=colormap)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('Azimuth')
    ax = plt.subplot(1,3,3)
    if azimuth_plot_type == 'lines':
        im = plot_birefringence_lines(st.session_state['ret_image'], st.session_state['azim_image'],cmap=colormap, line_color='white', ax=ax)
    else:
        plot_birefringence_colorized(st.session_state['ret_image'], st.session_state['azim_image'])
        plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('Ret+Azim')
    st.pyplot(fig)
    # plt.savefig(f'Forward_projection_off_axis_thickness03_deltan-01_{volume_type}_axial_offset_{volume_axial_offset}.pdf')
    # plt.pause(0.2)


    st.success("Images were successfully created!", icon="✅")