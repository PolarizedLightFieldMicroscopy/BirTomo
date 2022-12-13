'''User interface for forward projection using the Streamlit package'''
# Enter the following into the command line the refresh browser to see updates:
# pip install streamlit
# streamlit run forward_streamlit.py


######################################################################
# Content below is extracted from main_forward_projection.py
import streamlit as st
st.title("Forward projection")

import time         # to measure ray tracing time
import numpy as np  # to convert radians to degrees for plots
import matplotlib.pyplot as plt
from plotting_tools import plot_birefringence_lines, plot_birefringence_colorized
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume, BirefringentRaytraceLFM, JonesMatrixGenerators


######################################################################
#Collect Parameters form the streamlit page

# Get optical parameters template
optical_info = BirefringentVolume.get_optical_info_template()
# Alter some of the optical parameters
optical_info['volume_shape'] = [15, 51, 51]
optical_info['axial_voxel_size_um'] = 1.0
optical_info['pixels_per_ml'] = 17
optical_info['n_micro_lenses'] = 5
optical_info['n_voxels_per_ml'] = 1


st.header('Optical Properties')
optical_info['n_micro_lenses'] = st.slider('Number of microlenses', min_value=1, max_value=25, value=5)
optical_info['pixels_per_ml'] = st.slider('Pixels per microlens', min_value=1, max_value=33, value=17, step=2)
optical_info['axial_voxel_size_um'] = st.slider('Axial voxel size [um]', min_value=.1, max_value=10., value = 1.0)
optical_info['n_voxels_per_ml'] = st.slider('Number of voxels per microlens', min_value=1, max_value=5, value=1)
optical_info['volume_shape'][0] = st.slider('Axial volume dimension', min_value=1, max_value=50, value=15)
# y will follow x if x is changed. x will not follow y if y is changed
optical_info['volume_shape'][1] = st.slider('X volume dimension', min_value=1, max_value=100, value=51)
optical_info['volume_shape'][2] = st.slider('Y volume dimension', min_value=1, max_value=100, value=optical_info['volume_shape'][1])

st.header('Sample Properties')
volume_type = st.selectbox('Volume type',['ellipsoid','shell','2ellipsoids','single_voxel'],1)
shift_from_center = st.slider('Axial shift from center[chunks]', min_value = -int(optical_info['volume_shape'][0]/2), max_value = int(optical_info['volume_shape'][0]/2),value = 0)

st.header('Other Parameters')
backend_choice = st.radio('Backend',['numpy','torch'])


#display the current values on a sidebar
st.sidebar.title('Current Properties')
st.sidebar.header('Optical Properties')
st.sidebar.text('n_micro_lenses = %d' % optical_info['n_micro_lenses'])
st.sidebar.text('axial_voxel_size_um = %f' % optical_info['axial_voxel_size_um'])
st.sidebar.text('pixels_per_ml = %d' % optical_info['pixels_per_ml'])
st.sidebar.text('n_voxels_per_ml = %d' % optical_info['n_voxels_per_ml'])
st.sidebar.text('volume_shape = [%d,%d,%d]' % (optical_info['volume_shape'][0],optical_info['volume_shape'][1],optical_info['volume_shape'][2]))

st.sidebar.header('Sample Properties')
st.sidebar.text('volume_type = %s' % volume_type)
st.sidebar.text('shift_from_center = %d' % shift_from_center)

st.sidebar.header('Other Parameters')
st.sidebar.text('backend = %s' % backend_choice)

########################################################################
#Now we calculate based on the selected inputs
if st.button('Calculate!'):

    # final conversions of the streamlit inputs to things 
    # the forward model understands
    if backend_choice == 'torch':
        backend = BackEnds.PYTORCH
        from waveblocks.utils.misc_utils import *
    else:
        backend = BackEnds.NUMPY

    volume_axial_offset = optical_info['volume_shape'][0] // 2 + shift_from_center # for center

    # Now we begin the calculation
    # Changed all print statements to sp.text

    # Create a Birefringent Raytracer
    rays = BirefringentRaytraceLFM(backend=backend, optical_info=optical_info)

    # Compute the rays and use the Siddon's algorithm to compute the intersections with voxels.
    # If a filepath is passed as argument, the object with all its calculations
    #   get stored/loaded from a file
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


    # Create a volume
    my_volume = BirefringentVolume.create_dummy_volume(backend=backend, optical_info=optical_info, vol_type=volume_type, volume_axial_offset=volume_axial_offset)

    # Plot the volume
    # my_volume.plot_volume_plotly(optical_info, voxels_in=my_volume.Delta_n, opacity=0.1)



    startTime = time.time()
    ret_image, azim_image = rays.ray_trace_through_volume(my_volume)
    executionTime = (time.time() - startTime)
    st.text(f'Execution time in seconds with backend {backend}: ' + str(executionTime))

    if backend == BackEnds.PYTORCH:
        ret_image, azim_image = ret_image.numpy(), azim_image.numpy()

    # Plot
    colormap = 'viridis'
    plt.rcParams['image.origin'] = 'lower'
    fig = plt.figure(figsize=(12,2.5))
    plt.subplot(1,3,1)
    plt.imshow(ret_image,cmap=colormap)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(F'Retardance {backend}')
    plt.subplot(1,3,2)
    plt.imshow(np.rad2deg(azim_image), cmap=colormap)
    plt.colorbar(fraction=0.046, pad=0.04)
    azimuth_plot_type = 'hsv'
    plt.title('Azimuth')
    ax = plt.subplot(1,3,3)
    if azimuth_plot_type == 'lines':
        im = plot_birefringence_lines(ret_image, azim_image,cmap=colormap, line_color='white', ax=ax)
    else:
        plot_birefringence_colorized(ret_image, azim_image)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('Ret+Azim')

    st.pyplot(fig)
    # plt.savefig(f'Forward_projection_off_axis_thickness03_deltan-01_{volume_type}_axial_offset_{volume_axial_offset}.pdf')
    # plt.pause(0.2)









# Tutorial website: https://www.datacamp.com/tutorial/streamlit

######################### Working example ###############################
# import streamlit as st
# st.title("this is the app title")
# st.code("x=2021")
# st.latex(r''' a+a r^1+a r^2+a r^3 ''')
# st.checkbox('yes')
# st.button('Click')
# st.multiselect('Choose a planet',['Jupiter', 'Mars', 'neptune'])
# st.slider('Pick a number', 0,50)
# st.text_area('Description')
# st.file_uploader('Upload a photo')
# st.color_picker('Choose your favorite color')

####################################################################
