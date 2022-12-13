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
from VolumeRaytraceLFM.birefringence_implementations import OpticalElement, BirefringentRaytraceLFM, JonesMatrixGenerators


backend = BackEnds.NUMPY

# Get optical parameters template
optical_info = OpticalElement.get_optical_info_template()
# Alter some of the optical parameters
optical_info['volume_shape'] = [15, 51, 51]
optical_info['axial_voxel_size_um'] = 1.0
optical_info['pixels_per_ml'] = 17
optical_info['n_micro_lenses'] = 5
optical_info['n_voxels_per_ml'] = 1

st.slider('Number of microlenses', min_value=1, max_value=25, value=5)
st.slider('Pixels per microlens', min_value=1, max_value=21, value=17)



########################################################################



# Tutorial website: https://www.datacamp.com/tutorial/streamlit

######################### Working example ###############################
import streamlit as st
st.title("this is the app title")
st.code("x=2021")
st.latex(r''' a+a r^1+a r^2+a r^3 ''')
st.checkbox('yes')
st.button('Click')
st.multiselect('Choose a planet',['Jupiter', 'Mars', 'neptune'])
st.slider('Pick a number', 0,50)
st.text_area('Description')
st.file_uploader('Upload a photo')
st.color_picker('Choose your favorite color')

####################################################################
