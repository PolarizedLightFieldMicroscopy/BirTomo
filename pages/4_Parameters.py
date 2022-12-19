import streamlit as st

from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume


st.subheader("Parameters currently have the following values:")
# st.subheader('Optical Properties')

try:
    st.write(st.session_state['optical_info'])
except:
    st.write(BirefringentVolume.get_optical_info_template())

# st.sidebar.header('Sample Properties')
# st.sidebar.text('volume_type = %s' % volume_type)
# st.sidebar.text('shift_from_center = %d' % shift_from_center)

# st.sidebar.header('Other Parameters')
# st.sidebar.text('backend = %s' % backend_choice)