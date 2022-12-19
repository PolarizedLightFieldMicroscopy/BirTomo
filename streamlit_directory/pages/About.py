import streamlit as st

st.write("Test words on the About.py page")

from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume, BirefringentRaytraceLFM

st.header('Current (actually just the default) Properties')
st.subheader('Optical Properties')
optical_info = BirefringentVolume.get_optical_info_template()
st.text('n_micro_lenses = %d' % optical_info['n_micro_lenses'])
st.text('pixels_per_ml = %d' % optical_info['pixels_per_ml'])
st.text('n_voxels_per_ml = %d' % optical_info['n_voxels_per_ml'])
st.text('volume_shape = [%d, %d, %d]' % (optical_info['volume_shape'][0], 
                                        optical_info['volume_shape'][1], 
                                        optical_info['volume_shape'][2]))

# st.sidebar.header('Sample Properties')
# st.sidebar.text('volume_type = %s' % volume_type)
# st.sidebar.text('shift_from_center = %d' % shift_from_center)

# st.sidebar.header('Other Parameters')
# st.sidebar.text('backend = %s' % backend_choice)