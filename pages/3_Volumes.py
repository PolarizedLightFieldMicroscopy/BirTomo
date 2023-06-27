import streamlit as st

st.set_page_config(
    page_title="Volumes",
    page_icon="",
    layout="wide",
)

st.title("Birefringent Volumes")

import h5py
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume
try:
    import torch
except:
    pass

optical_info = BirefringentVolume.get_optical_info_template()


st.header('Volume Creation')
columns = st.columns(2)
with columns[0]:
    volume_container = st.container() # set up a home for other volume selections to go
    shift_from_center = st.slider('Axial shift from center [voxels]', \
                                    min_value = -20, \
                                    max_value = 20,value = 0)
    volume_axial_offset = optical_info['volume_shape'][0] // 2 + shift_from_center # for center
    backend = BackEnds.NUMPY

with columns[1]:
    # optical_info['axial_voxel_size_um'] = st.slider('Axial voxel size [um]', min_value=.1, max_value=10., value = 1.0)
    optical_info['volume_shape'][0] = st.slider('Axial volume dimension', min_value=1, max_value=50, value=15)
    # y will follow x if x is changed. x will not follow y if y is changed
    optical_info['volume_shape'][1] = st.slider('Y volume dimension', min_value=1, max_value=100, value=51)
    optical_info['volume_shape'][2] = st.slider('Z volume dimension', min_value=1, max_value=100, value=optical_info['volume_shape'][1])

def key_investigator(key_home, my_str='', prefix='- '):
    if hasattr(key_home, 'keys'):
        for my_key in key_home.keys():
            my_str = my_str + prefix + my_key +'\n'
            my_str = key_investigator(key_home[my_key], my_str, '\t'+prefix)
    return my_str

with volume_container: # now that we know backend and shift, we can fill in the rest of the volume params
    how_get_vol = st.radio("Volume can be created or uploaded as an h5 file", \
                            ['h5 upload', 'Create a new volume'], index=1)
    if how_get_vol == 'h5 upload':
        h5file = st.file_uploader("Upload Volume h5 Here", type=['h5'])
        if h5file is not None:
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

            st.session_state['my_volume'] = BirefringentVolume.init_from_file(h5file, backend=backend, \
                                                    optical_info=optical_info)

    else:
        volume_type = st.selectbox('Volume type',['ellipsoid','shell','2ellipsoids','single_voxel'], 2)
        st.session_state['my_volume'] = BirefringentVolume.create_dummy_volume(backend=backend, optical_info=optical_info, \
                                    vol_type=volume_type, volume_axial_offset=volume_axial_offset)

st.subheader("Volume viewing")

if st.button("Plot volume!"):
    st.write("Scroll over image to zoom in and out.")
    my_fig = st.session_state['my_volume'].plot_volume_plotly(optical_info, 
                            voxels_in=st.session_state['my_volume'].Delta_n, opacity=0.1)
    st.plotly_chart(my_fig, use_container_width=True)

if st.button("Plot volume with optic axis!"):
    st.write("Scroll over image to zoom in and out.")
    my_fig = st.session_state['my_volume'].plot_lines_plotly()
    st.session_state['my_volume'].plot_volume_plotly(optical_info, 
                            voxels_in=st.session_state['my_volume'].Delta_n, opacity=0.1)
    st.plotly_chart(my_fig, use_container_width=True)
