'''User interface for forward projection using the Streamlit package'''
import time         # to measure ray tracing time
import h5py         # for reading h5 volume files
import streamlit as st
from plotting_tools import plot_retardance_orientation
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import (
    BirefringentVolume, BirefringentRaytraceLFM
)
try:
    import torch
except ImportError:
    pass

st.set_page_config(
    page_title="Forward",
    page_icon="",
    layout="wide",
)

st.title("Forward Projection")

st.header("Choose our parameters")

# Get optical parameters template
st.session_state['optical_info'] = BirefringentVolume.get_optical_info_template()
optical_info = st.session_state['optical_info']
# st.write(optical_info)

columns = st.columns(2)
# First Column
with columns[0]:
############ Optical Params #################
    # Alter some of the optical parameters
    st.subheader('Optical')
    optical_info['n_micro_lenses'] = st.slider('Number of microlenses',
                                               min_value=1, max_value=51, value=5)
    optical_info['pixels_per_ml'] = st.slider('Pixels per microlens',
                                              min_value=1, max_value=33, value=17, step=2)
    optical_info['n_voxels_per_ml'] = st.slider('Number of voxels per microlens (supersampling)',
                                                min_value=1, max_value=7, value=1)
    # optical_info['axial_voxel_size_um'] = st.slider('Axial voxel size [um]',
    #                                                 min_value=.1, max_value=10., value = 1.0)
    optical_info['M_obj'] = st.slider('Magnification',
                                      min_value=10, max_value=100, value=60, step=10)
    optical_info['na_obj'] = st.slider('NA of objective',
                                       min_value=0.5, max_value=1.75, value=1.2)
    optical_info['wavelength'] = st.slider('Wavelength of the light',
                                           min_value=0.380, max_value=0.770, value=0.550)
    optical_info['camera_pix_pitch'] = st.slider('Camera pixel size [um]',
                                                 min_value=3.0, max_value=12.0, value=6.5, step=0.5)
    medium_option = st.radio('Refractive index of the medium',
                             ['Water: n = 1.35', 'Oil: n = 1.52'], 0)
    # if medium_option == 'Water: n = 1.35':
    optical_info['n_medium'] = float(medium_option[-4:-1])

    # st.write("Computed voxel size [um]:", optical_info['voxel_size_um'])

    # microlens size is 6.5*17 = 110.5 (then divided by mag 60)
############ Other #################
    st.subheader('Other')
    backend_choice = st.radio('Backend', ['numpy', 'torch'])

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

    if backend_choice == 'torch':
        backend = BackEnds.PYTORCH
        torch.set_grad_enabled(False)
    else:
        backend = BackEnds.NUMPY

    # Now that we know backend and shift, we can fill in the rest of the volume params
    with volume_container:
        how_get_vol = st.radio("Volume can be created or uploaded as an h5 file",
                                ['h5 upload', 'Create a new volume'], index=1)
        if how_get_vol == 'h5 upload':
            h5file = st.file_uploader("Upload Volume h5 Here", type=['h5'])
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
                            + f"loaded volume: {vol_shape_default}.")

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
        else:
            volume_type = st.selectbox('Volume type',
                                       ['ellipsoid','shell','2ellipsoids','single_voxel'], 1)
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
                st.session_state['my_volume'] = BirefringentVolume.init_from_file(
                                                        h5file,
                                                        backend=backend,
                                                        optical_info=optical_info
                                                        )
        else:
            st.session_state['my_volume'] = BirefringentVolume.create_dummy_volume(
                                                backend=backend,
                                                optical_info=optical_info,
                                                vol_type=volume_type,
                                                volume_axial_offset=volume_axial_offset
                                                )

st.subheader("Volume viewing")
if st.button("Plot volume!"):
    st.markdown("Scroll over image to zoom in and out.")
    my_fig = st.session_state['my_volume'].plot_volume_plotly(
                optical_info,
                voxels_in=st.session_state['my_volume'].Delta_n,
                opacity=0.1
                )
    st.plotly_chart(my_fig)

if st.button("Plot volume with optic axis!"):
    st.write("Scroll over image to zoom in and out.")
    my_fig = st.session_state['my_volume'].plot_lines_plotly()
    st.session_state['my_volume'].plot_volume_plotly(
                optical_info,
                voxels_in=st.session_state['my_volume'].Delta_n,
                opacity=0.1
                )
    st.plotly_chart(my_fig, use_container_width=True)
######################################################################
def forward_propagate():
    '''Ray trace through the volume'''
    try:
        rays = BirefringentRaytraceLFM(backend=backend, optical_info=optical_info)
        start_time = time.time()
        rays.compute_rays_geometry()
        execution_time = (time.time() - start_time)
        st.text('Ray-tracing time in seconds: ' + str(execution_time))

        # Move ray tracer to GPU
        if backend == BackEnds.PYTORCH:
            # Disable gradients
            torch.set_grad_enabled(False)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            st.text(f'Using computing device: {device}')
            rays = rays.to(device)

        start_time = time.time()
        [ret_image, azim_image] = rays.ray_trace_through_volume(st.session_state['my_volume'])
        execution_time = (time.time() - start_time)
        st.text(f'Execution time in seconds with backend {backend}: ' + str(execution_time))

        if backend == BackEnds.PYTORCH:
            ret_image, azim_image = ret_image.numpy(), azim_image.numpy()

        st.session_state['ret_image'] = ret_image
        st.session_state['azim_image'] = azim_image

        st.success("Geometric ray tracing was successful!", icon="✅")
    except KeyError:
        st.error('Please chose a volume first!')
    return None
########################################################################
# Now we calculate based on the selected inputs
st.header("Retardance and orientation images")

# st.write(st.session_state)
if st.button('Calculate!'):
    forward_propagate()

if "ret_image" in st.session_state:
    # Plot with streamlit
    azimuth_plot_type = st.selectbox('Azmiuth Plot Type', ['lines', 'hsv'], index = 1)
    output_ret_image = st.session_state['ret_image']
    output_azim_image = st.session_state['azim_image']
    my_fig = plot_retardance_orientation(output_ret_image, output_azim_image, azimuth_plot_type)
    st.pyplot(my_fig)

    st.success("Images were successfully created!", icon="✅")
