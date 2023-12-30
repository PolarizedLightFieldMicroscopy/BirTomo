"""
This module provides utility functions for interfacing and extracting information from HDF5 files.
It includes functions for investigating the keys in an HDF5 file, extracting microscope parameter 
names and values, converting data between dataframes and dictionaries, and displaying HDF5 metadata 
within a Streamlit application. Additionally, it defines a function to perform forward propagation 
of rays through a volume using ray tracing.
"""
import time
import streamlit as st
import pandas as pd
import h5py
try:
    import torch
except:
    pass
from VolumeRaytraceLFM.abstract_classes import BackEnds


def key_investigator(key_home, my_str='', prefix='- '):
    """Recursively investigates the keys in an HDF5 file and constructs a multiline string
    representation of the file's hierarchical structure."""
    if hasattr(key_home, 'keys'):
        for my_key in key_home.keys():
            my_str = my_str + prefix + my_key + '\n'
            my_str = key_investigator(key_home[my_key], my_str, '\t'+prefix)
    return my_str


def get_microscope_param_names():
    """Returns a tuple of lists containing parameter keys and their descriptions related to 
    microscope settings."""
    microscope_params_keys = [
        'n_micro_lenses',
        'wavelength',
        'M_obj',
        'n_medium',
        'na_obj',
        'pixels_per_ml',
        'camera_pix_pitch'
    ]
    microscope_params_descriptions = [
        'Number of microlenses',
        'Wavelength (microns)',
        'Magnification',
        'Refractive index of the medium',
        'NA of objective',
        'Pixels per microlens',
        'Camera pixel pitch (microns)'
    ]
    return microscope_params_keys, microscope_params_descriptions


def extract_scalar_params(dict_optical):
    """Extracts scalar parameters of a microscope from a dictionary containing optical 
    information and returns a pandas DataFrame with the parameters and their values."""
    keys, descriptions = get_microscope_param_names()
    microscope_vals = [dict_optical[k] for k in keys]
    df_microscope = pd.DataFrame(
        list(zip(descriptions, microscope_vals)),
        columns=['Parameter', 'Value']
    )
    return df_microscope


def dataframe_to_dict(df):
    """Converts microscope parameters from a pandas DataFrame back into a dictionary
    using a predefined set of parameter names."""
    keys, descriptions = get_microscope_param_names()
    key_to_description_dict = dict(zip(keys, descriptions))
    values_from_df = {}
    for key in keys:
        description = key_to_description_dict[key]
        value = df[df['Parameter'] == description]['Value'].iloc[0]
        if key in ['n_micro_lenses', 'pixels_per_ml']:
            value = int(value)
        values_from_df[key] = value
    return values_from_df

# def check_h5_format(h5file):


def get_vol_shape_from_h5(h5file):
    """Retrieves the volume shape from an HDF5 file and returns it as a list of integers."""
    with h5py.File(h5file) as file:
        try:
            vol_shape = file['optical_info']['volume_shape'][()]
        except KeyError:
            st.error('This file does specify the volume shape.')
        except Exception as e:
            st.error(e)
    vol_shape_default = [int(v) for v in vol_shape]
    return vol_shape_default


def display_h5_metadata(h5file):
    """Displays the metadata of an HDF5 file, including file structure, description, volume shape,
    and voxel size using Streamlit components."""
    with h5py.File(h5file) as file:
        st.markdown('**File Structure:**\n' + key_investigator(file))
        try:
            st.markdown('**Description:** ' +
                        str(file['optical_info']['description'][()])[2:-1])
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
            st.write(
                'This file does not specify the voxel size. We are assuming voxels are cubes.')
        except Exception as e:
            st.error(e)
    return


def forward_propagate(rays, volume):
    """Performs the forward propagation of rays through a volume and returns the resulting
    retardance and azimuth images, along with the execution time of the ray tracing."""
    try:
        # rays = BirefringentRaytraceLFM(backend=backend, optical_info=optical_info)
        start_time = time.time()
        rays.compute_rays_geometry()
        execution_time = time.time() - start_time
        # st.text('Ray-tracing time in seconds: ' + str(execution_time))

        # Move ray tracer to GPU
        if rays.backend == BackEnds.PYTORCH:
            # Disable gradients
            torch.set_grad_enabled(False)
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            # st.text(f'Using computing device: {device}')
            rays = rays.to(device)
            volume = volume.to(device)

        start_time = time.time()
        [ret_image, azim_image] = rays.ray_trace_through_volume(volume)
        execution_time = time.time() - start_time
        # st.text(f'Execution time in seconds with backend {backend}: ' + str(execution_time))

        if rays.backend == BackEnds.PYTORCH:
            ret_image, azim_image = ret_image.cpu().numpy(), azim_image.cpu().numpy()

        # st.session_state['ret_image'] = ret_image
        # st.session_state['azim_image'] = azim_image

    except KeyError as e:
        print(e)
        # st.error('Please chose a volume first!')
    return [ret_image, azim_image], execution_time
