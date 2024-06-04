"""This script uses numpy/pytorch back-end to:
    - Compute the ray geometry depending on the light field microscope and volume configuration.
    - Create a volume with different birefringent shapes.
    - Traverse the rays through the volume.
    - Compute the retardance and azimuth for every ray.
    - Generate 2D images.
"""

import torch
import streamlit as st
import h5py
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.simulations import ForwardModel
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume
from VolumeRaytraceLFM.volumes import volume_args
from VolumeRaytraceLFM.setup_parameters import setup_optical_parameters
from VolumeRaytraceLFM.visualization.plotting_volume import visualize_volume

# Select backend method
BACKEND = BackEnds.PYTORCH
# backend = BackEnds.NUMPY

if BACKEND == BackEnds.PYTORCH:
    import torch

    torch.set_grad_enabled(False)
else:
    device = "cpu"

st.set_page_config(
    page_title="Forward",
    page_icon="",
    layout="wide",
)

st.title("Forward Simulation")

st.header("Choose our parameters")

# Get optical parameters template
st.session_state["optical_info"] = BirefringentVolume.get_optical_info_template()
optical_info = st.session_state["optical_info"]
# st.write(optical_info)

column1, column2 = st.columns(2)
with column1:
    ############ Optical Params #################
    # Alter some of the optical parameters
    st.subheader("Optical")
    optical_info["n_micro_lenses"] = st.slider(
        "Number of microlenses", min_value=1, max_value=51, value=5
    )
    optical_info["pixels_per_ml"] = st.slider(
        "Pixels per microlens", min_value=1, max_value=33, value=17, step=2
    )
    optical_info["n_voxels_per_ml"] = st.slider(
        "Number of voxels per microlens (supersampling)",
        min_value=1,
        max_value=7,
        value=1,
    )
    # optical_info['axial_voxel_size_um'] = st.slider('Axial voxel size [um]',
    #                                                 min_value=.1, max_value=10., value = 1.0)
    optical_info["M_obj"] = st.slider(
        "Magnification", min_value=10, max_value=100, value=60, step=10
    )
    optical_info["na_obj"] = st.slider(
        "NA of objective", min_value=0.5, max_value=1.75, value=1.2
    )
    optical_info["wavelength"] = st.slider(
        "Wavelength of the light", min_value=0.380, max_value=0.770, value=0.550
    )
    optical_info["camera_pix_pitch"] = st.slider(
        "Camera pixel size [um]", min_value=3.0, max_value=12.0, value=6.5, step=0.5
    )
    medium_option = st.radio(
        "Refractive index of the medium", ["Water: n = 1.35", "Oil: n = 1.52"], 0
    )
    # if medium_option == 'Water: n = 1.35':
    optical_info["n_medium"] = float(medium_option[-4:])

    # st.write("Computed voxel size [um]:", optical_info['voxel_size_um'])

    # microlens size is 6.5*17 = 110.5 (then divided by mag 60)

    st.subheader("Other")
    backend_choice = st.radio("Backend", ["numpy", "torch"])


def key_investigator(key_home, my_str="", prefix="- "):
    if hasattr(key_home, "keys"):
        for my_key in key_home.keys():
            my_str = my_str + prefix + my_key + "\n"
            my_str = key_investigator(key_home[my_key], my_str, "\t" + prefix)
    return my_str


with column2:
    ############ Volume #################
    st.subheader("Volume")
    # set up a home for other volume selections to go
    volume_container = st.container()

    if backend_choice == "torch":
        backend = BackEnds.PYTORCH
        torch.set_grad_enabled(False)
    else:
        backend = BackEnds.NUMPY

    # Now that we know backend and shift, we can fill in the rest of the volume params
    with volume_container:
        how_get_vol = st.radio(
            "Volume can be created or uploaded as an h5 file",
            ["h5 upload", "Create a new volume"],
            index=1,
        )
        if how_get_vol == "h5 upload":
            h5file = st.file_uploader("Upload Volume h5 Here", type=["h5"])
            if h5file is not None:
                with h5py.File(h5file) as file:
                    try:
                        vol_shape = file["optical_info"]["volume_shape"][()]
                    except KeyError:
                        st.error("This file does specify the volume shape.")
                    except Exception as e:
                        st.error(e)
                vol_shape_default = [int(v) for v in vol_shape]
                optical_info["volume_shape"] = vol_shape_default
                st.markdown(
                    "Using a cube volume shape with the dimension of the"
                    + f"loaded volume: {vol_shape_default}."
                )

                display_h5 = st.checkbox("Display h5 file contents")
                if display_h5:
                    with h5py.File(h5file) as file:
                        st.markdown("**File Structure:**\n" + key_investigator(file))
                        try:
                            st.markdown(
                                "**Description:** "
                                + str(file["optical_info"]["description"][()])[2:-1]
                            )
                        except KeyError:
                            st.error("This file does not have a description.")
                        except Exception as e:
                            st.error(e)
                        try:
                            vol_shape = file["optical_info"]["volume_shape"][()]
                            # optical_info['volume_shape'] = vol_shape
                            st.markdown(f"**Volume Shape:** {vol_shape}")
                        except KeyError:
                            st.error("This file does specify the volume shape.")
                        except Exception as e:
                            st.error(e)
                        try:
                            voxel_size = file["optical_info"]["voxel_size_um"][()]
                            st.markdown(f"**Voxel Size (um):** {voxel_size}")
                        except KeyError:
                            st.error(
                                "This file does specify the voxel size. Voxels are likely to be cubes."
                            )
                        except Exception as e:
                            st.error(e)
        else:
            volume_type = st.selectbox(
                "Volume type", ["ellipsoid", "shell", "2ellipsoids", "single_voxel"], 1
            )
            optical_info["volume_shape"][0] = st.slider(
                "Axial volume dimension", min_value=1, max_value=50, value=15
            )
            # y will follow x if x is changed. x will not follow y if y is changed
            optical_info["volume_shape"][1] = st.slider(
                "Y volume dimension", min_value=1, max_value=100, value=51
            )
            optical_info["volume_shape"][2] = st.slider(
                "Z volume dimension",
                min_value=1,
                max_value=100,
                value=optical_info["volume_shape"][1],
            )
            shift_from_center = st.slider(
                "Axial shift from center [voxels]",
                min_value=-int(optical_info["volume_shape"][0] / 2),
                max_value=int(optical_info["volume_shape"][0] / 2),
                value=0,
            )
            # for center
            volume_axial_offset = (
                optical_info["volume_shape"][0] // 2 + shift_from_center
            )
    # Create the volume based on the selections.
    with volume_container:
        if how_get_vol == "h5 upload":
            if h5file is not None:
                st.session_state["my_volume"] = BirefringentVolume.init_from_file(
                    h5file, backend=backend, optical_info=optical_info
                )
        else:
            st.session_state["my_volume"] = BirefringentVolume.create_dummy_volume(
                backend=backend,
                optical_info=optical_info,
                vol_type=volume_type,
                volume_axial_offset=volume_axial_offset,
            )

if __name__ == "__main__":
    optical_info = setup_optical_parameters("config/optical_config1.json")
    optical_system = {"optical_info": optical_info}
    simulator = ForwardModel(optical_system, backend=BACKEND)
    volume_GT = BirefringentVolume(
        backend=BACKEND,
        optical_info=optical_info,
        volume_creation_args=volume_args.voxel_args,
    )
    visualize_volume(volume_GT, optical_info)
    simulator.forward_model(volume_GT)
    simulator.view_images()
