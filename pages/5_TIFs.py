import streamlit as st
import tifffile

st.title("Forward projection TIFs")
uploaded_file = st.file_uploader("Upload a TIF image generated from a forward projection",
                                 type=["tif", "tiff"])

# Open the TIFF file
if uploaded_file is not None:
    my_tif = tifffile.TiffFile(uploaded_file)
    image_data = my_tif.asarray()
    metadata = my_tif.pages[0].tags

# Display the image
st.header('Image')
if uploaded_file:
    st.image(image_data, clamp=True, use_column_width=True)

# Display the metadata
st.header('Metadata')
if uploaded_file:
    if False:
        for tag in metadata.values():
            st.write(f"{tag.name}: {tag.value}")
        # st.write(image_description_dict.values())
        # for key, value in image_description_dict.items():
        #     st.write(f"{key}: {value}")
        for key, value in image_description_dict.items():
            if isinstance(value, dict):
                st.write(f"\n{key}:")
                for subkey, subvalue in value.items():
                    st.write(f"{subkey}: {subvalue}")
            else:
                st.write(f"\n{key}: {value}")

    image_description = metadata['ImageDescription'].value
    image_description = image_description.replace(
        'true', 'True').replace('false', 'False')
    image_description_dict = eval(image_description)
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Description", "Optical info", "Parameters", "All metadata"])
    with tab1:
        st.header("Description")
        subset_dict = {}
        for key, value in image_description_dict.items():
            # if key in ["Description", "Comments"]:
            if key not in ["Optical info", "Streamlit parameters"]:
                subset_dict[key] = value
        st.write(subset_dict)
        # for key in ["Description", "Comments"]:
        #     st.write(f"{key}: {image_description_dict[key]}")
        # st.write(image_description_dict)
    with tab2:
        st.header("Optical info")
        optical_dict = image_description_dict["Optical info"]
        st.write(optical_dict)
        # for key in optical_dict:
        #     st.write(f"{key}: {optical_dict[key]}")
    with tab3:
        st.header("Streamlit parameters")
        st_param_dict = image_description_dict["Streamlit parameters"]
        st.write(st_param_dict)
    with tab4:
        st.header("All metadata")
        tag_dict = {}
        for tag in metadata:
            value = tag.value
            # Convert the tag value to a string if it's a byte string
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            tag_dict[tag.name] = value
        st.write(tag_dict)
