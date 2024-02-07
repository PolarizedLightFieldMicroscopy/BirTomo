from datetime import datetime
import os
try:
    import tifffile
except ImportError:
    print("The package tifffile was unable to be imported, " +
          "so images will not be able to be saved as TIFs.")


def save_as_tif(file_path, data, metadata):
    '''Save a retardance or orientation image as a TIF file.
    Parameters:
        file_path (str): file path including the file name with extension '.tiff' 
        data (np.array): retardance or orientation array
        metadata (dict): contains the dictionary metadata['Optical info']
    Returns:
    '''
    # Removing metadata if irrelevant or values are np.arrays
    keys_to_delete = ['polarizer', 'analyzer', 'polarizer_swing']
    for key in keys_to_delete:
        if key in metadata['Optical info']:
            del metadata['Optical info'][key]
    with tifffile.TiffWriter(file_path) as tif:
        tif.save(data.astype('float32'), metadata=metadata)
    return


def create_unique_directory(base_output_dir):
    # Get the current date and time
    now = datetime.now()
    # Format the date and time as a string in the desired format, e.g., 'YYYY-MM-DD_HH-MM-SS'
    dir_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    unique_output_dir = os.path.join(base_output_dir, dir_name)
    os.makedirs(unique_output_dir, exist_ok=True)
    print(f"Created the unique output directory {unique_output_dir}")
    return unique_output_dir


def get_forward_img_str_postfix(optical_info):
    """Generates the postfix for the forward_img_str variable based on
    the number of microlenses and pixels per microlens.
    
    Args:
        optical_info (dict)

    Returns:
        postfix (str): The postfix for the forward_img_str variable.
    """
    num_microlenses = optical_info['n_micro_lenses']
    num_pixels = optical_info['pixels_per_ml']
    postfix = f'_{num_microlenses}mla_{num_pixels}pix'
    return postfix
