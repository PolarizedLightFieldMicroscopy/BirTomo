import tifffile

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
        tif.save(data, metadata=metadata)
    return
