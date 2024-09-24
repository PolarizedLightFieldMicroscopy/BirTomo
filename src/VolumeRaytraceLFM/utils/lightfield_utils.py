import numpy as np


def average_intensity_per_lenslet(image_array, pix_per_lenslet):
    """
    Calculate the average intensity across each lenslet region in an image.
    Args:
        image_array (numpy.array): The image array.
        pix_per_lenslet (int): The number of pixels per lenslet (assuming square lenslets).
    Returns:
        numpy.array: A 2D array with the average intensity values of each lenslet.
    """
    height, width = image_array.shape
    # Determine the number of lenslets along each dimension
    num_lenslets_y = height // pix_per_lenslet
    num_lenslets_x = width // pix_per_lenslet
    
    # Initialize an array to store the average intensities
    lenslet_averages = np.zeros((num_lenslets_y, num_lenslets_x))
    
    # Iterate over each block
    for i in range(num_lenslets_y):
        for j in range(num_lenslets_x):
            # Calculate the start and end indices of the lenslet in the image
            start_y = i * pix_per_lenslet
            end_y = start_y + pix_per_lenslet
            start_x = j * pix_per_lenslet
            end_x = start_x + pix_per_lenslet
            
            # Extract the lenslet region
            lenslet_region = image_array[start_y:end_y, start_x:end_x]
            
            # Calculate the average intensity of the lenslet
            lenslet_averages[i, j] = np.mean(lenslet_region)
    
    return lenslet_averages
