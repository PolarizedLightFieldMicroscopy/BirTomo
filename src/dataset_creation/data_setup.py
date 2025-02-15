from tifffile import imread
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


def transform_into_perspective(img, n_lenses, n_pix):
    perspective_img = np.zeros((n_lenses * n_pix, n_lenses * n_pix))
    for lx in range(n_lenses):
        for ly in range(n_lenses):
            for i in range(n_pix):
                for j in range(n_pix):
                    lfx = lx * n_pix + i
                    lfy = ly * n_pix + j
                    psx = i * n_lenses + lx
                    psy = j * n_lenses + ly
                    perspective_img[psx, psy] = img[lfx, lfy]
    return perspective_img


def transform_into_perspective_reshape(img, n_lenses, n_pix):
    # img.shape = (L*n_pix, L*n_pix)
    # Step 1: reshape -> (L, P, L, P)
    reshaped = img.reshape((n_lenses, n_pix, n_lenses, n_pix))

    # Step 2: transpose -> (P, L, P, L)
    transposed = reshaped.transpose(1, 0, 3, 2)

    # Step 3: reshape -> (L*P, L*P)
    perspective_img = transposed.reshape((n_lenses * n_pix, n_lenses * n_pix))
    return perspective_img


def transform_into_perspective_einsum(img, n_lenses, n_pix):
    # "abcd -> badc"
    reshaped = img.reshape(n_lenses, n_pix, n_lenses, n_pix)
    perspective_img = np.einsum('abcd->badc', reshaped)
    return perspective_img.reshape((n_lenses * n_pix, n_lenses * n_pix))


def transform_from_perspective(perspective_img, n_lenses, n_pix):
    # perspective_img.shape = (L*P, L*P)
    reshaped = perspective_img.reshape((n_pix, n_lenses, n_pix, n_lenses))
    transposed = reshaped.transpose(1, 0, 3, 2)  # back to (L, P, L, P)
    light_field_img = transposed.reshape((n_lenses * n_pix, n_lenses * n_pix))
    return light_field_img


def transform_channels_to_2d(image):
    """
    Reshapes a 4D array of shape (2, num_pixels^2, num_lenslets, num_lenslets)
    into two 2D images for plotting. Specifically:
    
      (2, num_pixels^2, num_lenslets, num_lenslets)
      -> (2, num_pixels, num_pixels, num_lenslets, num_lenslets)
      -> (2, num_pixels * num_lenslets, num_pixels * num_lenslets)
    
    The first channel (index 0) corresponds to Retardance,
    the second (index 1) corresponds to Orientation.
    """
    assert image.dim() == 4, f"Expected 4D image, got {image.dim()}D"
    num_pixels = int(sqrt(image.shape[1]))
    image = image.reshape(2, num_pixels, num_pixels, *image.shape[2:])
    image = image.transpose(2, 3).flatten(1, 2).flatten(2, 3)
    return image


if __name__ == "__main__":
    filename = "sphere/0_sphere.tiff"
    image = imread(filename)

    # view image
    plt.imshow(image[0])

    psv_img = transform_into_perspective(image[0], 33, 17)
    plt.imshow(psv_img)
