from tifffile import imread
import numpy as np
import matplotlib.pyplot as plt


filename = 'sphere/0_sphere.tiff'
image = imread(filename)

def transform_into_perspective(img, n_lenses, n_pix):
    perspective_img = np.zeros((n_lenses * n_pix, n_lenses * n_pix))
    n_lenses = 33
    n_pix = 17
    for lx in range(n_lenses):
        for ly in range(n_lenses):
            for i in range(n_pix):
                for j in range(n_pix):
                    lfx = lx * n_pix + i
                    lfy = ly * n_pix + j
                    psx = i *  n_lenses + lx
                    psy = j *  n_lenses + ly
                    perspective_img[psx, psy] = img[lfx, lfy]
    return perspective_img

plt.imshow(image[0])

psv_img = transform_into_perspective(image[0], 33, 17)
plt.imshow(psv_img)