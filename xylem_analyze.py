'''Analyze the xylem images
- Load the xylem images from the folder xylem
- Find the shape of each image
'''

import os
import tifffile
import matplotlib.pyplot as plt

# Load the images
image_folder = 'xylem'
image_names = os.listdir(image_folder)
image_names.sort()
images = []
for image_name in image_names:
    image = tifffile.imread(os.path.join(image_folder, image_name))
    images.append(image)
# images = np.array(images)

# Find the shape of each image
shapes = []
for image in images:
    shape = image.shape
    shapes.append(shape)
# shapes = np.array(shapes)
print(shapes)