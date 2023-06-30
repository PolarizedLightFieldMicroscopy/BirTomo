"""Script to run forward intensity projection
This script using numpy back-end to:
    - Compute the ray geometry depending on the Light field microscope and volume configuration.
    - Traverse the rays through the volume.
    - Compute the intensity measurements for every ray for various polarization settings.
    - Generate 2D images.
    - Save and open the intensity images as bytes files.
"""
import os
import numpy as np
import time         # to measure ray tracing time
import matplotlib.pyplot as plt
from plotting_tools import plot_intensity_images, plot_retardance_orientation
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import  (
    BirefringentVolume,
    BirefringentRaytraceLFM,
    JonesMatrixGenerators
)

# Select backend method
backend = BackEnds.NUMPY
# backend = BackEnds.PYTORCH

create_images = False

testing = False    
if testing:
    output_dir = "intensity_tests"
    os.makedirs(output_dir, exist_ok=True)
    filename = "image.bin"
    image_float32 = image.astype(np.float32)
    image_bytes = image_float32.tobytes()
    file_path = os.path.join(output_dir, filename)
    with open(file_path, "wb") as file:
        file.write(image_bytes)
    with open(file_path, "rb") as file:
        image_bytes_new = file.read()
    image_new = np.frombuffer(image_bytes, dtype=np.float32)
    image_shaped = image_new.reshape((shape, shape, 1))

if create_images:
    # Get optical parameters template
    optical_info = BirefringentVolume.get_optical_info_template()
    # Alter some of the optical parameters
    optical_info['volume_shape'] = [15, 51, 51]
    optical_info['axial_voxel_size_um'] = 1.0
    optical_info['cube_voxels'] = True
    optical_info['pixels_per_ml'] = 17
    optical_info['n_micro_lenses'] = 1
    optical_info['n_voxels_per_ml'] = 1
    # Create non-identity polarizers and analyzers
    # LC-PolScope setup
    optical_info['analyzer'] = JonesMatrixGenerators.left_circular_polarizer()
    optical_info['polarizer_swing'] = 0.03

    # number is the shift from the end of the volume, change it as you wish,
    #       do single_voxel{volume_shape[0]//2} for a voxel in the center
    shift_from_center = 0
    volume_axial_offset = optical_info['volume_shape'][0] // 2 + shift_from_center # for center

    # Create a Birefringent Raytracer!
    rays = BirefringentRaytraceLFM(backend=backend, optical_info=optical_info)

    # Compute the rays and use the Siddon's algorithm to compute the intersections with voxels.
    # If a filepath is passed as argument, the object with all its calculations
    #   get stored/loaded from a file
    startTime = time.time()
    rays.compute_rays_geometry()
    executionTime = time.time() - startTime
    print('Ray-tracing time in seconds: ' + str(executionTime))

    # Load volume from a file
    loaded_volume = BirefringentVolume.init_from_file("objects/single_voxel.h5", backend, optical_info)
    # loaded_volume = BirefringentVolume.init_from_file("objects/shell.h5", backend, optical_info)
    my_volume = loaded_volume

    image_list = rays.ray_trace_through_volume(my_volume, intensity=True)
    executionTime = time.time() - startTime
    print(f'Execution time in seconds with backend {backend}: ' + str(executionTime))


    output_dir = f"intensity_images_shape_{image_list[0].shape[0]}"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over the image list
    for i, image in enumerate(image_list):
        # Generate a unique filename for each image
        filename = f"image_{i}.bin"
        
        image_float32 = image.astype(np.float32)
        image_bytes = image_float32.tobytes()

        # Write the bytes to a file in the output directory
        file_path = os.path.join(output_dir, filename)
        with open(file_path, "wb") as file:
            file.write(image_bytes)
else:
    image_list = []

    output_dir = f"intensity_images_shape_51"
    output_dir = f"intensity_images_shape_85"
    output_dir = f"intensity_images_shape_17"
    # Read each byte file and reconstruct the images
    for filename in os.listdir(output_dir):
        if filename.endswith(".bin"):
            # Read the bytes from the file
            file_path = os.path.join(output_dir, filename)
            with open(file_path, "rb") as file:
                image_bytes = file.read()

            shape = int(output_dir.split("_")[-1])

            # Convert the bytes to a NumPy array
            image = np.frombuffer(image_bytes, dtype=np.float32)
            image = image.reshape((shape, shape))

            # Append the image to the list
            image_list.append(image)



if backend==BackEnds.PYTORCH:
    image_list = [img.detach().cpu().numpy() for img in image_list]


[ret_image, azim_image] = BirefringentRaytraceLFM.ret_and_azim_from_intensity(image_list, 0.03)

my_fig = plot_retardance_orientation(ret_image, azim_image)
plt.pause(0.2)
plt.show(block=True)

my_fig = plot_intensity_images(image_list)
plt.pause(0.2)
plt.show(block=True)
