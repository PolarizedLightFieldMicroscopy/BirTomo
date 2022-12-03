import time
import matplotlib.pyplot as plt
from jones import *
from ray_optics import *

""" This script using numpy and main_test_VolumeRaytraceLFM using Pytorch
    have the exact same functionality:
    - Create a volume with a single birefringent voxel.
    - Compute the ray geometry depending on the Light field microscope.
    - Traverse the rays through the volume and accumulate the retardance.
    - Compute the final ray retardance and azimuth for every ray.
    - Generate 2D images of a single lenslet. """

# Objective configuration
magnObj = 60
wavelength = 0.550
naObj = 1.2
nMedium = 1.52
# Camera and volume configuration
camPixPitch = 6.5
# MLA configuration
pixels_per_ml = 17 # num pixels behind lenslet
microLensPitch = pixels_per_ml * camPixPitch / magnObj
# voxPitch is the width of each voxel in um (dividing by 5 to supersample)
voxPitch = microLensPitch / 1
axialPitch = voxPitch
voxel_size_um = [axialPitch, voxPitch, voxPitch]
# Volume shape
volume_shape = [11, 11, 11]

# Volume span in um
voxCtr = np.array([volume_shape[0] / 2, volume_shape[1] / 2, volume_shape[2] / 2]) # okay if not integers
vox_ctr_idx = voxCtr.astype(int)
volCtr = [voxCtr[0] * axialPitch, voxCtr[1] * voxPitch, voxCtr[2] * voxPitch]   # in vol units (um)

############### Implementation

# Computed ray geometry mapping through the volume until hitting the camera
# based on LFM configuration
ray_enter, ray_exit, ray_diff = rays_through_vol(pixels_per_ml, naObj, nMedium, volCtr)

# Define volume with a birefringent voxel in the center
voxel_parameters = np.zeros([4, volume_shape[0],volume_shape[1],volume_shape[2]])
offset = 0
voxel_parameters[
    :, 
    vox_ctr_idx[0], 
    vox_ctr_idx[1]+offset, 
    vox_ctr_idx[2]+offset] \
    = np.array([.1, 0, 1.0, 0])

# Traverse volume for every ray, and generate retardance and azimuth images
startTime = time.time()
ret_image, azim_image = ret_and_azim_images(ray_enter, ray_exit, ray_diff, 
        pixels_per_ml, 
        voxel_parameters, 
        voxel_size_um)
executionTime = (time.time() - startTime)
print('Execution time in seconds with Numpy: ' + str(executionTime))

# Plot
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(ret_image)
plt.colorbar()
plt.title('Retardance numpy')
plt.subplot(1,2,2)
plt.imshow(azim_image)
plt.colorbar()
plt.title('Azimuth numpy')

plt.show(block=True)
