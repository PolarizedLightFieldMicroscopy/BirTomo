import torch
from VolumeRaytraceLFM.birefringence_implementations import *

from jones import *
import time

# Objective configuration
magnObj = 60
wavelength = 0.550
naObj = 1.2
nMedium = 1.33
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
volume_shape = [1, 3, 3]





# Volume span in um
nVoxX = axialPitch * volume_shape[0]
nVoxYZ = voxPitch * volume_shape[1]
voxCtr = np.array([volume_shape[0] / 2, volume_shape[1] / 2, volume_shape[2] / 2]) # okay if not integers
vox_ctr_idx = np.round(voxCtr).astype(int)
volCtr = [voxCtr[0] * axialPitch, voxCtr[1] * voxPitch, voxCtr[2] * voxPitch]   # in vol units (um)

# Computed with numpy functions
ray_enter, ray_exit, ray_diff = rays_through_vol(pixels_per_ml, naObj, nMedium, volCtr)

# Define volume with a birefringent voxel in the center
voxel_parameters = np.zeros([4, volume_shape[0],volume_shape[1],volume_shape[2]])
offset = 0
voxel_parameters[
    :, 
    vox_ctr_idx[0], 
    vox_ctr_idx[1]+offset, 
    vox_ctr_idx[2]+offset] \
    = torch.tensor([0.1, 0, 1, 0])

from ray import *
startTime = time.time()
ret_image, azim_image = ret_and_azim_images(ray_enter, ray_exit, ray_diff, pixels_per_ml, voxel_parameters, voxel_size_um)
executionTime = (time.time() - startTime)
print('Execution time in seconds with Numpy: ' + str(executionTime))

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
