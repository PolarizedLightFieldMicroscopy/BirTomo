import numpy as np
from my_siddon import *
from object import *
from ray_optics import *
from jones import *
from plotting_tools import *

# Import waveblocks objects
from waveblocks.blocks.optic_config import *

plot = False

magnObj = 60
pixels_per_ml = 17 # num pixels behind lenslet
camPixPitch = 6.5
microLensPitch = pixels_per_ml * camPixPitch / magnObj
# voxPitch is the width of each voxel in um (dividing by 5 to supersample)
voxPitch = microLensPitch / 1
# voxPitch = 1
axialPitch = voxPitch

'''The number of voxels along each side length of the cube is determined by the voxPitch. 
An odd number of voxels will allow the center of a voxel in the center of object space.
Object space center:
    - voxCtr:center voxel where all rays of the central microlens converge
    - volCtr:same center in micrometers'''

# Volume shape
voxNrX = 5
voxNrYZ = 5
# Volume span in um
nVoxX = axialPitch * voxNrX
nVoxYZ = voxPitch * voxNrYZ

voxCtr = np.array([voxNrX / 2, voxNrYZ / 2, voxNrYZ / 2]) # okay if not integers
volCtr = [voxCtr[0] * axialPitch, voxCtr[1] * voxPitch, voxCtr[2] * voxPitch]   # in vol units (um)

wavelength = 0.550
naObj = 1.2
nMedium = 1.33 # 1.52

# Populate OpticConfig from WaveBlocks with these values, to start migration
optic_config = OpticConfig()
use_default_values = False
if not use_default_values:
    # Set objective info
    optic_config.PSF_config.M = magnObj      # Objective magnification
    optic_config.PSF_config.NA = naObj    # Objective NA
    optic_config.PSF_config.ni = nMedium   # Refractive index of sample (experimental)
    optic_config.PSF_config.ni0 = nMedium  # Refractive index of sample (design value)
    optic_config.PSF_config.wvl = wavelength
    optic_config.mla_config.n_pixels_per_mla = pixels_per_ml
    optic_config.camera_config.sensor_pitch = camPixPitch
    optic_config.mla_config.pitch = optic_config.mla_config.n_pixels_per_mla * optic_config.camera_config.sensor_pitch
    optic_config.mla_config.n_mlas = 100

    optic_config.volume_config.volume_shape = [voxNrX, voxNrYZ, voxNrYZ]
    optic_config.volume_config.voxel_size_um = [axialPitch, voxPitch, voxPitch]
    optic_config.volume_config.volume_size_um = np.array(optic_config.volume_config.volume_shape) * np.array(optic_config.volume_config.voxel_size_um)
else:
    # Set objective info
    optic_config.PSF_config.M = 60      # Objective magnification
    optic_config.PSF_config.NA = 1.2    # Objective NA
    optic_config.PSF_config.ni = 1.52   # Refractive index of sample (experimental)
    optic_config.PSF_config.ni0 = 1.52  # Refractive index of sample (design value)
    optic_config.PSF_config.wvl = 0.550
    optic_config.mla_config.n_pixels_per_mla = 16
    optic_config.camera_config.sensor_pitch = 6.5
    optic_config.mla_config.pitch = optic_config.mla_config.n_pixels_per_mla * optic_config.camera_config.sensor_pitch
    optic_config.mla_config.n_mlas = 100

    optic_config.volume_config.volume_shape = [10, 10, 10]
    optic_config.volume_config.voxel_size_um = [1,] + 2*[optic_config.mla_config.pitch / optic_config.PSF_config.M]
    optic_config.volume_config.volume_size_um = np.array(optic_config.volume_config.volume_shape) * np.array(optic_config.volume_config.voxel_size_um)

def main():
    ray_enter, ray_exit, ray_diff = rays_through_vol(pixels_per_ml, naObj, nMedium, volCtr)
    # i = 3
    # j = 10
    # effective_JM = calc_cummulative_JM_of_ray(ray_enter, ray_exit, ray_diff, i, j)

    if plot:
        # plot_rays_at_sample(ray_enter, ray_exit, colormap='inferno', optical_config=None, use_matplotlib=False)
        start = ray_enter[:,i,j]
        stop = ray_exit[:,i,j]
        voxels_of_segs, ell_in_voxels = siddon(start, stop, optic_config.volume_config.voxel_size_um, optic_config.volume_config.volume_shape)
        plot_ray_path(start, stop, voxels_of_segs, optic_config, ell_in_voxels, colormap='hot')
    # {
    #     'volume_shape' : [voxNrX,voxNrYZ,voxNrYZ], 
    #     'volume_size_um' : }optic_config)

    ret_image = np.zeros((pixels_per_ml, pixels_per_ml))
    azim_image = np.zeros((pixels_per_ml, pixels_per_ml))
    for i in range(pixels_per_ml):
        for j in range(pixels_per_ml):
            if np.isnan(ray_enter[0, i, j]):
                ret_image[i, j] = 0
                azim_image[i, j] = 0
            else:
                effective_JM = calc_cummulative_JM_of_ray(ray_enter, ray_exit, ray_diff, i, j)
                ret_image[i, j] = calc_retardance(effective_JM)
                azim_image[i, j] = calc_azimuth(effective_JM)
    plt.imshow(ret_image)
    plt.show()


    print(f"Effective Jones matrix for the ray hitting pixel {i, j}: {effective_JM}")
    ret = calc_retardance(effective_JM)
    azim = calc_azimuth(effective_JM)
    print(f"Cummulated retardance: {round(ret / np.pi, 2)}*pi, azimuth {round(azim / np.pi, 2)}*pi")

if __name__ == '__main__':
    main()
    