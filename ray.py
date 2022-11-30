import numpy as np
from my_siddon import *
from object import *
from ray_optics import *
from jones import *
from plotting_tools import *

# Import waveblocks objects
from waveblocks.blocks.optic_config import *

magnObj = 60
nrCamPix = 16 # num pixels behind lenslet
camPixPitch = 6.5
microLensPitch = nrCamPix * camPixPitch / magnObj
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
# voxCtr = np.array([(voxNrX - 1) / 2, (voxNrYZ - 1) / 2, (voxNrYZ - 1) / 2]) # in index units
volCtr = [voxCtr[0] * axialPitch, voxCtr[1] * voxPitch, voxCtr[2] * voxPitch]   # in vol units (um)

wavelength = 0.550
naObj = 1.2
nMedium = 1.52

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
    optic_config.mla_config.n_pixels_per_mla = nrCamPix
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
    pixels_per_ml = 17
    rayEnter, rayExit, rayDiff = rays_through_vol(pixels_per_ml, naObj, nMedium, volCtr)

    '''For the (i,j) pixel behind a single microlens'''
    i = 3
    j = 14
    start = rayEnter[:,i,j]
    stop = rayExit[:,i,j]
    siddon_list = siddon_params(start, stop, optic_config.volume_config.voxel_size_um, optic_config.volume_config.volume_shape)
    seg_mids = siddon_midpoints(start, stop, siddon_list)
    voxels_of_segs = vox_indices(seg_mids, optic_config.volume_config.voxel_size_um)
    ell_in_voxels = siddon_lengths(start, stop, siddon_list)

    # Plot
    plot_ray_path(start, stop, voxels_of_segs, optic_config, ell_in_voxels, colormap='hot')
    # {
    #     'volume_shape' : [voxNrX,voxNrYZ,voxNrYZ], 
    #     'volume_size_um' : }optic_config)
    ray = rayDiff[:,i,j]
    rayDir = calc_rayDir(ray)
    JM_list = []
    for m in range(len(ell_in_voxels)):
        ell = ell_in_voxels[m]
        vox = voxels_of_segs[m]
        Delta_n, opticAxis = get_ellipsoid(vox)
        JM = voxRayJM(Delta_n, opticAxis, rayDir, ell)
        JM_list.append(JM)
    effective_JM = rayJM(JM_list)
    print(f"Effective Jones matrix for the ray hitting pixel {i, j}: {effective_JM}")

if __name__ == '__main__':
    main()
    