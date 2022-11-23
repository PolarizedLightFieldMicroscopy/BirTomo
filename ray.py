import numpy as np
from my_siddon import *
from object import *

magnObj = 60
nrCamPix = 16 # num pixels behind lenslet
camPixPitch = 6.5
microLensPitch = nrCamPix * camPixPitch / magnObj
voxPitch = microLensPitch / 5

'''The number of voxels along each side length of the cube is determined by the voxPitch. 
An odd number of voxels will allow the center of a voxel in the center of object space.
Object space center:
    - voxCtr:center voxel where all rays of the central microlens converge
    - volCtr:same center in micrometers'''

voxNrX = round(250/voxPitch)
if voxNrX % 2 == 1:
    voxNrX += 1
voxNrYZ = round(700/voxPitch)
if voxNrYZ % 2 == 1:
    voxNrYZ += 1
voxCtr = np.array([voxNrX/2, voxNrYZ/2, voxNrYZ/2])
volCtr = voxCtr * voxPitch

wavelength = 0.550
naObj = 1.2
nMedium = 1.52

def main():
    '''Finding angles to/between central lenset, which is the angle going to each 
    of the 16 pixels for each microlens.'''

    microLensCtr = [8, 8] # (unit: camera pixels)
    rNA = 7.5 # radius of edge of microlens lens (unit:camera pixels), 
                # can be measured in back focal plane of microlenses
    camPixRays = np.zeros([nrCamPix, nrCamPix])
    i = np.linspace(1, nrCamPix, nrCamPix)
    j = np.linspace(1, nrCamPix, nrCamPix)
    jv, iv = np.meshgrid(i, j) # row/column defined instead of by coordinate
    distFromCtr = np.sqrt((iv-0.5-microLensCtr[0])**2 + (jv-0.5-microLensCtr[1])**2)
    camPixRays[distFromCtr > rNA] = np.NaN
    iRel2Ctr = iv-0.5-microLensCtr[0]
    jRel2Ctr = jv-0.5-microLensCtr[1]
    camPixRaysAzim = np.round(np.rad2deg(np.arctan2(jRel2Ctr, iRel2Ctr)))
    camPixRaysAzim[distFromCtr > rNA] = np.NaN
    distFromCtr[distFromCtr > rNA] = np.NaN
    camPixRaysTilt = np.round(np.rad2deg(np.arcsin(distFromCtr/rNA*naObj/nMedium)))

    '''Camera ray entrance. For each inital ray position, we find the position on the 
    entrance face of the object cube for which the ray enters.
    This is bascially the same as "rayEnter". Here x=0.'''

    camRayEntranceX = np.zeros([nrCamPix, nrCamPix])
    camRayEntranceY = volCtr[0]*np.tan(np.deg2rad(camPixRaysTilt))*np.sin(np.deg2rad(camPixRaysAzim))+volCtr[1]
    camRayEntranceZ = volCtr[0]*np.tan(np.deg2rad(camPixRaysTilt))*np.cos(np.deg2rad(camPixRaysAzim))+volCtr[2]
    camRayEntranceX[np.isnan(camRayEntranceY)] = np.NaN
    nrRays = np.sum(~np.isnan(camRayEntranceY)) # Number of all rays in use
    camRayEntrance = np.array([camRayEntranceX, camRayEntranceY, camRayEntranceZ])
    rayEnter = camRayEntrance.copy()
    volCtrGridTemp = np.array([np.full((nrCamPix,nrCamPix), volCtr[i]) for i in range(3)])
    rayExit = rayEnter + 2 * (volCtrGridTemp - rayEnter)

    '''Direction of the rays at the exit plane'''
    rayDiff = rayExit - rayEnter
    mags = np.linalg.norm(rayDiff, axis=0)
    rayDiff = rayDiff / mags

    '''For the (i,j) pixel behind a single microlens'''
    i = 3
    j = 8
    start = rayEnter[:,i,j]
    stop = rayExit[:,i,j]
    siddon_list = siddon_params(start, stop, [voxPitch]*3, [voxNrX, voxNrYZ, voxNrYZ])
    seg_mids = siddon_midpoints(start, stop, siddon_list)
    voxels_of_segs = vox_indices(seg_mids, [voxPitch]*3)
    ell_in_voxels = siddon_lengths(start, stop, siddon_list)

    ray = rayDiff[:,i,j]
    rayUnitVectors = calc_rayUnitVectors(ray)
    rayDir = rayUnitVectors[0:3]
    JM_list = []
    for m in range(len(ell_in_voxels)):
        ell = ell_in_voxels[m]
        vox = voxels_of_segs[m]
        Delta_n, opticAxis = get_ellipsoid(vox)
        JM = voxRayJM(Delta_n, opticAxis, rayDir, ell)
        JM_list.append(JM)
    effective_JM = rayJM(JM_list)
    print(f"Effective Jones matrix for the ray hitting pixel {i, j}: {effective_JM}")

def rotation_matrix(axis, angle):
    '''Generates the rotation matrix that will rotate a 3D vector
    around "axis" by "angle" counterclockwise.'''
    ax, ay, az = axis[0], axis[1], axis[2]
    s = np.sin(angle)
    c = np.cos(angle)
    u = 1 - c
    R_tuple = ( ( ax*ax*u + c,    ax*ay*u - az*s, ax*az*u + ay*s ),
        ( ay*ax*u + az*s, ay*ay*u + c,    ay*az*u - ax*s ),
        ( az*ax*u - ay*s, az*ay*u + ax*s, az*az*u + c    ) )
    R = np.asarray(R_tuple)
    return R

def voxRayJM(Delta_n, opticAxis, rayDir, ell):
    '''Compute Jones matrix associated with a particular ray and voxel combination'''
    azim = np.arctan2(np.dot(opticAxis, rayDir[1]), np.dot(opticAxis, rayDir[2]))
    # add dependence of azim on birefringence
    print(f"Azimuth angle of index ellipsoid is {np.around(np.rad2deg(azim), decimals=0)} degrees.")
    ret = abs(Delta_n) * (1 - np.dot(opticAxis, rayDir[0]) ** 2) * 2 * np.pi * ell / wavelength
    print(f"Accumulated retardance from index ellipsoid is {np.around(np.rad2deg(ret), decimals=0)} ~ {int(np.rad2deg(ret)) % 360} degrees.")
    offdiag = 1j * np.sin(2 * azim) * np.sin(ret / 2)
    diag1 = np.cos(ret / 2) + 1j * np.cos(2 * azim) * np.sin(ret / 2)
    diag2 = np.conj(diag1)
    return np.matrix([[diag1, offdiag], [offdiag, diag2]])

def rayJM(JMlist):
    '''Computes product of Jones matrix sequence
    Equivalent method: np.linalg.multi_dot([JM1, JM2])
    '''
    product = np.identity(2)
    for JM in JMlist:
        product = product @ JM
    return product

def calc_rayUnitVectors(ray):
    '''
    Allows to the calculations to be done in ray-space coordinates
    as oppossed to laboratory coordinates
    Parameters:
        ray (np.array): normalized 3D vector giving the direction 
                        of the light ray
    Returns:
        ray (np.array): same as input
        ray_perp1 (np.array): normalized 3D vector
        ray_perp2 (np.array): normalized 3D vector
        R (np.array): 3x3 rotation matrix form ray to lab frame
    '''
    theta = np.arccos(np.dot(ray, np.array([1,0,0])))
    # Unit vectors that give the laboratory axes, can be changed
    scope_axis = np.array([1,0,0])
    scope_perp1 = np.array([0,1,0])
    scope_perp2 = np.array([0,0,1])
    theta = np.arccos(np.dot(ray, scope_axis))
    print(f"Rotating by {np.around(np.rad2deg(theta), decimals=0)} degrees")
    normal_vec = np.cross(ray, scope_axis) / np.linalg.norm(np.cross(ray, scope_axis))
    R = rotation_matrix(normal_vec, theta)
    Rinv = rotation_matrix(normal_vec, -theta)
    # Extracting basis vectors that are orthogonal to the ray and will be parallel
    # to the laboratory axes that are not the optic axis after a rotation.
    # Note: If scope_perp1 if the y-axis, then ray_perp1 if the 2nd column of Rinv.
    ray_perp1 = np.dot(Rinv, scope_perp1)
    ray_perp2 = np.dot(Rinv, scope_perp2)

    return [ray, ray_perp1, ray_perp2, R]

if __name__ == '__main__':
    main()
    