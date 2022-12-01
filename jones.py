import numpy as np
from my_siddon import siddon
from object import get_ellipsoid

global wavelength
wavelength = 0.550

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
    # Azimuth is the angle of the sloq axis of retardance.
    azim = np.arctan2(np.dot(opticAxis, rayDir[1]), np.dot(opticAxis, rayDir[2]))
    if Delta_n == 0:
        azim = 0
    elif Delta_n < 0:
        azim = azim + np.pi / 2
    # print(f"Azimuth angle of index ellipsoid is {np.around(np.rad2deg(azim), decimals=0)} degrees.")
    ret = abs(Delta_n) * (1 - np.dot(opticAxis, rayDir[0]) ** 2) * 2 * np.pi * ell / wavelength
    # print(f"Accumulated retardance from index ellipsoid is {np.around(np.rad2deg(ret), decimals=0)} ~ {int(np.rad2deg(ret)) % 360} degrees.")
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

def find_orthogonal_vec(v1, v2):
    '''v1 and v2 are numpy arrays (3d vectors)
    This function accomodates for a divide by zero error.'''
    x = np.dot(v1, v2) / (np.norm(v1) * np.norm(v2))
    # Check if vectors are parallel or anti-parallel
    if x == 1 or x == -1:
        if v1[1] == 0:
            normal_vec = np.array([0, 1, 0])
        elif v1[2] == 0:
            normal_vec = np.array([0, 0, 1])
        elif v1[0] == 0:
            normal_vec = np.array([1, 0, 0])
        else:
            non_par_vec = np.array([1, 0, 0])
            normal_vec = np.cross(v1, non_par_vec) / np.linalg.norm(np.cross(v1, non_par_vec))
    else:
        normal_vec = np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2))
    return normal_vec

def calc_rayDir(ray):
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
    '''
    theta = np.arccos(np.dot(ray, np.array([1,0,0])))
    # Unit vectors that give the laboratory axes, can be changed
    scope_axis = np.array([1,0,0])
    scope_perp1 = np.array([0,1,0])
    scope_perp2 = np.array([0,0,1])
    theta = np.arccos(np.dot(ray, scope_axis))
    # print(f"Rotating by {np.around(np.rad2deg(theta), decimals=0)} degrees")
    normal_vec = find_orthogonal_vec(ray, scope_axis)
    # normal_vec = np.cross(ray, scope_axis) / np.linalg.norm(np.cross(ray, scope_axis))
    Rinv = rotation_matrix(normal_vec, -theta)
    # Extracting basis vectors that are orthogonal to the ray and will be parallel
    # to the laboratory axes that are not the optic axis after a rotation.
    # Note: If scope_perp1 if the y-axis, then ray_perp1 if the 2nd column of Rinv.
    ray_perp1 = np.dot(Rinv, scope_perp1)
    ray_perp2 = np.dot(Rinv, scope_perp2)
    return [ray, ray_perp1, ray_perp2]

def calc_cummulative_JM_of_ray(ray_enter, ray_exit, ray_diff, i, j, optic_config, voxel_parameters):
    '''For the (i,j) pixel behind a single microlens'''
    start = ray_enter[:,i,j]
    stop = ray_exit[:,i,j]
    voxels_of_segs, ell_in_voxels = siddon(start, stop, optic_config.volume_config.voxel_size_um, optic_config.volume_config.volume_shape)
    ray = ray_diff[:,i,j]
    rayDir = calc_rayDir(ray)
    JM_list = []
    for m in range(len(ell_in_voxels)):
        ell = ell_in_voxels[m]
        vox = voxels_of_segs[m]
        my_params = voxel_parameters[:, vox[0], vox[1], vox[2]].numpy()
        Delta_n = my_params[0]
        opticAxis = my_params[1:]
        # get_ellipsoid(vox)
        JM = voxRayJM(Delta_n, opticAxis, rayDir, ell)
        JM_list.append(JM)
    effective_JM = rayJM(JM_list)
    return effective_JM


def calc_retardance(JM):
    '''Calculates the retardance magnitude in radians from a Jones matrix
    Parameters:
        JM (np.array): 2x2 Jones matrix
    Returns:
        retardance (float): retardance magnitude in radians
    '''
    diag_sum = JM[0, 0] + JM[1, 1]
    diag_diff = JM[1, 1] - JM[0, 0]
    off_diag_sum = JM[0, 1] + JM[1, 0]
    # Note: np.arctan(1j) and np.arctan(-1j) gives an divide by zero error
    value = np.sqrt((off_diag_sum / diag_sum) ** 2 + (diag_diff / diag_sum) ** 2)
    if value == 1 or value == -1:
        retardance = 0
    else:
        arctan = np.arctan(1j * np.sqrt((off_diag_sum / diag_sum) ** 2 + (diag_diff / diag_sum) ** 2))
        retardance = 2 * np.real(arctan)
    return retardance

def calc_azimuth(JM):
    '''Calculates the retardance azimuth in radians from a Jones matrix
    Parameters:
        JM (np.array): 2x2 Jones matrix
    Returns:
        azimuth (float): azimuth of slow axis orientation in radians 
                            between -3pi/4 and pi/4
    '''
    diag_sum = JM[0, 0] + JM[1, 1]
    diag_diff = JM[1, 1] - JM[0, 0]
    off_diag_sum = JM[0, 1] + JM[1, 0]
    a = np.imag(diag_diff / diag_sum)
    b = np.imag(off_diag_sum / diag_sum)
    if a == 0 and b == 0:
        azimuth = 0
    else:
        azimuth = np.real(np.arctan2(a, b)) / 2 + np.pi / 2
    return azimuth

def main():
    JM = np.array([[3, 0], [0, 0]])
    ret = calc_retardance(JM)
    azim = calc_azimuth(JM)
    print(ret / np.pi, azim / np.pi)

    JM = voxRayJM(1, np.array([1, 0, 0]), [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])], 1)
    print(JM)
    ret = calc_retardance(JM)
    azim = calc_azimuth(JM)
    print(ret / np.pi, azim / np.pi)

if __name__ == '__main__':
    main()
