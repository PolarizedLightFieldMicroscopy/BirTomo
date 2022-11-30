import numpy as np

global wavelength

wavelength = 0.550

def set_wavelength(wl):
    wavelength = wl

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
    print(f"Rotating by {np.around(np.rad2deg(theta), decimals=0)} degrees")
    normal_vec = np.cross(ray, scope_axis) / np.linalg.norm(np.cross(ray, scope_axis))
    Rinv = rotation_matrix(normal_vec, -theta)
    # Extracting basis vectors that are orthogonal to the ray and will be parallel
    # to the laboratory axes that are not the optic axis after a rotation.
    # Note: If scope_perp1 if the y-axis, then ray_perp1 if the 2nd column of Rinv.
    ray_perp1 = np.dot(Rinv, scope_perp1)
    ray_perp2 = np.dot(Rinv, scope_perp2)
    return [ray, ray_perp1, ray_perp2]

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
    # print(f"sqrt portion: {np.sqrt((off_diag_sum / diag_sum) ** 2 + (diag_diff / diag_sum) ** 2)}")
    # Note: np.arctan(1j) and np.arctan(-1j) gives an divide by zero error
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

if __name__ == '__main__':
    main()
