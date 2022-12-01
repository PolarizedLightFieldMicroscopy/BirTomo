import torch
import numpy as np

global wavelength
wavelength = 0.550

def rotation_matrix(axis, angle):
    '''Generates the rotation matrix that will rotate a 3D vector
    around "axis" by "angle" counterclockwise.'''
    ax, ay, az = axis[0], axis[1], axis[2]
    s = torch.sin(angle)
    c = torch.cos(angle)
    u = 1 - c
    R_tuple = ( ( ax*ax*u + c,    ax*ay*u - az*s, ax*az*u + ay*s ),
        ( ay*ax*u + az*s, ay*ay*u + c,    ay*az*u - ax*s ),
        ( az*ax*u - ay*s, az*ay*u + ax*s, az*az*u + c    ) )
    R = torch.asarray(R_tuple)
    return R

def voxRayJM(Delta_n, opticAxis, rayDir, ell):
    '''Compute Jones matrix associated with a particular ray and voxel combination'''
    if not torch.is_tensor(opticAxis):
        opticAxis = torch.from_numpy(opticAxis)
    # Azimuth is the angle of the sloq axis of retardance.
    azim = torch.arctan2(torch.linalg.multi_dot((opticAxis , rayDir[1])), torch.linalg.multi_dot((opticAxis , rayDir[2])))
    if Delta_n == 0:
        azim = torch.tensor([0.0])
    elif Delta_n < 0:
        azim = azim + torch.pi / 2
    # print(f"Azimuth angle of index ellipsoid is {np.around(torch.rad2deg(azim).numpy(), decimals=0)} degrees.")
    ret = abs(Delta_n) * (1 - torch.dot(opticAxis, rayDir[0]) ** 2) * 2 * torch.pi * ell / wavelength
    # print(f"Accumulated retardance from index ellipsoid is {np.around(torch.rad2deg(ret).numpy(), decimals=0)} ~ {int(torch.rad2deg(ret).numpy()) % 360} degrees.")
    offdiag = 1j * torch.sin(2 * azim) * torch.sin(ret / 2)
    diag1 = torch.cos(ret / 2) + 1j * torch.cos(2 * azim) * torch.sin(ret / 2)
    diag2 = torch.conj(diag1)
    return torch.tensor([[diag1, offdiag], [offdiag, diag2]])

def rayJM(JMlist):
    '''Computes product of Jones matrix sequence
    Equivalent method: torch.linalg.multi_dot([JM1, JM2])
    '''
    product = torch.tensor([[1.0,0],[0,1.0]], dtype=torch.complex64)
    for JM in JMlist:
        product = product @ JM
    return product

def find_orthogonal_vec(v1, v2):
    '''v1 and v2 are numpy arrays (3d vectors)
    This function accomodates for a divide by zero error.'''
    x = torch.dot(v1, v2) / (torch.linalg.norm(v1) * torch.linalg.norm(v2))
    # Check if vectors are parallel or anti-parallel
    if x == 1 or x == -1:
        if v1[1] == 0:
            normal_vec = torch.tensor([0, 1.0, 0])
        elif v1[2] == 0:
            normal_vec = torch.tensor([0, 0, 1.0])
        elif v1[0] == 0:
            normal_vec = torch.tensor([1.0, 0, 0])
        else:
            non_par_vec = torch.tensor([1.0, 0, 0])
            normal_vec = torch.cross(v1, non_par_vec) / torch.linalg.norm(np.cross(v1, non_par_vec))
    else:
        normal_vec = torch.cross(v1, v2) / torch.linalg.norm(torch.cross(v1, v2))
    return normal_vec

def calc_rayDir(ray_in):
    '''
    Allows to the calculations to be done in ray-space coordinates
    as oppossed to laboratory coordinates
    Parameters:
        ray_in (torch.array): normalized 3D vector giving the direction 
                        of the light ray
    Returns:
        ray (torch.array): same as input
        ray_perp1 (torch.array): normalized 3D vector
        ray_perp2 (torch.array): normalized 3D vector
    '''
    if not torch.is_tensor(ray_in):
        ray = torch.from_numpy(ray_in)
    else:
        ray = ray_in
    theta = torch.arccos(torch.dot(ray, torch.tensor([1.0,0,0],dtype=ray.dtype)))
    # Unit vectors that give the laboratory axes, can be changed
    scope_axis = torch.tensor([1.0,0,0],dtype=ray.dtype)
    scope_perp1 = torch.tensor([0,1.0,0],dtype=ray.dtype)
    scope_perp2 = torch.tensor([0,0,1.0],dtype=ray.dtype)
    theta = torch.arccos(torch.dot(ray, scope_axis))
    # print(f"Rotating by {np.around(torch.rad2deg(theta).numpy(), decimals=0)} degrees")
    normal_vec = find_orthogonal_vec(ray, scope_axis)
    Rinv = rotation_matrix(normal_vec, -theta)
    # Extracting basis vectors that are orthogonal to the ray and will be parallel
    # to the laboratory axes that are not the optic axis after a rotation.
    # Note: If scope_perp1 if the y-axis, then ray_perp1 if the 2nd column of Rinv.
    ray_perp1 = torch.linalg.multi_dot((Rinv, scope_perp1)) 
    ray_perp2 = torch.linalg.multi_dot((Rinv, scope_perp2))
    return [ray, ray_perp1, ray_perp2]

def calc_retardance(JM):
    '''Calculates the retardance magnitude in radians from a Jones matrix
    Parameters:
        JM (torch.tensor): 2x2 Jones matrix
    Returns:
        retardance (float): retardance magnitude in radians
    '''
    diag_sum = JM[0, 0] + JM[1, 1]
    diag_diff = JM[1, 1] - JM[0, 0]
    off_diag_sum = JM[0, 1] + JM[1, 0]
    # Note: np.arctan(1j) and np.arctan(-1j) gives an divide by zero error
    value = torch.sqrt((off_diag_sum / diag_sum) ** 2 + (diag_diff / diag_sum) ** 2)
    if torch.isnan(value) or value == 1 or value == -1:
        retardance = torch.tensor([0.0])
    else:
        arctan = torch.arctan(1j * torch.sqrt((off_diag_sum / diag_sum) ** 2 + (diag_diff / diag_sum) ** 2))
        retardance = 2 * torch.real(arctan)
    
    # Alternative way of computing this, according to:
    # "Three-dimensional polarization ray-tracing calculus II: retardance"
    # x = np.linalg.eigvals(JM)
    # retardance = np.angle(x[1], deg=False)-np.angle([x[0]], deg=False)
    return torch.abs(retardance)

def calc_azimuth(JM):
    '''Calculates the retardance azimuth in radians from a Jones matrix
    Parameters:
        JM (torch.tensor): 2x2 Jones matrix
    Returns:
        azimuth (float): azimuth of slow axis orientation in radians 
                            between -3pi/4 and pi/4
    '''
    diag_sum = JM[0, 0] + JM[1, 1]
    diag_diff = JM[1, 1] - JM[0, 0]
    off_diag_sum = JM[0, 1] + JM[1, 0]
    a = (diag_diff / diag_sum).imag
    b = (off_diag_sum / diag_sum).imag
    if a == 0 and b == 0:
        azimuth = 0
    else:
        azimuth = torch.real(torch.arctan2(a, b)) / 2 + torch.pi / 2
    return azimuth



def main():
    JM = torch.tensor([[3, 0], [0, 0]])
    ret = calc_retardance(JM)
    azim = calc_azimuth(JM)
    print(ret / np.pi, azim / np.pi)

if __name__ == '__main__':
    main()
