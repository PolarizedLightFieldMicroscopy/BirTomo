import torch
import numpy as np

global wavelength
wavelength = 0.550

def rotation_matrix(axis, angle):
    '''Generates the rotation matrix that will rotate a 3D vector
    around "axis" by "angle" counterclockwise.'''
    ax, ay, az = axis[:,0], axis[:,1], axis[:,2]
    s = torch.sin(angle)
    c = torch.cos(angle)
    u = 1 - c
    R = torch.zeros([angle.shape[0],3,3], device=axis.device)
    # R_tuple = ( ( ax*ax*u + c,    ax*ay*u - az*s, ax*az*u + ay*s ),
    #     ( ay*ax*u + az*s, ay*ay*u + c,    ay*az*u - ax*s ),
    #     ( az*ax*u - ay*s, az*ay*u + ax*s, az*az*u + c    ) )
    # R = torch.asarray(R_tuple)
    # todo: pvjosue dangerous, this might be transposed
    R[:,0,0] = ax*ax*u + c
    R[:,0,1] = ax*ay*u - az*s
    R[:,0,2] = ax*az*u + ay*s
    R[:,1,0] = ay*ax*u + az*s
    R[:,1,1] = ay*ay*u + c
    R[:,1,2] = ay*az*u - ax*s
    R[:,2,0] = az*ax*u - ay*s
    R[:,2,1] = az*ay*u + ax*s
    R[:,2,2] = az*az*u + c
    return R

def voxRayJM(Delta_n, opticAxis, rayDir, ell):
    '''Compute Jones matrix associated with a particular ray and voxel combination'''
    n_voxels = opticAxis.shape[0]
    if not torch.is_tensor(opticAxis):
        opticAxis = torch.from_numpy(opticAxis).to(Delta_n.device)
    # Azimuth is the angle of the sloq axis of retardance.
    azim = torch.arctan2(torch.linalg.vecdot(opticAxis , rayDir[1]), torch.linalg.vecdot(opticAxis , rayDir[2])) # todo: pvjosue dangerous, vecdot similar to dot?
    azim[Delta_n==0] = 0
    azim[Delta_n<0] += torch.pi / 2
    # print(f"Azimuth angle of index ellipsoid is {np.around(torch.rad2deg(azim).numpy(), decimals=0)} degrees.")
    ret = abs(Delta_n) * (1 - torch.linalg.vecdot(opticAxis, rayDir[0]) ** 2) * 2 * torch.pi * ell[:n_voxels] / wavelength
    # print(f"Accumulated retardance from index ellipsoid is {np.around(torch.rad2deg(ret).numpy(), decimals=0)} ~ {int(torch.rad2deg(ret).numpy()) % 360} degrees.")
    offdiag = 1j * torch.sin(2 * azim) * torch.sin(ret / 2)
    diag1 = torch.cos(ret / 2) + 1j * torch.cos(2 * azim) * torch.sin(ret / 2)
    diag2 = torch.conj(diag1)
    # Construct Jones Matrix
    JM = torch.zeros([Delta_n.shape[0], 2, 2], dtype=torch.complex64, device=Delta_n.device)
    JM[:,0,0] = diag1
    JM[:,0,1] = offdiag
    JM[:,1,0] = offdiag
    JM[:,1,1] = diag2
    return JM

def rayJM(JMlist, voxels_of_segs):
    '''Computes product of Jones matrix sequence
    Equivalent method: torch.linalg.multi_dot([JM1, JM2])
    '''
    n_rays = len(JMlist[0])
    product = torch.tensor([[1.0,0],[0,1.0]], dtype=torch.complex64, device=JMlist[0].device).unsqueeze(0).repeat(n_rays,1,1)
    for ix,JM in enumerate(JMlist):
        rays_with_voxels = [len(vx)>ix for vx in voxels_of_segs]
        product[rays_with_voxels,...] = product[rays_with_voxels,...] @ JM
    return product

def find_orthogonal_vec(v1, v2):
    '''v1 and v2 are numpy arrays (3d vectors)
    This function accomodates for a divide by zero error.'''
    x = torch.linalg.multi_dot((v1, v2)) / (torch.linalg.norm(v1.unsqueeze(2),dim=1) * torch.linalg.norm(v2))[0]
    # Check if vectors are parallel or anti-parallel
    normal_vec = torch.zeros_like(v1)

    # Search for invalid indices 
    invalid_indices = torch.isclose(x.abs(),torch.ones([1], device=x.device))
    valid_indices = ~invalid_indices
    # Compute the invalid normal_vectors
    if invalid_indices.sum():
        for n_axis in range(3):
            normal_vec[invalid_indices,n_axis] = (v1[invalid_indices,n_axis]==0).float() * 1
            # Turn off fixed indices
            invalid_indices[v1[:,n_axis]==0] = False
        if invalid_indices.sum(): # treat remaning ones
            non_par_vec = torch.tensor([1.0, 0, 0], device=x.device).unsqueeze(0).repeat(v1.shape[0],1)
            C = torch.cross(v1[invalid_indices,:], non_par_vec[invalid_indices,:])
            normal_vec[invalid_indices,:] = C / torch.linalg.norm(C,dim=1)

    # Compute the valid normal_vectors
    normal_vec[valid_indices] = torch.cross(v1[valid_indices], v2.unsqueeze(0).repeat(v1.shape[0],1)[valid_indices]) / torch.linalg.norm(torch.linalg.cross(v1[valid_indices], v2.unsqueeze(0).repeat(v1.shape[0],1)[valid_indices]).unsqueeze(2),dim=1)
    return normal_vec

def calc_rayDir(ray_in):
    '''
    Allows to the calculations to be done in ray-space coordinates
    as oppossed to laboratory coordinates
    Parameters:
        ray_in [n_rays,3] (torch.array): normalized 3D vector giving the direction 
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
    theta = torch.arccos(torch.linalg.multi_dot((ray, torch.tensor([1.0,0,0] ,dtype=ray.dtype, device=ray_in.device))))
    # Unit vectors that give the laboratory axes, can be changed
    scope_axis = torch.tensor([1.0,0,0],dtype=ray.dtype, device=ray_in.device)
    scope_perp1 = torch.tensor([0,1.0,0],dtype=ray.dtype, device=ray_in.device)
    scope_perp2 = torch.tensor([0,0,1.0],dtype=ray.dtype, device=ray_in.device)
    # print(f"Rotating by {np.around(torch.rad2deg(theta).numpy(), decimals=0)} degrees")
    normal_vec = find_orthogonal_vec(ray, scope_axis)
    Rinv = rotation_matrix(normal_vec, -theta)
    # Extracting basis vectors that are orthogonal to the ray and will be parallel
    # to the laboratory axes that are not the optic axis after a rotation.
    # Note: If scope_perp1 if the y-axis, then ray_perp1 if the 2nd column of Rinv.
    if scope_perp1[0]==0 and scope_perp1[1]==1 and scope_perp1[2]==0:
        ray_perp1 = Rinv[:,:,1] # dot product needed
    else: 
        # todo: we need to put a for loop to do this operation
        # ray_perp1 = torch.linalg.multi_dot((Rinv, scope_perp1))
        raise NotImplementedError
    if scope_perp2[0]==0 and scope_perp2[1]==0 and scope_perp2[2]==1:
        ray_perp2 = Rinv[:,:,2]
    else: 
        # todo: we need to put a for loop to do this operation
        # ray_perp2 = torch.linalg.multi_dot((Rinv, scope_perp2))
        raise NotImplementedError
    
    # Returns a list size 3, where each element is a torch tensor shaped [n_rays, 3]
    return [ray, ray_perp1, ray_perp2]

def calc_retardance(JM):
    '''Calculates the retardance magnitude in radians from a Jones matrix
    Parameters:
        JM (torch.tensor): 2x2 Jones matrix
    Returns:
        retardance (float): retardance magnitude in radians
    '''
    diag_sum = JM[:,0, 0] + JM[:,1, 1]
    diag_diff = JM[:,1, 1] - JM[:,0, 0]
    off_diag_sum = JM[:,0, 1] + JM[:,1, 0]
    # Note: np.arctan(1j) and np.arctan(-1j) gives an divide by zero error
    value = torch.sqrt((off_diag_sum / diag_sum) ** 2 + (diag_diff / diag_sum) ** 2)
    value[torch.isnan(value)] = 0
    value[value.abs()==1] = 0
    arctan = torch.arctan(1j * torch.sqrt((off_diag_sum / diag_sum) ** 2 + (diag_diff / diag_sum) ** 2))
    retardance = 2 * torch.real(arctan)
    return retardance.abs()

def calc_azimuth(JM):
    '''Calculates the retardance azimuth in radians from a Jones matrix
    Parameters:
        JM (torch.tensor): 2x2 Jones matrix
    Returns:
        azimuth (float): azimuth of slow axis orientation in radians 
                            between -3pi/4 and pi/4
    '''
    diag_sum = JM[:,0, 0] + JM[:,1, 1]
    diag_diff = JM[:,1, 1] - JM[:,0, 0]
    off_diag_sum = JM[:,0, 1] + JM[:,1, 0]
    a = (diag_diff / diag_sum).imag
    b = (off_diag_sum / diag_sum).imag
    faulty_index_a = a==0
    faulty_index_b = b==0
    azimuth = torch.pi / 2 - torch.arctan2(b, -a) / 2
    azimuth[faulty_index_a.bitwise_and(faulty_index_b)] = 0
    return azimuth



def main():
    pass


if __name__ == '__main__':
    main()
