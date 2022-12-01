import numpy as np
from math import floor, ceil

def siddon_params(start, stop, vox_pitch, vox_count):
    x1, y1, z1 = start
    x2, y2, z2 = stop
    dx, dy, dz = vox_pitch
    pix_numx, pix_numy, pix_numz = vox_count

    # Compute starting and ending parametric values in each dimension
    ray_diff = stop - start
    print(f"Difference between start and stop: {ray_diff}")
    if np.all(ray_diff):
        a_0 = - start / ray_diff
        a_N = (np.array(vox_count) * np.array(vox_pitch) - start) / ray_diff
    else:
        # Start and stop rays are on the same x, y, or z plane
        a_0 = np.zeros(3)
        a_N = np.zeros(3)
        for i in range(len(ray_diff)):
            if ray_diff[i] == 0:
                a_0[i] = 0
                a_N[i] = 1
            else:
                a_0[i] = - start[i] / ray_diff[i]
                a_N[i] = (np.array(vox_count)[i] * np.array(vox_pitch)[i] - start[i]) / ray_diff[i]
    # Calculate absolute max and min parametric values
    a_min = max(0, max([min(a_0[i], a_N[i]) for i in range(3)]))
    a_max = min(1, min([max(a_0[i], a_N[i]) for i in range(3)]))

    #now find range of indices corresponding to max/min a values
    if (x2 - x1) >= 0:
        i_min = ceil(pix_numx - (pix_numx*dx - a_min*(x2 - x1) - x1)/dx)
        i_max = floor((x1 + a_max*(x2 - x1))/dx)
    elif (x2 - x1) < 0:
        i_min = ceil(pix_numx - (pix_numx*dx - a_max*(x2 - x1) - x1)/dx)
        i_max = floor((x1 + a_min*(x2 - x1))/dx)

    if (y2 - y1) >= 0:
        j_min = ceil(pix_numy - (pix_numy*dy - a_min*(y2 - y1) - y1)/dy)
        j_max = floor((y1 + a_max*(y2 - y1))/dy)
    elif (y2 - y1) < 0:
        j_min = ceil(pix_numy - (pix_numy*dy - a_max*(y2 - y1) - y1)/dy)
        j_max = floor((y1 + a_min*(y2 - y1))/dy)
    
    if (z2 - z1) >= 0:
        k_min = ceil(pix_numz - (pix_numz*dz - a_min*(z2 - z1) - z1)/dz)
        k_max = floor((z1 + a_max*(z2 - z1))/dz)
    elif (z2 - z1) < 0:
        k_min = ceil(pix_numz - (pix_numz*dz - a_max*(z2 - z1) - z1)/dz)
        k_max = floor((z1 + a_min*(z2 - z1))/dz)

    #next calculate the list of parametric values for each coordinate
    a_x = []
    if (x2 - x1) > 0:
        for i in range(i_min, i_max):
            a_x.append((i*dx - x1)/(x2 - x1))
    elif (x2 - x1) < 0:
        for i in range(i_min, i_max):
            a_x.insert(0, (i*dx - x1)/(x2 - x1))

    a_y = []
    if (y2 - y1) > 0:
        for j in range(j_min, j_max):
            a_y.append((j*dy - y1)/(y2 - y1))
    elif (y2 - y1) < 0:
        for j in range(j_min, j_max):
            a_y.insert(0, (j*dy - y1)/(y2 - y1))

    a_z = []
    if (z2 - z1) > 0:
        for k in range(k_min, k_max):
            a_z.append((k*dz - z1)/(z2 - z1))
    elif (z2 - z1) < 0:
        for k in range(k_min, k_max):
            a_z.insert(0, (k*dz - z1)/(z2 - z1))

    #finally, form the list of parametric values
    a_list = [a_min] + a_x + a_y + a_z + [a_max]
    a_list = list(set(a_list))
    a_list.sort()
    return a_list

def siddon_midpoints(start, stop, a_list):
    '''Calculates the midpoints of the ray sections that intersect each voxel'''
    # loop though, computing midpoints for each adjacent pair of a values for chosen coord
    mids = []
    for m in range(1, len(a_list)):
        (x, y, z) = 0.5 * (a_list[m] + a_list[m - 1]) * (stop - start) + start
        mids.append((x, y, z))
    return mids

def vox_indices(midpoints, vox_pitch):
    '''Identifies the voxels for which the midpoints belong by converting to 
    voxel units, then rounding down to get the voxel index used we are using
    to refer to the voxel'''
    dx, dy, dz = vox_pitch
    i_voxels = []
    for (x,y,z) in midpoints:
        i_voxels.append((int(x / dx), int(y / dy), int(z / dz)))
    return i_voxels

def siddon_lengths(start, stop, a_list):
    '''Finds length of intersections by multiplying difference in parametric 
    values by entire ray length'''
    entire_length = np.linalg.norm(stop - start)
    lengths = []
    for m in range(1, len(a_list)):
        lengths.append(entire_length * (a_list[m] - a_list[m - 1]))
    return lengths

def siddon(start, stop, voxel_size, volume_shape):
    siddon_list = siddon_params(start, stop, voxel_size, volume_shape)
    seg_mids = siddon_midpoints(start, stop, siddon_list)
    voxels_of_segs = vox_indices(seg_mids, voxel_size)
    ell_in_voxels = siddon_lengths(start, stop, siddon_list)
    return voxels_of_segs, ell_in_voxels