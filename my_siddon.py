import numpy as np
from math import floor, ceil

def siddon_params(start, stop, vox_pitch, vox_count):
    x1, y1, z1 = start
    x2, y2, z2 = stop
    dx, dy, dz = vox_pitch
    pix_numx, pix_numy, pix_numz = vox_count

    #first, compute starting and ending parametric values in each dimension
    if (x2 - x1) != 0:
        ax_0 = -x1/(x2 - x1)
        ax_N = (pix_numx*dx - x1)/(x2 - x1)
    else:
        ax_0 = 0
        ax_N = 1
    if (y2 - y1) != 0:
        ay_0 = -y1/(y2 - y1)
        ay_N = (pix_numy*dy - y1)/(y2 - y1)
    else:
        ay_0 = 0
        ay_N = 1
    if (z2 - z1) != 0:
        az_0 = -z1/(z2 - z1)
        az_N = (pix_numz*dz - z1)/(z2 - z1) #todo: geneva, was this an error? not multiplying by dz
    else:
        az_0 = 0
        az_N = 1
        
    #then calculate absolute max and min parametric values
    a_min = max(0, min(ax_0, ax_N), min(ay_0, ay_N), min(az_0, az_N))
    a_max = min(1, max(ax_0, ax_N), max(ay_0, ay_N), max(az_0, az_N))

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
    i_mid = []
    for m in range(1, len(a_list)):
        (x, y, z) = 0.5 * (a_list[m] + a_list[m - 1]) * (stop - start) + start
        i_mid.append((x, y, z))
    return i_mid

def vox_indices(midpoints, vox_pitch):
    '''Identifies the voxels for which the midpoints belong
        - shifts down by 0.5, then rounds to nearest integer'''
    dx, dy, dz = vox_pitch
    i_mid = []
    for (x,y,z) in midpoints:
        x_ix = round((x-0.5*dx)/dx)
        y_ix = round((y-0.5*dy)/dy)
        z_ix = round((z-0.5*dz)/dz)
        i_mid.append((x_ix,y_ix,z_ix))
    return i_mid

def siddon_lengths(start, stop, a_list):
    '''Finds length of intersections by multiplying difference in parametric 
    values by entire ray length'''
    entire_length = np.linalg.norm(stop - start)
    lengths = []
    for m in range(1, len(a_list)):
        lengths.append(entire_length * (a_list[m] - a_list[m - 1]))
    return lengths
