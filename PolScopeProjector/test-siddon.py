from siddon import *
import matplotlib.pyplot as plt
import numpy as np

# Assemble arguments
x1, y1, z1 = 0, 0, 0
x2, y2, z2 =0.7, 1, 1
dx, dy, dz = 0.1, 0.1, 0.1
nx, ny, nz = 10, 10, 10
args = (x1, y1, z1, x2, y2, z2, dx, dy, dz, 0, nx, 0, ny, 0, nz)
args2 = (x1, y1, z1, x2, y2, z2, dx, dy, dz, nx, ny, nz)

# Test Siddon
a_list = raytrace(*args)
a_list = raytrace2(*args2)
midpoints = midpoints(*(list(args[:6])+[a_list]))
colition_indexes = vol_indexes(midpoints, dx, dy, dz)
lengths = intersect_length(*(list(args[:6])+[a_list]))
# print('Midpoints: ' + str([(round(x[0], 3), round(x[1], 3), round(x[2], 3)) for x in midpoints]))
# print('Lengths: ' + str([round(x, 3) for x in lengths]))
# print(len(midpoints))

x_midpoint = [x for (x,y,z) in midpoints]
y_midpoint = [y for (x,y,z) in midpoints]
z_midpoint = [z for (x,y,z) in midpoints]

x_indeces = [x for (x,y,z) in colition_indexes]
y_indeces = [y for (x,y,z) in colition_indexes]
z_indeces = [z for (x,y,z) in colition_indexes]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x_midpoint,y_midpoint,z_midpoint, )
ax.scatter(x1,y1,z1, c='red')
ax.scatter(x2,y2,z2, c='green')


if False: #simple rendering

    # Create box around volume
    voxels = np.zeros((nx,ny,nz))

    facecolors = np.where(voxels==0, '#00000000', '#7A88CCC0')
    edgecolors = np.where(voxels==0, '#0000000F', '#7A88CCC0')
    filled = voxels + 1
    x_coords,y_coords,z_coords = np.indices(np.array(voxels.shape) + 1).astype(float)
    x_coords *= dx
    y_coords *= dy
    z_coords *= dz
    ax.voxels(x_coords, y_coords, z_coords, filled, facecolors=facecolors, edgecolors=edgecolors)
    # plt.savefig('output.png')
    fig.show()

else:
    def explode(data):
        size = np.array(data.shape)*2
        data_e = np.zeros(size - 1, dtype=data.dtype)
        data_e[::2, ::2, ::2] = data
        return data_e
    # Create box around volume
    voxels = np.zeros((nx,ny,nz))
    # Fill visited voxels
    voxels[x_indeces, y_indeces, z_indeces] = 1
    facecolors = explode(np.where(voxels==0, '#00000000', '#7A88CCC0'))
    edgecolors = explode(np.where(voxels==0, '#00000002', '#7A88CCC0'))
    filled = explode(voxels + 1)
    x_coords,y_coords,z_coords = np.indices(np.array(facecolors.shape) + 1).astype(float)
    x_coords[0::2, :, :] += 0.05
    y_coords[:, 0::2, :] += 0.05
    z_coords[:, :, 0::2] += 0.05
    x_coords[1::2, :, :] += 0.95
    y_coords[:, 1::2, :] += 0.95
    z_coords[:, :, 1::2] += 0.95
    x_coords *= 0.5*dx
    y_coords *= 0.5*dy
    z_coords *= 0.5*dz
    ax.voxels(x_coords, y_coords, z_coords, filled, facecolors=facecolors, edgecolors=edgecolors)
    # plt.savefig('output.png')
    ax.set_xlabel('X')
    ax.set_xlabel('Y')
    ax.set_xlabel('Z')
    fig.show()

    plt.show()