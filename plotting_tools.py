import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def explode(data):
    size = np.array(data.shape)*2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e

def plot_ray_path(ray_entry, ray_exit, colition_indexes, midpoints, mla_info):

    [dz, dxy, dxy] = mla_info.vox_pitch

    z1,y1,x1 = ray_entry
    z2,y2,x2 = ray_exit
    offset = 0.5
    z_indices = np.array([x for (x,y,z) in colition_indexes])
    y_indices = np.array([y for (x,y,z) in colition_indexes])
    x_indices = np.array([z for (x,y,z) in colition_indexes])

    # Create box around volume
    voxels = np.zeros((mla_info.n_voxels_z,mla_info.n_voxels_xy,mla_info.n_voxels_xy))

    # Define grid 
    z_coords,y_coords,x_coords = np.indices(np.array(voxels.shape) + 1).astype(float)
    
    x_coords += 0.5
    y_coords += 0.5
    z_coords += 0.5
    x_coords *= dxy
    y_coords *= dxy
    z_coords *= dz

    voxels[z_indices,y_indices,x_indices] = 1
    # Fast rendering
    if False:
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter((z_indices+offset)*dxy,(y_indices+offset)*dxy,(x_indices+offset)*dxy, s=dxy)
        ax.scatter(z1,y1,x1, c='red')
        ax.scatter(z2,y2,x2, c='green')

        facecolor = '#FF00000F'
        edgecolor = '#FF0000FF'
        voxels[z_indices,y_indices,x_indices] = 1

        facecolors = np.where(voxels==1, facecolor, '#0000000F')
        edgecolors = np.where(voxels==1, edgecolor, '#0000000F')

        ax.voxels(z_coords, y_coords, x_coords, voxels, facecolors=facecolors, edgecolors=edgecolors)
        ax.plot([z1,z2],[y1,y2],[x1,x2])
        plt.xlabel('Axial')
        plt.ylabel('Y axis')
        # show backward mesh?
        # ax.voxels(z_coords, y_coords, x_coords, voxels+1, facecolors='#00FF000F', edgecolors='#0000000F')
        # plt.savefig('output.png')
        plt.show()
    else:
        import plotly.graph_objects as go
        
        fig = go.Figure(data=go.Volume(
            x=z_coords[:-1,:-1,:-1].flatten(),
            y=y_coords[:-1,:-1,:-1].flatten(),
            z=x_coords[:-1,:-1,:-1].flatten(),
            value=voxels.flatten(),
            isomin=0,
            isomax=0.1,
            opacity=0.1, # needs to be small to see through all surfaces
            surface_count=1, # needs to be a large number for good volume rendering
            ))
        fig.add_scatter3d(  x=(z_indices+offset)*dz,
                            y=(y_indices+offset)*dxy,
                            z=(x_indices+offset)*dxy)
        
        fig.add_scatter3d(x=[z1,z2],y=[y1,y2],z=[x1,x2])
        
        fig.update_layout(
        scene = dict(
                    xaxis = dict(nticks=mla_info.n_voxels_z, range=[0,mla_info.z_span],),
                    yaxis = dict(nticks=mla_info.n_voxels_xy, range=[0,mla_info.xy_span]),
                    zaxis = dict(nticks=mla_info.n_voxels_xy, range=[0,mla_info.xy_span]),
                    xaxis_title='Axial dimension',),
        # width=700,
        # margin=dict(r=20, l=10, b=10, t=10)
        )
    fig.show()


def plot_rays_at_sample(ray_entry, ray_exit, colormap='inferno', mla_info=None):

    i_shape,j_shape = ray_entry.shape[1:]

    # Grab all rays
    all_entry = np.reshape(ray_entry,[ray_entry.shape[0],i_shape*j_shape])
    all_exit = np.reshape(ray_exit,[ray_entry.shape[0],i_shape*j_shape])
    x_entry,y_entry,z_entry = all_entry[1,:],all_entry[2,:],all_entry[0,:]
    x_exit,y_exit,z_exit = all_exit[1,:],all_exit[2,:],all_exit[0,:]

    # grab the ray index to color them
    ray_index = list(range(len(x_exit)))
    # And plot them
    plt.clf()
    ax = plt.subplot(1,3,1)
    plt.scatter(x_entry, y_entry, c=ray_index, cmap=colormap)
    ax.set_box_aspect(1)
    plt.title('entry rays coords')
    ax = plt.subplot(1,3,2)
    plt.scatter(x_exit, y_exit, c=ray_index, cmap=colormap)
    ax.set_box_aspect(1)
    plt.title('exit rays coords')
    ax = plt.subplot(1,3,3, projection='3d')
    for ray_ix in range(len(x_entry)):
        cmap = matplotlib.cm.get_cmap(colormap)
        rgba = cmap(ray_ix/len(x_entry))
        plt.plot([x_entry[ray_ix],x_exit[ray_ix]],[y_entry[ray_ix],y_exit[ray_ix]],[z_entry[ray_ix],z_exit[ray_ix]], color=rgba)

    # Add area covered by MLAs
    if mla_info is not None:
        m = mla_info.xy_span/2
        mz = mla_info.z_span/2
        n_mlas = mla_info.n_mlas//2
        mla_sample_pitch = mla_info.pitch / mla_info.obj_M
        x = [m-n_mlas*mla_sample_pitch,m+n_mlas*mla_sample_pitch,m+n_mlas*mla_sample_pitch,m-n_mlas*mla_sample_pitch]
        y = [m-n_mlas*mla_sample_pitch,m-n_mlas*mla_sample_pitch,m+n_mlas*mla_sample_pitch,m+n_mlas*mla_sample_pitch]
        z = [mz,mz,mz,mz]
        verts = [list(zip(x,y,z))]
        ax.add_collection3d(Poly3DCollection(verts,alpha=.20))

    # ax.set_box_aspect((1,1,5))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()