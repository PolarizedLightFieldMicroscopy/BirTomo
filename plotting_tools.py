import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def explode(data):
    size = np.array(data.shape)*2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e

def plot_ray_path(ray_entry, ray_exit, colition_indexes, optical_config, data_to_plot=None, colormap='inferno', use_matplotlib=False):

    # is optical_config a Waveblocks object or a dictionary?
    wave_blocks_found = True
    try:
        from waveblocks.blocks.optic_config import OpticConfig
    except:
        wave_blocks_found = False
    if wave_blocks_found and isinstance(optical_config, OpticConfig):
        volume_shape = optical_config.volume_config.volume_shape
        volume_size_um = optical_config.volume_config.volume_size_um
        [dz, dxy, dxy] = optical_config.volume_config.voxel_size_um
    else:
        try:
            volume_shape = optical_config['volume_shape']
            volume_size_um = optical_config['volume_size_um']
            [dz, dxy, dxy] = optical_config['voxel_size']
        except:
            print('Error in plot_ray_path: optical_config should be either a waveblock.OpticConfig or a dictionary containing the required variables...')
            return

    z1,y1,x1 = ray_entry
    z2,y2,x2 = ray_exit
    offset = 0
    z_indices = np.array([x for (x,y,z) in colition_indexes])
    y_indices = np.array([y for (x,y,z) in colition_indexes])
    x_indices = np.array([z for (x,y,z) in colition_indexes])

    # Create box around volume
    voxels = np.zeros(volume_shape)

    # Define grid 
    z_coords,y_coords,x_coords = np.indices(np.array(voxels.shape) + 1).astype(float)
    
    x_coords += 0.5
    y_coords += 0.5
    z_coords += 0.5
    x_coords *= dxy
    y_coords *= dxy
    z_coords *= dz

    voxels[z_indices,y_indices,x_indices] = 1

    # Fast rendering with matplotlib
    if use_matplotlib:
        
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
        

        # Draw entry and exit point
        fig = go.Figure(data=go.Scatter3d(x=[z1,z2],y=[y1,y2],z=[x1,x2],
            marker=dict(
            size=12,
            color='blue',  # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            ),
            line=dict(
            width=3,
            color='blue',  # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            )))

        # Draw the whole volume span
        fig.add_mesh3d(
                # 8 vertices of a cube
                x=[0, 0, volume_size_um[0], volume_size_um[0], 0, 0, volume_size_um[0], volume_size_um[0]],
                y=[0, volume_size_um[1], volume_size_um[1], 0, 0, volume_size_um[1], volume_size_um[1], 0],
                z=[0, 0, 0, 0, volume_size_um[2], volume_size_um[2], volume_size_um[2], volume_size_um[2]],
                colorbar_title='z',
                colorscale='inferno',
                opacity=0.1,
                # Intensity of each vertex, which will be interpolated and color-coded
                intensity = np.linspace(0, 1, 8, endpoint=True),
                # i, j and k give the vertices of triangles
                i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            )
        
        # Draw all the voxels
        cmap = matplotlib.cm.get_cmap(colormap)
        for vix in range(len(z_indices)):
            
            voxel_color = 0.5 / len(z_indices)
            opacity = data_to_plot[vix] / max(data_to_plot)
            if data_to_plot is not None:
                rgba = cmap(opacity)
                voxel_color = f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})'
            offset = 0
            voxel_coord_low = [(z_indices[vix]+offset)*dz, (y_indices[vix]+offset)*dxy, (x_indices[vix]+offset)*dxy]
            offset = 1
            voxel_coord_top = [(z_indices[vix]+offset)*dz, (y_indices[vix]+offset)*dxy, (x_indices[vix]+offset)*dxy]
            fig.add_mesh3d(
                # 8 vertices of a cube
                x=[voxel_coord_low[0], voxel_coord_low[0], voxel_coord_top[0], voxel_coord_top[0], voxel_coord_low[0], voxel_coord_low[0], voxel_coord_top[0], voxel_coord_top[0]],
                y=[voxel_coord_low[1], voxel_coord_top[1], voxel_coord_top[1], voxel_coord_low[1], voxel_coord_low[1], voxel_coord_top[1], voxel_coord_top[1], voxel_coord_low[1]],
                z=[voxel_coord_low[2], voxel_coord_low[2], voxel_coord_low[2], voxel_coord_low[2], voxel_coord_top[2], voxel_coord_top[2], voxel_coord_top[2], voxel_coord_top[2]],
                alphahull=5,
                opacity=opacity/2,
                color=voxel_color)
        
        
        fig.update_layout(
        scene = dict(
                    xaxis = dict(nticks=volume_shape[0], range=[0, volume_size_um[0]]),
                    yaxis = dict(nticks=volume_shape[1], range=[0, volume_size_um[1]]),
                    zaxis = dict(nticks=volume_shape[2], range=[0, volume_size_um[2]]),
                    xaxis_title='Axial dimension',),
        # width=700,
        margin=dict(r=0, l=0, b=0, t=0)
        )
        # Disable legend and colorbar
        fig.update_traces(showlegend=False)
        fig.update_coloraxes(showscale=False)
        fig.update(layout_coloraxis_showscale=False)
    fig.show()


def plot_birefringence_lines(retardance_img, azimuth_img, origin='lower', upscale=1, cmap='Wistia_r', line_color='blue', ax=None):
    # TODO: don't plot if retardance is zero
    # Get pixel coords
    s_i,s_j = retardance_img.shape
    ii,jj = np.meshgrid(np.arange(s_i)*upscale, np.arange(s_j)*upscale)
    
    upscale = np.ones_like(retardance_img)
    upscale *= 0.75
    upscale[retardance_img==0] = 0
    
    l_ii = (ii - 0.5*upscale*np.cos(azimuth_img)).flatten()
    h_ii = (ii + 0.5*upscale*np.cos(azimuth_img)).flatten()

    l_jj = (jj - 0.5*upscale*np.sin(azimuth_img)).flatten()
    h_jj = (jj + 0.5*upscale*np.sin(azimuth_img)).flatten()
    
    lc_data = [[(l_ii[ix], l_jj[ix]), (h_ii[ix], h_jj[ix])] for ix in range(len(l_ii))]
    colors = retardance_img.flatten()
    cmap = matplotlib.cm.get_cmap(cmap)
    rgba = cmap(colors/(2*np.pi))

    lc = matplotlib.collections.LineCollection(lc_data, colors=line_color, linewidths=1)
    if ax is None:
        fig,ax = plt.subplots()
    im = ax.imshow(retardance_img, origin='lower', cmap=cmap)
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    return im


def plot_birefringence_colorized(retardance_img, azimuth_img):
    # Get pixel coords
    colors = np.zeros([azimuth_img.shape[0], azimuth_img.shape[0], 3])
    A = azimuth_img * 1
    # A = np.fmod(A,np.pi)
    colors[:,:,0] = A / A.max()
    colors[:,:,1] = 0.5
    colors[:,:,2] = retardance_img / retardance_img.max()

    colors[np.isnan(colors)] = 0

    from matplotlib.colors import hsv_to_rgb
    rgb = hsv_to_rgb(colors)
    
    plt.imshow(rgb, cmap='hsv')