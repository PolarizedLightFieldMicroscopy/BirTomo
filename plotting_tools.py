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


def plot_rays_at_sample(ray_entry, ray_exit, colormap='inferno', optical_config=None, use_matplotlib=False):

    i_shape,j_shape = ray_entry.shape[1:]

    # Grab all rays
    all_entry = np.reshape(ray_entry,[ray_entry.shape[0],i_shape*j_shape])
    all_exit = np.reshape(ray_exit,[ray_entry.shape[0],i_shape*j_shape])
    x_entry,y_entry,z_entry = all_entry[1,:],all_entry[2,:],all_entry[0,:]
    x_exit,y_exit,z_exit = all_exit[1,:],all_exit[2,:],all_exit[0,:]

    # grab the ray index to color them
    ray_index = list(range(len(x_exit)))
    if use_matplotlib:
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
        if optical_config is not None:
            mz = optical_config.volume_config.volume_size_um[0]/2
            m = optical_config.volume_config.volume_size_um[1]/2
            n_mlas = optical_config.mla_config.n_mlas//2
            mla_sample_pitch = optical_config.mla_config.pitch / optical_config.PSF_config.M
            x = [m-n_mlas*mla_sample_pitch,m+n_mlas*mla_sample_pitch,m+n_mlas*mla_sample_pitch,m-n_mlas*mla_sample_pitch]
            y = [m-n_mlas*mla_sample_pitch,m-n_mlas*mla_sample_pitch,m+n_mlas*mla_sample_pitch,m+n_mlas*mla_sample_pitch]
            z = [mz,mz,mz,mz]
            verts = [list(zip(x,y,z))]
            ax.add_collection3d(Poly3DCollection(verts,alpha=.20))

        # ax.set_box_aspect((1,1,5))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    else:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import plotly.express as px

        # Plot entry and exit?
        if False:
            fig = make_subplots(rows=1, cols=3,
                                specs=[[{'is_3d': False}, {'is_3d': False}, {'is_3d': True}]],
                                subplot_titles=("Entry rays coords", "Exit rays coords", "Rays through volume"),
                                print_grid=True)
            fig.update_layout(autosize=True, scene=dict(aspectratio = dict( x=1, y=1, z=1 ), aspectmode = 'manual'))#height=300, width=900)
            fig.append_trace(go.Scatter(x=x_entry, y=y_entry, mode='markers', marker=dict(color=ray_index, colorscale=colormap)), row=1, col=1)
            fig.append_trace(go.Scatter(x=x_exit, y=y_exit, mode='markers', marker=dict(color=ray_index, colorscale=colormap)), row=1, col=2)

            # Plot rays
            for ray_ix in range(len(x_entry)):
                cmap = matplotlib.cm.get_cmap(colormap)
                rgba = cmap(ray_ix/len(x_entry))
                if not np.isnan(x_entry[ray_ix]) and not np.isnan(x_exit[ray_ix]):
                    fig.append_trace(go.Scatter3d(x=[x_entry[ray_ix],x_exit[ray_ix]], y=[y_entry[ray_ix],y_exit[ray_ix]],z=[z_entry[ray_ix],z_exit[ray_ix]],
                        marker=dict(color=rgba, size=4), 
                        line=dict(color=rgba))
                    , row=1, col=3)
        else:
            fig = make_subplots(rows=1, cols=1,
                                specs=[[{'is_3d': True}]],
                                subplot_titles=("Rays through volume",),
                                print_grid=True)
            # Gather all rays in single arrays, to plot them all at once, placing NAN in between them
            # Prepare colormap
            all_x = np.empty((3*len(x_entry)))
            all_x[::3] = x_entry
            all_x[1::3] = x_exit
            all_x[2::3] = np.NaN

            all_y = np.empty((3*len(y_entry)))
            all_y[::3] = y_entry
            all_y[1::3] = y_exit
            all_y[2::3] = np.NaN

            all_z = np.empty((3*len(z_entry)))
            all_z[::3] = z_entry
            all_z[1::3] = z_exit
            all_z[2::3] = np.NaN

            # prepare colors for each line
            rgba = [ray_ix/len(all_x) for ray_ix in range(len(all_x))] 
            # Draw the lines and markers
            fig.append_trace(go.Scatter3d(z=all_x, y=all_y, x=all_z,
                marker=dict(color=rgba, colorscale=colormap, size=4), 
                line=dict(color=rgba, colorscale=colormap, ), 
                connectgaps=False, mode='lines+markers'
                ),
                row=1, col=1)
            fig.update_layout(
            scene = dict(
                        xaxis_title='Axial dimension',),
            # width=700,
            margin=dict(r=0, l=0, b=0, t=0)
            )

        fig.show()



def plot_birefringence_lines(retardance_img, azimuth_img, upscale=1, cmap='Wistia_r', line_color='blue', ax=None):
    # Get pixel coords
    s_i,s_j = retardance_img.shape
    ii,jj = np.meshgrid(np.arange(s_i)*upscale, np.arange(s_j)*upscale)
    upscale *= 0.75
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
    im = ax.imshow(retardance_img, cmap=cmap)
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    return im


def plot_birefringence_colorized(retardance_img, azimuth_img, ax=None):
    # Get pixel coords
    colors = 0.5*np.ones([azimuth_img.shape[0], azimuth_img.shape[0], 3])
    A = azimuth_img * 1
    A = np.fmod(A,np.pi)
    colors[:,:,0] = A / A.max()
    colors[:,:,2] = retardance_img / retardance_img.max()
    colors[np.isnan(colors)] = 0
    # Back to original size
    if ax is None:
        fig,ax = plt.subplots()
    im = ax.imshow(colors, cmap='hsv')
    return im