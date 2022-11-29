import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def explode(data):
    size = np.array(data.shape)*2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e

def plot_ray_path(ray_entry, ray_exit, colition_indexes, optical_config, use_matplotlib=False):

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
    offset = 0.5
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
        
        fig = go.Figure(data=go.Volume(
            x=z_coords[:-1,:-1,:-1].flatten(),
            y=y_coords[:-1,:-1,:-1].flatten(),
            z=x_coords[:-1,:-1,:-1].flatten(),
            value=voxels.flatten(),
            isomin=0,
            isomax=0.1,
            opacity=0.01, # needs to be small to see through all surfaces
            surface_count=20, # needs to be a large number for good volume rendering
            ))
        fig.add_scatter3d(  x=(z_indices+offset)*dz,
                            y=(y_indices+offset)*dxy,
                            z=(x_indices+offset)*dxy)
        
        fig.add_scatter3d(x=[z1,z2],y=[y1,y2],z=[x1,x2],
            marker=dict(
            size=12,
            color='blue',  # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            ),
            line=dict(
            width=3,
            color='blue',  # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            ))
        
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
            # Plot rays
            for ray_ix in range(len(x_entry)):
                cmap = matplotlib.cm.get_cmap(colormap)
                rgba = cmap(ray_ix/len(x_entry))
                if not np.isnan(x_entry[ray_ix]) and not np.isnan(x_exit[ray_ix]):
                    fig.append_trace(go.Scatter3d(z=[x_entry[ray_ix],x_exit[ray_ix]], y=[y_entry[ray_ix],y_exit[ray_ix]],x=[z_entry[ray_ix],z_exit[ray_ix]], name=f'Ray {ray_ix}',
                        marker=dict(color=rgba,size=4), 
                        line=dict(color=rgba)),
                        row=1, col=1)
        fig.show()