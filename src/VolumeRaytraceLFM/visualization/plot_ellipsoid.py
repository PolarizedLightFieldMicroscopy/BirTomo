import numpy as np
import plotly.graph_objects as go


def generate_ellipsoid_volume(volume_shape, center=[0.5, 0.5, 0.5],
                              radius=[10, 10, 10], alpha=0.1, delta_n=0.1):
    ''' generate_ellipsoid_volume: Creates an ellipsoid with optical axis normal to the ellipsoid surface.
        Args:
            Center [3]: [cz,cy,cx] from 0 to 1 where 0.5 is the center of the volume_shape.
            radius [3]: in voxels, the radius in z,y,x for this ellipsoid.
            alpha float: Border thickness.
            delta_n float: Delta_n value of birefringence in the volume
        '''
    # Grabbed from https://math.stackexchange.com/questions/2931909/normal-of-a-point-on-the-surface-of-an-ellipsoid
    vol = np.zeros([4,] + volume_shape)

    kk, jj, ii = np.meshgrid(np.arange(volume_shape[0]), np.arange(
        volume_shape[1]), np.arange(volume_shape[2]), indexing='ij')
    # shift to center
    kk = np.floor(center[0]*volume_shape[0]) - kk.astype(float)
    jj = np.floor(center[1]*volume_shape[1]) - jj.astype(float)
    ii = np.floor(center[2]*volume_shape[2]) - ii.astype(float)

    # DEBUG: checking the indicies
    # np.argwhere(ellipsoid_border == np.min(ellipsoid_border))
    # plt.imshow(ellipsoid_border_mask[int(volume_shape[0] / 2),:,:])
    ellipsoid_border = (
        kk**2) / (radius[0]**2) + (jj**2) / (radius[1]**2) + (ii**2) / (radius[2]**2)
    ellipsoid_border_mask = np.abs(ellipsoid_border-alpha) <= 1
    vol[0, ...] = ellipsoid_border_mask.astype(float)
    # Compute normals
    kk_normal = 2 * kk / radius[0]
    jj_normal = 2 * jj / radius[1]
    ii_normal = 2 * ii / radius[2]
    norm_factor = np.sqrt(kk_normal**2 + jj_normal**2 + ii_normal**2)
    # Avoid division by zero
    norm_factor[norm_factor == 0] = 1
    vol[1, ...] = (kk_normal / norm_factor) * vol[0, ...]
    vol[2, ...] = (jj_normal / norm_factor) * vol[0, ...]
    vol[3, ...] = (ii_normal / norm_factor) * vol[0, ...]
    vol[0, ...] *= delta_n
    # vol = vol.permute(0,2,1,3)

    inner_alpha = 0.5
    outer_mask = np.abs(ellipsoid_border-alpha) <= 1
    inner_mask = ellipsoid_border < inner_alpha

    # Hollowing out the ellipsoid
    combined_mask = np.logical_and(outer_mask, ~inner_mask)

    vol[0, ...] = combined_mask.astype(float)
    return vol


def plot_ellipsoid(vol):
    """ Plots the ellipsoid using Plotly.
    Args:
    - vol (numpy array): The output from generate_ellipsoid_volume.
    """

    # Extract the ellipsoid's border mask (which is in the first channel of vol)
    ellipsoid_mask = vol[0, ...] > 0

    # Extract the x, y, z coordinates of the surface voxels
    z, y, x = np.where(ellipsoid_mask)

    # Create a scatter plot of the surface voxels
    scatter = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=2))

    # Plot
    fig = go.Figure(data=[scatter])
    fig.show()

# def plot_ellipsoid(vol):
#     """ Plots the ellipsoid using Plotly.
#     Args:
#     - vol (numpy array): The output from generate_ellipsoid_volume.
#     """

#     fig = go.Figure(data=go.Volume(
#         x=vol[3, ...].flatten(),
#         y=vol[2, ...].flatten(),
#         z=vol[1, ...].flatten(),
#         value=vol[0, ...].flatten(),
#         isomin=0.1,
#         isomax=0.8,
#         opacity=0.1, # adjust this for visualization clarity
#         surface_count=17, # adjust this based on preference
#         colorscale='Viridis'
#     ))

#     fig.show()


# Example usage
volume_shape = [50, 50, 50]
myshape = [15, 51, 51]
radius = [5.5, 5.5, 3.5]
myradius = [5.5, 9.5, 5.5]
vol = generate_ellipsoid_volume(
    myshape, radius=myradius, center=[0.5, 0.5, 0.5])
plot_ellipsoid(vol)
