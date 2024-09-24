import plotly.graph_objects as go
import numpy as np


def initialize_figure(fig=None):
    """Initialize a Plotly figure if it's not provided."""
    if fig is None:
        fig = go.Figure()
    return fig


def get_vol_shape_and_size(optical_info, use_microns=True):
    """Extract volume shape and size based on voxel size and optical information."""
    volume_shape = optical_info["volume_shape"]
    
    # Handle voxel size if not present
    if "voxel_size_um" not in optical_info:
        optical_info["voxel_size_um"] = [1, 1, 1]
        print("Notice: 'voxel_size_um' was not found in optical_info. Size of [1, 1, 1] assigned.")

    voxel_size_um = optical_info["voxel_size_um"]
    volume_size_um = [voxel_size_um[i] * volume_shape[i] for i in range(3)]
    
    # Determine if volume should be in microns or voxel units
    volume_size = volume_size_um if use_microns else volume_shape
        
    if use_microns:
        volume_size = volume_size_um
    else:
        volume_size = volume_shape
        voxel_size_um = [1, 1, 1]

    return volume_shape, volume_size, voxel_size_um


def prepare_scene(volume_shape, volume_size, use_ticks=False):
    """Prepare the scene dictionary for Plotly layout."""
    # Sometimes the volume_shape is causing an error when being used as the nticks parameter
    if use_ticks:
        scene_dict = dict(
            xaxis={"nticks": volume_shape[0], "range": [0, volume_size[0]]},
            yaxis={"nticks": volume_shape[1], "range": [0, volume_size[1]]},
            zaxis={"nticks": volume_shape[2], "range": [0, volume_size[2]]},
            xaxis_title="Axial dimension",
            aspectratio={"x": volume_size[0], "y": volume_size[1], "z": volume_size[2]},
            aspectmode="manual",
        )
    else:
        scene_dict = dict(
            xaxis_title="Axial dimension",
            aspectratio={"x": volume_size[0], "y": volume_size[1], "z": volume_size[2]},
            aspectmode="manual",
        )
    return scene_dict


def get_base_tip_coordinates(optic_axis, delta_n, volume_shape, voxel_length):
    """Calculate the base and tip coordinates for plotting."""
    coords = np.indices(np.array(volume_shape)).astype(float)
    coords_base = [(coords[i] + 0.5) * voxel_length[i] for i in range(3)]
    coords_tip = [(coords[i] + 0.5 + optic_axis[i, ...] * delta_n * 0.75) * voxel_length[i] for i in range(3)]
    return coords_base, coords_tip


def get_coords(volume_shape, voxel_length, use_microns=True):
    """Generate 3D grid coordinates based on the volume shape and voxel size."""
    coords = np.indices(np.array(volume_shape)).astype(float)
    
    # Shift by half a voxel and multiply by voxel size
    if use_microns:
        coords = [(coords[i] + 0.5) * voxel_length[i] for i in range(3)]
    else:
        coords = [(coords[i] + 0.5) for i in range(3)]
    
    return coords


def apply_mask_and_nan(coords_base, coords_tip, delta_n):
    """Mask and set values to NaN for plotting based on zero values in delta_n."""
    mask = delta_n == 0
    for base, tip in zip(coords_base, coords_tip):
        base[mask] = np.nan
        tip[mask] = np.nan
    return coords_base, coords_tip


def compute_colors(x_base, y_base, z_base, x_tip, y_tip, z_tip):
    """Compute color values for plotting lines based on distance between base and tip."""
    all_color = (
        (x_base - x_tip).flatten() ** 2
        + (y_base - y_tip).flatten() ** 2
        + (z_base - z_tip).flatten() ** 2
    )
    all_color -= all_color.min()
    all_color /= all_color.max()
    return all_color


def check_non_zero_values(array, error_msg):
    """Check that the given array has non-zero values and raise an error if not."""
    assert np.any(array != 0), error_msg
