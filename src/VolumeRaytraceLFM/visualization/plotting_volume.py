import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume


def visualize_volume(volume: BirefringentVolume, optical_info: dict, use_microns=True):
    with torch.no_grad():
        plotly_figure = volume.plot_lines_plotly(use_microns=use_microns)
        plotly_figure = volume.plot_volume_plotly(
            optical_info,
            voxels_in=volume.get_delta_n(),
            opacity=0.01,
            fig=plotly_figure,
            use_microns=use_microns,
        )
        plotly_figure.show()
    return


def convert_volume_to_2d_mip(
    volume_input,
    projection_func=torch.max,
    scaling_factors=(1, 1, 1),
    depths_in_channel=True,
    thresholds=(0.0, 1.0),
    normalize=False,
    border_thickness=1,
    add_view_separation_lines=True
):
    """
    Convert a 3D volume to a single 2D Maximum Intensity Projection (MIP) image.

    Args:
    - volume_input (Tensor): The input volume tensor of shape [batch, 1, zDim, yDim, xDim].
    - projection_func (function): Function to use for projection, e.g., torch.sum or torch.max.
    - scaling_factors (tuple): Scaling factors for z, y, and x dimensions respectively.
    - depths_in_channel (bool): If True, include depth as a channel in the output.
    - thresholds (tuple): Minimum and maximum thresholds for volume intensity.
    - normalize (bool): If True, normalize the volume to be between 0 and 1.
    - border_thickness (int): Thickness of the border to be added around projections.
    - add_view_separation_lines (bool): If True, add lines between to the projection views.

    Returns:
    - out_img (Tensor): The resulting 2D MIP image.
    """
    volume = prepare_volume(volume_input, normalize, depths_in_channel)

    # Apply intensity thresholds
    if thresholds != (0.0, 1.0):
        volume = apply_thresholds(volume, thresholds)

    # Prepare new volume size after scaling
    batch_size, num_channels, scaled_vol_size = compute_scaled_volume_size(volume, scaling_factors)

    # Compute projections
    projections = compute_projections(volume, projection_func)

    # Create output image and place projections
    out_img = create_output_image(batch_size, num_channels, scaled_vol_size, projections, border_thickness)

    # Add white border lines between views
    if add_view_separation_lines:
        add_border_lines(out_img, scaled_vol_size, border_thickness, volume.max())

    return out_img


def projection_wrapper(volume, func, dim):
    result = func(volume, dim=dim)
    if isinstance(result, tuple):
        return result[0]  # Assuming the first element is the projection
    return result


def safe_normalize(volume):
    epsilon = 1e-8
    volume_min = volume.min()
    volume_max = volume.max()
    return (volume - volume_min) / (volume_max - volume_min + epsilon)


def prepare_plot_mip(mip_image, img_index=0, plot=True):
    # If the batch size > 1, select the image you want to display, here we select the first image
    single_image_np = mip_image[img_index].squeeze().cpu().numpy()

    if plot:
        # Plot the single image
        plt.imshow(single_image_np, cmap="gray")  # Use a grayscale colormap
        plt.axis("off")  # Turn off axis labels and ticks
        plt.show()
    return single_image_np


def prepare_volume(volume_input, normalize, depths_in_channel):
    """Prepare the volume by detaching, normalizing, and permuting if necessary."""
    volume = volume_input.detach().abs().clone()
    if normalize:
        volume = safe_normalize(volume)
    volume = volume.flip(3)
    if depths_in_channel:
        volume = volume.unsqueeze(1)
    return volume


def apply_thresholds(volume, thresholds):
    """Apply intensity thresholds to the volume."""
    vol_min, vol_max = volume.min(), volume.max()
    threshold_min = vol_min + (vol_max - vol_min) * thresholds[0]
    threshold_max = vol_min + (vol_max - vol_min) * thresholds[1]
    return torch.clamp(volume, min=threshold_min, max=threshold_max)


def compute_scaled_volume_size(volume, scaling_factors):
    """Compute the new volume size after applying scaling factors."""
    scaled_vol_size = [int(volume.shape[i + 2] * scaling_factors[i]) for i in range(3)]
    batch_size, num_channels = volume.shape[:2]
    return batch_size, num_channels, scaled_vol_size


def compute_projections(volume, projection_func):
    """Compute the projections along the x, y, and z axes."""
    volume_cpu = volume.float().cpu()
    proj0 = projection_wrapper(volume_cpu, projection_func, dim=2)
    proj1 = projection_wrapper(volume_cpu, projection_func, dim=3)
    proj2 = projection_wrapper(volume_cpu, projection_func, dim=4)
    return proj0, proj1, proj2


def create_output_image(batch_size, num_channels, scaled_vol_size, projections, border_thickness):
    """Create the output image and insert the projections."""
    proj0, proj1, proj2 = projections
    out_img = torch.zeros(
        batch_size,
        num_channels,
        scaled_vol_size[0] + scaled_vol_size[1] + border_thickness, # height
        scaled_vol_size[0] + scaled_vol_size[2] + border_thickness, # width
    )
    # Placing in top left corner
    out_img[:, :, :scaled_vol_size[1], :scaled_vol_size[2]] = proj0
    # Placing in bottom left corner
    out_img[:, :, scaled_vol_size[1] + border_thickness:, :scaled_vol_size[2]] = F.interpolate(
        proj1, size=(scaled_vol_size[0], scaled_vol_size[2]), mode="nearest"
    )
    # Placing in top right corner
    out_img[:, :, :scaled_vol_size[1], scaled_vol_size[2] + border_thickness:] = F.interpolate(
        proj2.transpose(2, 3).flip(3), size=(scaled_vol_size[1], scaled_vol_size[0]), mode="nearest"
    )
    return out_img


def add_border_lines(out_img, scaled_vol_size, border_thickness, line_color):
    """Add white border lines between the projections."""
    out_img[:, :, scaled_vol_size[2]:scaled_vol_size[2] + border_thickness, :] = line_color
    out_img[:, :, :, scaled_vol_size[1]:scaled_vol_size[1] + border_thickness] = line_color
