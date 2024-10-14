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
    add_view_separation_lines=True,
    transpose_and_flip=True
):
    """
    Convert a 3D volume to a single 2D Maximum Intensity Projection (MIP) image.

    Args:
    - volume_input (Tensor): The input volume tensor of shape [batch, 1, xDim, yDim, zDim].
    - projection_func (function): Function to use for projection, e.g., torch.sum or torch.max.
    - scaling_factors (tuple): Scaling factors for x, y, and z dimensions respectively.
    - depths_in_channel (bool): If True, include depth as a channel in the output.
    - thresholds (tuple): Minimum and maximum thresholds for volume intensity.
    - normalize (bool): If True, normalize the volume to be between 0 and 1.
    - border_thickness (int): Thickness of the border to be added around projections.
    - add_view_separation_lines (bool): If True, add lines between to the projection views.

    Returns:
    - out_img (Tensor): The resulting 2D MIP image.
    """
    volume = volume_input.detach().abs().clone()

    # Normalize if required
    if normalize:
        volume = safe_normalize(volume)
    # Permute and add channel if required
    if depths_in_channel:
        volume = volume.permute(0, 3, 2, 1).unsqueeze(1)

    # Apply intensity thresholds
    if thresholds != (0.0, 1.0):
        vol_min, vol_max = volume.min(), volume.max()
        threshold_min = vol_min + (vol_max - vol_min) * thresholds[0]
        threshold_max = vol_min + (vol_max - vol_min) * thresholds[1]
        volume = torch.clamp(volume, min=threshold_min, max=threshold_max)

    # Prepare new volume size after scaling
    scaled_vol_size = [int(volume.shape[i + 2] * scaling_factors[i]) for i in range(3)]
    batch_size, num_channels = volume.shape[:2]

    # Compute projections
    volume_cpu = volume.float().cpu()
    x_projection = projection_wrapper(volume_cpu, projection_func, dim=2)
    y_projection = projection_wrapper(volume_cpu, projection_func, dim=3)
    z_projection = projection_wrapper(volume_cpu, projection_func, dim=4)

    # Initialize output image with zeros
    out_img = torch.zeros(
        batch_size,
        num_channels,
        scaled_vol_size[0] + scaled_vol_size[2] + border_thickness,
        scaled_vol_size[1] + scaled_vol_size[2] + border_thickness,
    )
    # Place projections into the output image
    out_img[:, :, : scaled_vol_size[0], : scaled_vol_size[1]] = z_projection
    out_img[:, :, scaled_vol_size[0] + border_thickness :, : scaled_vol_size[1]] = (
        F.interpolate(
            x_projection.permute(0, 1, 3, 2),
            size=(scaled_vol_size[2], scaled_vol_size[0]),
            mode="nearest",
        )
    )
    out_img[:, :, : scaled_vol_size[0], scaled_vol_size[1] + border_thickness :] = (
        F.interpolate(
            y_projection, size=(scaled_vol_size[0], scaled_vol_size[2]), mode="nearest"
        )
    )

    # Add white border lines between views
    if add_view_separation_lines:
        line_color = volume.max()
        out_img[:, :, scaled_vol_size[0] : scaled_vol_size[0] + border_thickness, :] = (
            line_color
        )
        out_img[:, :, :, scaled_vol_size[1] : scaled_vol_size[1] + border_thickness] = (
            line_color
        )

    # Transpose and flip the MIP image if specified
    if transpose_and_flip:
        # Transpose the image (swap axes 2 and 3)
        out_img = torch.transpose(out_img, 2, 3)
        # Flip along the 3rd axis (axis=2 after transpose)
        out_img = torch.flip(out_img, dims=[2])

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
