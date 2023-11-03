import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from plotting_tools import plot_iteration_update
from VolumeRaytraceLFM.optic_config import volume_2_projections
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume

def convert_volume_to_2d_mip(
    volume_input, 
    projection_func=torch.sum, 
    scaling_factors=(1, 1, 1), 
    depths_in_channel=True, 
    thresholds=(0.0, 1.0), 
    normalize=False, 
    border_thickness=1, 
    add_view_separation_lines=True,
):
    """
    Convert a 3D volume to a single 2D Maximum Intensity Projection (MIP) image.
    
    Parameters:
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
        volume -= volume.min()
        volume /= volume.max()
    
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
    x_projection = projection_func(volume_cpu, dim=2)
    y_projection = projection_func(volume_cpu, dim=3)
    z_projection = projection_func(volume_cpu, dim=4)
    
    # Initialize output image with zeros
    out_img = torch.zeros(
        batch_size, num_channels,
        scaled_vol_size[0] + scaled_vol_size[2] + border_thickness,
        scaled_vol_size[1] + scaled_vol_size[2] + border_thickness
    )
    
    # Place projections into the output image
    out_img[:, :, :scaled_vol_size[0], :scaled_vol_size[1]] = z_projection
    out_img[:, :, scaled_vol_size[0] + border_thickness:, :scaled_vol_size[1]] = \
        F.interpolate(x_projection.permute(0, 1, 3, 2), size=(scaled_vol_size[2], scaled_vol_size[0]), mode='nearest')
    out_img[:, :, :scaled_vol_size[0], scaled_vol_size[1] + border_thickness:] = \
        F.interpolate(y_projection, size=(scaled_vol_size[0], scaled_vol_size[2]), mode='nearest')

    if add_view_separation_lines:
        line_color = volume.max()
        # Add white border lines
        out_img[:, :, scaled_vol_size[0]: scaled_vol_size[0] + border_thickness, :] = line_color
        out_img[:, :, :, scaled_vol_size[1]: scaled_vol_size[1] + border_thickness] = line_color

    return out_img

def prepare_plot_mip(mip_image, img_index=0, plot=True):
    # If the batch size > 1, select the image you want to display, here we select the first image
    single_image_np = mip_image[img_index].squeeze().cpu().numpy()

    if plot:
        # Plot the single image
        plt.imshow(single_image_np, cmap='gray')  # Use a grayscale colormap
        plt.axis('off')  # Turn off axis labels and ticks
        plt.show()

    return single_image_np

def handle_visualization_and_saving(ep, Delta_n_GT, ret_image_measured, azim_image_measured, volume_estimation, figure, output_dir, ret_image_current, azim_image_current, losses, data_term_losses, regularization_term_losses):
    if ep % 10 == 0:
        plt.clf()
        plot_iteration_update(
            volume_2_projections(Delta_n_GT.unsqueeze(0))[0, 0].detach().cpu().numpy(),
            ret_image_measured.detach().cpu().numpy(),
            azim_image_measured.detach().cpu().numpy(),
            volume_2_projections(volume_estimation.get_delta_n().unsqueeze(0))[0, 0].detach().cpu().numpy(),
                ret_image_current.detach().cpu().numpy(),
                azim_image_current.detach().cpu().numpy(),
                losses,
                data_term_losses,
                regularization_term_losses
        )
        figure.canvas.draw()
        figure.canvas.flush_events()
        time.sleep(0.1)
        plt.savefig(f"{output_dir}/Optimization_ep_{'{:02d}'.format(ep)}.pdf")
        time.sleep(0.1)

    if ep % 100 == 0:
        volume_estimation.save_as_file(f"{output_dir}/volume_ep_{'{:02d}'.format(ep)}.h5")
    return

def plot_volume_subplot(index, volume, title):
    """Helper function to plot a volume subplot."""
    ax = plt.subplot(2, 4, index)
    im = ax.imshow(volume)
    plt.colorbar(im, ax=ax)
    plt.title(title, weight='bold')
    plt.axis('off')  # Optionally turn off the axis

def plot_image_subplot(ax, image, title, cmap='plasma'):
    """Helper function to plot an image in a subplot with a colorbar and title."""
    im = ax.imshow(image, cmap=cmap)
    plt.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=8)
    ax.axis('off')  # Hide the axis for a cleaner look

def plot_loss_subplot(ax, losses, title, has_xlabel=True):
    """Helper function to plot loss on a given axis."""
    ax.plot(losses)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel(title)
    if not has_xlabel:
        ax.xaxis.set_visible(False)  # Hide the x-axis if not needed

def plot_combined_loss_subplot(ax, losses, data_term_losses, regularization_term_losses):
    """Helper function to plot all losses on a given axis."""
    epochs = list(range(len(losses)))
    ax.plot(epochs, losses, label='total loss', color='g')
    ax.plot(epochs, data_term_losses, label='data-fidelity term loss', color='b')
    ax.plot(epochs, regularization_term_losses,
            label='regularization term loss', color=(1.0, 0.92, 0.23))
    ax.set_xlim(left=0)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    ax.legend(loc='upper right')

def plot_est_iteration_update(
    vol_current, ret_current, azim_current,
    losses, data_term_losses, regularization_term_losses,
    streamlit_purpose=False
):
    """Plots the current state of the volume, retardance, orientation, and the loss terms during an iteration process."""
    
    if streamlit_purpose:
        plt.figure(figsize=(18, 9))
        plt.rcParams['image.origin'] = 'lower'
    
    # Plot the current state of the volume, retardance, and orientation
    plot_volume_subplot(5, vol_current, 'Predicted volume (MIP)')
    plot_volume_subplot(6, ret_current, 'Retardance of predicted volume')
    plot_volume_subplot(7, azim_current, 'Orientation of predicted volume')

    # Plot the losses
    plot_loss_subplot(losses, data_term_losses, regularization_term_losses)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def plot_iteration_update(
    vol_meas, ret_meas, azim_meas,
    vol_current, ret_current, azim_current,
    losses, data_term_losses, regularization_term_losses,
    streamlit_purpose=False
):
    """Plots measured and predicted volumes, retardance, orientation, and losses in a grid layout."""
    
    # Create figure with specified size and style for Streamlit if needed
    if streamlit_purpose:
        plt.figure(figsize=(18, 9))
        plt.rcParams['image.origin'] = 'lower'
    
    # Plot measured data
    ax1 = plt.subplot(2, 4, 1)
    plot_image_subplot(ax1, vol_meas, 'Ground truth volume (MIP)')
    
    ax2 = plt.subplot(2, 4, 2)
    plot_image_subplot(ax2, ret_meas, 'Measured retardance')
    
    ax3 = plt.subplot(2, 4, 3)
    plot_image_subplot(ax3, azim_meas, 'Measured orientation', cmap='twilight')

    # Plot predictions
    ax4 = plt.subplot(2, 4, 5)
    plot_image_subplot(ax4, vol_current, 'Predicted volume (MIP)')
    
    ax5 = plt.subplot(2, 4, 6)
    plot_image_subplot(ax5, ret_current, 'Retardance of predicted volume')
    
    ax6 = plt.subplot(2, 4, 7)
    plot_image_subplot(ax6, azim_current, 'Orientation of predicted volume', cmap='twilight')

    # Plot losses
    ax7 = plt.subplot(3, 4, 4)
    plot_loss_subplot(ax7, data_term_losses, 'Data term loss', has_xlabel=False)
    
    ax8 = plt.subplot(3, 4, 8)
    plot_loss_subplot(ax8, regularization_term_losses, 'Regularization term loss', has_xlabel=False)
    
    ax9 = plt.subplot(3, 4, 12)
    plot_loss_subplot(ax9, losses, 'Total loss')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Return the figure object if in Streamlit, else show the plot
    if streamlit_purpose:
        return plt.gcf()
    else:
        plt.show()
        x = 5
        return None

def plot_iteration_update_gridspec(
    vol_meas, ret_meas, azim_meas,
    vol_current, ret_current, azim_current,
    losses, data_term_losses, regularization_term_losses,
    figure=None,
    streamlit_purpose=False
):
    """Plots measured and predicted volumes, retardance, orientation,
        and combined losses using GridSpec for layout.
    """
    # If a figure is provided, use it; otherwise, use the current figure
    if figure is not None:
        fig = figure
    else:
        fig = plt.gcf()  # Get the current figure
    # Clear the current figure to ensure we're not plotting over old data
    fig.clf()
    # Create GridSpec layout
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.2, wspace=0.2)
    titles = ['Birefringence (MIP)', 'Retardance', 'Orientation']
    cmaps = ['plasma', 'plasma', 'twilight']
    # Plot measured data and predictions
    for i, (meas, pred, title, cmap) in enumerate(zip([vol_meas, ret_meas, azim_meas], [vol_current, ret_current, azim_current], titles, cmaps)):
        ax_meas = fig.add_subplot(gs[0, i])
        plot_image_subplot(ax_meas, meas, f'{title}', cmap=cmap)
        
        ax_pred = fig.add_subplot(gs[1, i])
        plot_image_subplot(ax_pred, pred, f'{title}', cmap=cmap)
    # Add row titles
    fig.text(0.5, 0.96, 'Measurements', ha='center', va='center', fontsize=10, weight='bold')
    fig.text(0.5, 0.645, 'Predictions', ha='center', va='center', fontsize=10, weight='bold')
    fig.text(0.5, 0.33, 'Loss Function', ha='center', va='center', fontsize=10, weight='bold')
    # Plot combined losses across the entire bottom row
    ax_combined = fig.add_subplot(gs[2, :])
    plot_combined_loss_subplot(ax_combined, losses, data_term_losses, regularization_term_losses)
    # Adjust layout to prevent overlap, leave space for row titles
    plt.subplots_adjust(left=0.05, right=0.91, bottom=0.07, top=0.92)
    # Return the figure object if in Streamlit, else show the plot
    if streamlit_purpose:
        return fig
    else:
        return None

def setup_visualization():
    plt.ion()
    fig_size = (10, 9)
    figure = plt.figure(figsize=fig_size)
    plt.rcParams['image.origin'] = 'lower'
    if False:
        manager = plt.get_current_fig_manager()
        manager.window.setGeometry(4000, 600, 1800, 900)
    return figure

def visualize_volume(volume: BirefringentVolume, optical_info: dict):
    with torch.no_grad():
        plotly_figure = volume.plot_lines_plotly()
        plotly_figure = volume.plot_volume_plotly(optical_info,
                                                voxels_in=volume.get_delta_n(),
                                                opacity=0.01,
                                                fig=plotly_figure
                                                )
        plotly_figure.show()
    return
