import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_iteration_update(
    vol_meas, ret_meas, azim_meas,
    vol_current, ret_current, azim_current,
    losses, data_term_losses, regularization_term_losses,
    streamlit_purpose=False
):
    if streamlit_purpose:
        fig = plt.figure(figsize=(18, 9))
        plt.rcParams['image.origin'] = 'lower'

    # Plot measurements
    plt.subplot(2, 4, 1)
    plt.imshow(vol_meas, cmap='plasma')
    plt.colorbar()
    plt.title('Ground truth volume (MIP)', weight='bold')
    plt.subplot(2, 4, 2)
    plt.imshow(ret_meas, cmap='plasma')
    plt.colorbar()
    plt.title('Measured retardance')
    plt.subplot(2, 4, 3)
    plt.imshow(azim_meas, cmap='twilight')
    plt.colorbar()
    plt.title('Measured orientation')

    # Plot predictions
    plt.subplot(2, 4, 5)
    plt.imshow(vol_current, cmap='plasma')
    plt.colorbar()
    plt.title('Predicted volume (MIP)', weight='bold')
    plt.subplot(2, 4, 6)
    plt.imshow(ret_current, cmap='plasma')
    plt.colorbar()
    plt.title('Retardance of predicted volume')
    plt.subplot(2, 4, 7)
    plt.imshow(azim_current, cmap='twilight')
    plt.colorbar()
    plt.title('Orientation of predicted volume')

    # Plot losses
    plt.subplot(3, 4, 4)
    plt.plot(list(range(len(losses))), data_term_losses)
    plt.gca().yaxis.set_label_position("right")
    plt.gca().yaxis.tick_right()
    plt.ylabel('Data term loss')
    plt.gca().xaxis.set_visible(False)

    plt.subplot(3, 4, 8)
    plt.plot(list(range(len(losses))), regularization_term_losses)
    plt.gca().yaxis.set_label_position("right")
    plt.gca().yaxis.tick_right()
    plt.ylabel('Regularization term loss')
    plt.gca().xaxis.set_visible(False)

    plt.subplot(3, 4, 12)
    plt.plot(list(range(len(losses))), losses)
    plt.gca().yaxis.set_label_position("right")
    plt.gca().yaxis.tick_right()
    plt.xlabel('Epoch')
    plt.ylabel('Total loss')

    if streamlit_purpose:
        return fig
    else:
        return None


def plot_est_iteration_update(
    vol_current, ret_current, azim_current,
    losses, data_term_losses, regularization_term_losses,
    streamlit_purpose=False
):
    if streamlit_purpose:
        fig = plt.figure(figsize=(18, 9))
        plt.rcParams['image.origin'] = 'lower'

    # Plot predictions
    plt.subplot(2, 4, 5)
    plt.imshow(vol_current)
    plt.colorbar()
    plt.title('Predicted volume (MIP)', weight='bold')
    plt.subplot(2, 4, 6)
    plt.imshow(ret_current)
    plt.colorbar()
    plt.title('Retardance of predicted volume')
    plt.subplot(2, 4, 7)
    plt.imshow(azim_current)
    plt.colorbar()
    plt.title('Orientation of predicted volume')

    plt.subplot(3, 4, 12)
    plt.title('Loss')
    plt.plot(list(range(len(losses))), losses,
             color='green', label='total loss')
    plt.plot(list(range(len(losses))), data_term_losses,
             color='red', label='data fidelity')
    plt.plot(list(range(len(losses))), regularization_term_losses,
             color='blue', label='regularization')
    plt.gca().yaxis.set_label_position("right")
    plt.gca().yaxis.tick_right()
    plt.xlabel('Iterations')
    plt.ylabel('Total loss')
    plt.legend()

    if streamlit_purpose:
        return fig
    else:
        return None


def plot_volume_subplot(index, volume, title):
    """Helper function to plot a volume subplot."""
    ax = plt.subplot(2, 4, index)
    im = ax.imshow(volume)
    plt.colorbar(im, ax=ax)
    plt.title(title, weight='bold')
    plt.axis('off')  # Optionally turn off the axis


def plot_image_subplot(ax, image, title, cmap='plasma'):
    """Helper function to plot an image in a subplot with a colorbar and title."""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    im = ax.imshow(image, cmap=cmap)
    plt.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=8)
    ax.axis('off')  # Hide the axis for a cleaner look
    ax.xaxis.set_visible(False)  # Hide the x-axis if not needed


def plot_combined_loss_subplot(ax, losses, data_term_losses, regularization_term_losses):
    """Helper function to plot all losses on a given axis."""
    epochs = list(range(len(losses)))
    ax.plot(epochs, losses, label='total loss', color='g')
    ax.plot(epochs, data_term_losses,
            label='data-fidelity term loss', color='b')
    ax.plot(epochs, regularization_term_losses,
            label='regularization term loss', color=(1.0, 0.92, 0.23))
    ax.set_xlim(left=0)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    ax.legend(loc='upper right')


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
    fig.text(0.5, 0.96, 'Measurements', ha='center',
             va='center', fontsize=10, weight='bold')
    fig.text(0.5, 0.645, 'Predictions', ha='center',
             va='center', fontsize=10, weight='bold')
    fig.text(0.5, 0.33, 'Loss Function', ha='center',
             va='center', fontsize=10, weight='bold')
    # Plot combined losses across the entire bottom row
    ax_combined = fig.add_subplot(gs[2, :])
    plot_combined_loss_subplot(
        ax_combined, losses, data_term_losses, regularization_term_losses)
    # Adjust layout to prevent overlap, leave space for row titles
    plt.subplots_adjust(left=0.05, right=0.91, bottom=0.07, top=0.92)
    # Return the figure object if in Streamlit, else show the plot
    if streamlit_purpose:
        return fig
    else:
        return None
