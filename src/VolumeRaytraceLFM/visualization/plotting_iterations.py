import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker


def plot_iteration_update(
    vol_meas,
    ret_meas,
    azim_meas,
    vol_current,
    ret_current,
    azim_current,
    losses,
    data_term_losses,
    regularization_term_losses,
    streamlit_purpose=False,
):
    if streamlit_purpose:
        fig = plt.figure(figsize=(18, 9))
        plt.rcParams["image.origin"] = "upper"

    # Plot measurements
    plt.subplot(2, 4, 1)
    plt.imshow(vol_meas, cmap="plasma")
    plt.colorbar()
    plt.title("Ground truth volume (MIP)", weight="bold")
    plt.subplot(2, 4, 2)
    plt.imshow(ret_meas, cmap="plasma")
    plt.colorbar()
    plt.title("Measured retardance")
    plt.subplot(2, 4, 3)
    plt.imshow(azim_meas, cmap="twilight")
    plt.colorbar()
    plt.title("Measured orientation")

    # Plot predictions
    plt.subplot(2, 4, 5)
    plt.imshow(vol_current, cmap="plasma")
    plt.colorbar()
    plt.title("Predicted volume (MIP)", weight="bold")
    plt.subplot(2, 4, 6)
    plt.imshow(ret_current, cmap="plasma")
    plt.colorbar()
    plt.title("Retardance of predicted volume")
    plt.subplot(2, 4, 7)
    plt.imshow(azim_current, cmap="twilight")
    plt.colorbar()
    plt.title("Orientation of predicted volume")

    # Plot losses
    plt.subplot(3, 4, 4)
    plt.plot(list(range(len(losses))), data_term_losses)
    plt.gca().yaxis.set_label_position("right")
    plt.gca().yaxis.tick_right()
    plt.ylabel("Data term loss")
    plt.gca().xaxis.set_visible(False)

    plt.subplot(3, 4, 8)
    plt.plot(list(range(len(losses))), regularization_term_losses)
    plt.gca().yaxis.set_label_position("right")
    plt.gca().yaxis.tick_right()
    plt.ylabel("Regularization term loss")
    plt.gca().xaxis.set_visible(False)

    plt.subplot(3, 4, 12)
    plt.plot(list(range(len(losses))), losses)
    plt.gca().yaxis.set_label_position("right")
    plt.gca().yaxis.tick_right()
    plt.xlabel("Iterations")
    plt.ylabel("Total loss")

    if streamlit_purpose:
        return fig
    else:
        return None


def plot_est_iteration_update(
    vol_current,
    ret_current,
    azim_current,
    losses,
    data_term_losses,
    regularization_term_losses,
    streamlit_purpose=False,
):
    if streamlit_purpose:
        fig = plt.figure(figsize=(18, 9))
        plt.rcParams["image.origin"] = "upper"

    # Plot predictions
    plt.subplot(2, 4, 5)
    plt.imshow(vol_current)
    plt.colorbar()
    plt.title("Predicted volume (MIP)", weight="bold")
    plt.subplot(2, 4, 6)
    plt.imshow(ret_current)
    plt.colorbar()
    plt.title("Retardance of predicted volume")
    plt.subplot(2, 4, 7)
    plt.imshow(azim_current)
    plt.colorbar()
    plt.title("Orientation of predicted volume")

    plt.subplot(3, 4, 12)
    plt.title("Loss")
    plt.plot(list(range(len(losses))), losses, color="green", label="total loss")
    plt.plot(
        list(range(len(losses))), data_term_losses, color="red", label="data fidelity"
    )
    plt.plot(
        list(range(len(losses))),
        regularization_term_losses,
        color="blue",
        label="regularization",
    )
    plt.gca().yaxis.set_label_position("right")
    plt.gca().yaxis.tick_right()
    plt.xlabel("Iterations")
    plt.ylabel("Total loss")
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
    plt.title(title, weight="bold")
    plt.axis("off")  # Optionally turn off the axis


def plot_image_subplot(ax, image, title, cmap="plasma"):
    """Helper function to plot an image in a subplot with a colorbar and title."""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    im = ax.imshow(image, cmap=cmap)
    plt.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=8)
    ax.axis("off")  # Hide the axis for a cleaner look
    ax.xaxis.set_visible(False)  # Hide the x-axis if not needed
    if title == "Orientation":
        im.set_clim(0, np.pi)


def plot_combined_loss_subplot(
    ax, losses, data_term_losses, regularization_term_losses, max_y_limit=None
):
    """Helper function to plot all losses on a given axis."""
    iterations = list(range(len(losses)))
    ax.plot(iterations, losses, label="total loss", color="g")
    ax.plot(iterations, data_term_losses, label="data-fidelity term loss", color="b")
    ax.plot(
        iterations,
        regularization_term_losses,
        label="regularization term loss",
        color=(1.0, 0.92, 0.23),
    )
    ax.set_xlim(left=0)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.legend(loc="upper right")
    ax.grid(True)

    # Set y-axis limit to zoom in on the lower range of loss values
    if max_y_limit is not None:
        ax.set_ylim([0, max_y_limit])

    # Use scientific notation for the y-axis
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))


def calculate_dynamic_max_y_limit(losses, window_size=10, scale_factor=1.1):
    """Calculate the dynamic max_y_limit based on the recent loss values.

    Args:
        losses (list or np.array): List of total loss values.
        window_size (int): Number of recent iterations to consider.
        scale_factor (float): Factor to scale the maximum loss for the y-axis limit.

    Returns:
        float: Calculated max_y_limit.
    """
    if len(losses) < window_size:
        window_size = len(losses)

    recent_losses = losses[-window_size:]
    max_recent_loss = max(recent_losses)

    return max_recent_loss * scale_factor


def plot_discrepancy_loss_subplot(ax, discrepancy_losses, max_y_limit=None):
    """Plots discrepancy losses on a separate subplot."""
    iterations = list(range(len(discrepancy_losses)))
    ax.plot(iterations, discrepancy_losses, label="discrepancy", color='purple', linestyle="-")
    ax.set_xlim(left=0)
    ax.set_xlabel("iteration")
    ax.set_ylabel("discrepancy")
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    ax.grid(True)

    # Set dynamic y-axis limit
    min_discrepancy = min(discrepancy_losses)
    max_discrepancy = max(discrepancy_losses)
    y_min = min(min_discrepancy * 0.9, max_discrepancy * 0.5)

    # Set y-axis limit to zoom in on the lower range of loss values
    if max_y_limit is not None:
        ax.set_ylim([0, max_y_limit])
    else:
        ax.set_ylim([y_min, max_discrepancy * 1.05])

    # Switch to a logarithmic scale for better visualization of small changes
    ax.set_yscale('log')

    # Use scientific notation for the y-axis
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))


def plot_iteration_update_gridspec(
    vol_meas,
    ret_meas,
    azim_meas,
    vol_current,
    ret_current,
    azim_current,
    losses,
    data_term_losses,
    regularization_term_losses,
    discrepancy_losses=None,
    figure=None,
    streamlit_purpose=False,
):
    """Plots measured and predicted volumes, retardance, orientation,
    and combined losses using GridSpec for layout. Optionally plots discrepancy loss in a separate subplot.
    """
    # If a figure is provided, use it; otherwise, use the current figure
    if figure is not None:
        fig = figure
    else:
        fig = plt.gcf()  # Get the current figure
    # Clear the current figure to ensure we're not plotting over old data
    fig.clf()
    
    # Adjust GridSpec layout: Add extra rows for subheaders
    nrows = 10 if discrepancy_losses is not None else 7
    # Define the height ratios for the rows: smaller for the text rows, larger for the plot rows
    height_ratios = [0.2, 1, 0.2, 1, 0.05, 0.1, 1]
    if discrepancy_losses is not None:
        height_ratios.append(0.15)
        height_ratios.append(0.1)
        height_ratios.append(1)
    
    # Create GridSpec layout with custom height ratios
    gs = gridspec.GridSpec(nrows, 3, figure=fig, hspace=0.2, wspace=0.2, height_ratios=height_ratios)

    titles = ["Birefringence (MIP)", "Retardance", "Orientation"]
    cmaps = ["plasma", "plasma", "twilight"]
    text_params = {
        "ha": "center",
        "va": "center",
        "fontsize": 10,
        "weight": "bold"
    }

    # Plot the 'Measurements' header across all three columns
    ax_measurements_header = fig.add_subplot(gs[0, :])
    ax_measurements_header.text(0.5, 0.5, "Measurements", **text_params)
    ax_measurements_header.axis("off")

    # Plot measured data
    for i, (meas, title, cmap) in enumerate(zip([vol_meas, ret_meas, azim_meas], titles, cmaps)):
        ax_meas = fig.add_subplot(gs[1, i])
        plot_image_subplot(ax_meas, meas, f"{title}", cmap=cmap)

    # Plot the 'Predictions' header across all three columns
    ax_predictions_header = fig.add_subplot(gs[2, :])
    ax_predictions_header.text(0.5, 0.5, "Predictions", **text_params)
    ax_predictions_header.axis("off")

    # Plot predicted data
    for i, (pred, title, cmap) in enumerate(zip([vol_current, ret_current, azim_current], titles, cmaps)):
        ax_pred = fig.add_subplot(gs[3, i])
        plot_image_subplot(ax_pred, pred, f"{title}", cmap=cmap)

    # Plot the 'Loss Function' header
    ax_loss_header = fig.add_subplot(gs[5, :])
    ax_loss_header.text(0.5, 0.5, "Loss Function", **text_params)
    ax_loss_header.axis("off")

    # Calculate dynamic max_y_limit based on recent loss values
    window = max(50, int(len(losses) / 2))
    max_y_limit = calculate_dynamic_max_y_limit(losses, window_size=window, scale_factor=1.1)

    # Plot combined losses across the entire row
    ax_combined = fig.add_subplot(gs[6, :])
    plot_combined_loss_subplot(
        ax_combined,
        losses,
        data_term_losses,
        regularization_term_losses,
        max_y_limit=max_y_limit,
    )

    # If discrepancy losses are provided, create a new subplot for them
    if discrepancy_losses is not None and len(discrepancy_losses) > 0:
        ax_discrepancy_header = fig.add_subplot(gs[8, :])
        ax_discrepancy_header.text(0.5, 0.5, "Discrepancy from Ground Truth", **text_params)
        ax_discrepancy_header.axis("off")
        ax_discrepancy = fig.add_subplot(gs[9, :])  # New subplot in the final row
        # max_y_limit_discrepancy = calculate_dynamic_max_y_limit(
        #     discrepancy_losses, window_size=500, scale_factor=1.1
        # )
        max_y_limit_discrepancy = None
        plot_discrepancy_loss_subplot(
            ax_discrepancy,
            discrepancy_losses,
            max_y_limit=max_y_limit_discrepancy,
        )

    plt.subplots_adjust(left=0.05, right=0.91, bottom=0.07, top=0.98)

    if streamlit_purpose:
        return fig
    else:
        return None
