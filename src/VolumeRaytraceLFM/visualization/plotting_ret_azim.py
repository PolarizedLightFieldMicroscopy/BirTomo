import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.colors import hsv_to_rgb, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_birefringence_lines(
    retardance_img,
    azimuth_img,
    origin="upper",
    upscale=1,
    cmap="Wistia_r",
    line_color="blue",
    ax=None,
):
    # TODO: don't plot if retardance is zero
    # Get pixel coords
    s_i, s_j = retardance_img.shape
    ii, jj = np.meshgrid(np.arange(s_i) * upscale, np.arange(s_j) * upscale)

    upscale = np.ones_like(retardance_img)
    upscale *= 0.75
    upscale[retardance_img == 0] = 0

    l_ii = (ii - 0.5 * upscale * np.cos(azimuth_img)).flatten()
    h_ii = (ii + 0.5 * upscale * np.cos(azimuth_img)).flatten()

    l_jj = (jj - 0.5 * upscale * np.sin(azimuth_img)).flatten()
    h_jj = (jj + 0.5 * upscale * np.sin(azimuth_img)).flatten()

    lc_data = [[(l_ii[ix], l_jj[ix]), (h_ii[ix], h_jj[ix])] for ix in range(len(l_ii))]
    colors = retardance_img.flatten()
    cmap = matplotlib.cm.get_cmap(cmap)
    rgba = cmap(colors / np.pi)

    lc = matplotlib.collections.LineCollection(lc_data, colors=line_color, linewidths=1)
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(retardance_img, origin="upper", cmap=cmap)
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    return im


def hue_map(img):
    """Replaces greyscale with rbg
    Args:
        img (np.array): 2D image
    Returns:
        rgb (np.array): 3D image with RGB meaning
    """
    # Get pixel coords
    colors = np.zeros([img.shape[0], img.shape[0], 3])
    A = img * 1
    # A = np.fmod(A,np.pi)
    colors[:, :, 0] = A / A.max()
    colors[:, :, 1] = 0.5
    colors[:, :, 2] = 1
    colors[np.isnan(colors)] = 0

    from matplotlib.colors import hsv_to_rgb

    rgb = hsv_to_rgb(colors)
    return rgb


def plot_birefringence_colorized(retardance_img, azimuth_img):
    # Get pixel coords
    colors = np.zeros([azimuth_img.shape[0], azimuth_img.shape[0], 3])
    A = azimuth_img * 1
    # A = np.fmod(A,np.pi)
    colors[:, :, 0] = A / A.max()
    colors[:, :, 1] = 0.5
    colors[:, :, 2] = retardance_img / retardance_img.max()

    colors[np.isnan(colors)] = 0

    from matplotlib.colors import hsv_to_rgb

    rgb = hsv_to_rgb(colors)

    plt.imshow(rgb, cmap="hsv")


def plot_hue_map(
    retardance_img,
    azimuth_img,
    ax=None,
    enhance_contrast=False,
    save_path=None,
    dpi=300,
):
    """Plots the overlay of the retardance and orientation images with
    colorbars and optionally saves the image with high DPI."""
    # Note: may want to use interpolation="nearest" to avoid aliasing affects

    def contrast_stretching(img):
        """Performs contrast stretching on an image."""
        p2, p98 = np.percentile(img, (2, 98))
        return np.clip((img - p2) / (p98 - p2), 0, 1)

    if enhance_contrast:
        retardance_img = contrast_stretching(retardance_img)

    # Get pixel coords
    colors = np.zeros([azimuth_img.shape[0], azimuth_img.shape[1], 3])
    A = azimuth_img * 1
    colors[:, :, 0] = A / A.max()
    colors[:, :, 1] = 0.5
    colors[:, :, 2] = retardance_img / retardance_img.max()
    colors[np.isnan(colors)] = 0

    rgb = hsv_to_rgb(colors)
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(rgb, cmap="hsv")
    ax.set_xticks([])
    ax.set_yticks([])

    # Add colorbar for hue (azimuth)
    axins1 = inset_axes(
        ax,
        width="5%",  # width = 5% of parent_bbox width
        height="45%",  # height : 50%
        loc="lower left",
        bbox_to_anchor=(1.05, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cmap_hue = plt.get_cmap("hsv")
    cb1 = plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_hue), cax=axins1, orientation="vertical"
    )
    cb1.set_label("Orientation", rotation=270, labelpad=15)
    # Assuming the data for azimuth ranges from 0 to pi, we normalize this range to 0-1 for the colorbar.
    cb1.set_ticks([0, 0.5, 1])
    cb1.set_ticklabels(["0", r"$\pi/2$", r"$\pi$"])
    axins1.set_title("Hue", fontsize=8)

    # Add colorbar for saturation (retardance)
    axins2 = inset_axes(
        ax,
        width="5%",  # width = 5% of parent_bbox width
        height="45%",  # height : 50%
        loc="upper left",
        bbox_to_anchor=(1.05, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cmap_grey = LinearSegmentedColormap.from_list(
        "custom_gray", [(0, 0, 0), (1, 1, 1)], N=256
    )
    cb2 = plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_grey), cax=axins2, orientation="vertical"
    )
    cb2.set_label("Retardance", rotation=270, labelpad=15)
    # Assuming the data for azimuth ranges from 0 to 2π, we normalize this range to 0-1 for the colorbar.
    cb2.set_ticks([0, 0.5, 1])
    cb2.set_ticklabels(
        ["0", round(retardance_img.max() / 2, 1), round(retardance_img.max(), 1)]
    )
    axins2.set_title("Value", fontsize=8)

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved image to {save_path}")

    display_plot = False
    if display_plot:
        plt.show()


def plot_retardance_orientation(
    ret_image, azim_image, azimuth_plot_type="hsv", include_labels=False
):
    plt.ioff()  # Prevents plots from popping up
    fig = plt.figure(figsize=(12, 3))
    plt.rcParams["image.origin"] = "upper"
    # Retardance subplot
    plt.subplot(1, 3, 1)
    plt.imshow(ret_image, cmap="plasma")  # viridis
    cbar1 = plt.colorbar(fraction=0.046, pad=0.04)
    cbar1.set_label("Radians")
    plt.title("Retardance")
    plt.xticks([])
    plt.yticks([])
    # Azimuth orientation subplot
    plt.subplot(1, 3, 2)
    plt.imshow(azim_image, cmap="twilight")
    cbar2 = plt.colorbar(fraction=0.046, pad=0.04)
    cbar2.set_label("Radians")
    plt.title("Orientation")
    plt.xticks([])
    plt.yticks([])
    plt.clim(0, np.pi)
    # Combined retardance and orientation subplot
    ax = plt.subplot(1, 3, 3)
    if azimuth_plot_type == "lines":
        plt.title("Retardance & Orientation")
        plot_birefringence_lines(
            ret_image, azim_image, cmap="viridis", line_color="white", ax=ax
        )
    else:
        if include_labels:
            plot_hue_map(ret_image, azim_image, ax=ax)
        else:
            plt.title("Retardance & Orientation")
            plt.xticks([])
            plt.yticks([])
            plot_birefringence_colorized(ret_image, azim_image)
    plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif"})
    plt.subplots_adjust(wspace=0.3, hspace=0)
    return fig


def main():
    width, height = 512, 512
    gradient_horizontal = np.linspace(0, 1, width)
    horizontal_img = np.tile(gradient_horizontal, (height, 1))
    rgb = hue_map(horizontal_img)
    plt.imshow(rgb, cmap="hsv")


if __name__ == "__main__":
    main()
