import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import hsv_to_rgb, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _to_numpy(data):
    """
    Converts the input to a NumPy array if it's a PyTorch tensor.
    Otherwise, returns it as a NumPy array (possibly a view).
    """
    if hasattr(data, "detach") and callable(data.detach):
        # PyTorch Tensor
        return data.detach().cpu().numpy()
    return np.array(data)


class ImageVisualizer:
    """
    A utility class to visualize a single pair of 2D images:
      - Retardance (self.ret_img)
      - Azimuth/orientation (self.azim_img)

    By default, all methods reference these stored attributes for plotting.
    """

    def __init__(self, ret_img, azim_img, intensity_imgs=None):
        """
        Args:
            ret_img (np.ndarray or torch.Tensor): 2D array/tensor of retardance values.
            azim_img (np.ndarray or torch.Tensor): 2D array/tensor of azimuth/orientation values.
        """
        # Convert to NumPy if needed
        self.ret_img = _to_numpy(ret_img)
        self.azim_img = _to_numpy(azim_img)
        self.intensity_imgs = [_to_numpy(img) for img in intensity_imgs] if intensity_imgs is not None else None

    def plot_retardance(
        self,
        cmap="plasma",
        ax=None,
        show_colorbar=True,
        colorbar_label="Radians",
    ):
        """
        Plot the retardance image as a heatmap.

        Args:
            cmap (str): Colormap name for the retardance heatmap.
            ax (matplotlib.axes.Axes): Axes to draw on. If None, create a new figure.
            show_colorbar (bool): Whether to display a colorbar.
            colorbar_label (str): Label text for the colorbar.

        Returns:
            im (matplotlib.image.AxesImage): The displayed image object.
        """
        if ax is None:
            fig, ax = plt.subplots()

        im = ax.imshow(self.ret_img, cmap=cmap)
        ax.set_title("Retardance")
        ax.set_xticks([])
        ax.set_yticks([])

        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(colorbar_label)

        return im

    def plot_azimuth(
        self,
        cmap="twilight",
        ax=None,
        show_colorbar=True,
        colorbar_label="Radians",
    ):
        """
        Plot the azimuth (orientation) image as a heatmap.

        Args:
            cmap (str): Colormap name for the orientation heatmap.
            ax (matplotlib.axes.Axes): Axes to draw on. If None, create a new figure.
            show_colorbar (bool): Whether to display a colorbar.
            colorbar_label (str): Label text for the colorbar.

        Returns:
            im (matplotlib.image.AxesImage): The displayed image object.
        """
        if ax is None:
            fig, ax = plt.subplots()

        im = ax.imshow(self.azim_img, cmap=cmap)
        ax.set_title("Orientation")
        ax.set_xticks([])
        ax.set_yticks([])

        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(colorbar_label)

        return im

    def plot_azimuth_lines(
        self,
        origin="upper",
        cmap="plasma",
        line_color="white",
        ax=None,
        show_colorbar=True,
        colorbar_label="Radians",
    ):
        """
        Plot the retardance heatmap and overlay short line segments
        indicating local orientation (azimuth).

        Args:
            origin (str): 'upper' or 'lower' for the Matplotlib image origin.
            cmap (str):   Colormap name for the retardance heatmap.
            line_color (str): Color of the orientation lines.
            ax (matplotlib.axes.Axes): Axes to draw on. If None, create a new figure.
            show_colorbar (bool): Whether to display a colorbar for the retardance.
            colorbar_label (str): Label for the colorbar.
        """
        created_new_figure = False
        if ax is None:
            fig, ax = plt.subplots()
            created_new_figure = True

        im = ax.imshow(self.ret_img, origin=origin, cmap=cmap)
        ax.set_title("Retardance & Orientation")
        ax.set_xticks([])
        ax.set_yticks([])

        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(colorbar_label)

        s_i, s_j = self.ret_img.shape
        xx, yy = np.meshgrid(np.arange(s_j), np.arange(s_i))

        # Scale factor for line length
        scale_factors = np.ones_like(self.ret_img, dtype=float) * 0.75
        scale_factors[self.ret_img == 0] = 0
        half_len = 0.5 * scale_factors

        # Using x +/- cos(theta) and y +/- sin(theta) 
        l_x = (xx - half_len * np.cos(self.azim_img)).flatten()
        r_x = (xx + half_len * np.cos(self.azim_img)).flatten()
        l_y = (yy + half_len * np.sin(self.azim_img)).flatten()
        r_y = (yy - half_len * np.sin(self.azim_img)).flatten()

        # Build line segments
        segments = [((l_x[i], l_y[i]), (r_x[i], r_y[i])) for i in range(len(l_x))]

        # Draw as a line collection
        lc = matplotlib.collections.LineCollection(segments, colors=line_color, linewidths=1, alpha=1.0)
        ax.add_collection(lc)
        ax.autoscale()

        if created_new_figure:
            plt.show()


    def plot_birefringence_colorized(self, gamma=1.0):
        """
        Create a colorized display in HSV space:
          - Hue   = azimuth
          - Value = retardance (with optional gamma correction)
          - Saturation is fixed at 0.5
        """
        retardance_img = self.ret_img
        azimuth_img = self.azim_img

        H, W = azimuth_img.shape
        hsv_data = np.zeros((H, W, 3), dtype=float)

        # Hue
        max_azim = azimuth_img.max() if azimuth_img.max() != 0 else 1e-8
        hsv_data[:, :, 0] = azimuth_img / max_azim

        # Saturation
        hsv_data[:, :, 1] = 0.5

        # Value
        max_ret = retardance_img.max() if retardance_img.max() != 0 else 1e-8
        val_channel = np.power(retardance_img / max_ret, gamma)
        hsv_data[:, :, 2] = val_channel

        hsv_data[np.isnan(hsv_data)] = 0
        rgb = hsv_to_rgb(hsv_data)

        plt.imshow(rgb, cmap="hsv")
        plt.title("Polarized Light Field Image")
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def plot_hue_map(
        self,
        enhance_contrast=False,
        save_path=None,
        dpi=300,
        max_retardance=None,
        gamma=1.0,
        data_scale=None,
        ax=None,
    ):
        """
        Plot an HSV overlay where:
        - Hue = self.azim_img
        - Value = self.ret_img
        - Saturation = 0.5
        plus optional colorbars for orientation and retardance.
        """
        def contrast_stretching(img, low_pct=0, high_pct=99, max_r=None):
            """
            Contrast-stretch the image between [p_low..p_high], 
            but ensure that '1' in normalized scale still corresponds to 'max_r'.
            """
            if max_r is None:
                max_r = np.max(img) or 1e-8
            p_low, p_high = np.percentile(img, (low_pct, high_pct))
            if p_high <= p_low:
                p_low = 0
                p_high = max_r
            clipped = np.clip(img, p_low, p_high)
            stretched_01 = (clipped - p_low) / (p_high - p_low)
            stretched_to_max = stretched_01 * max_r

            return stretched_to_max

        # Copy data so as not to mutate self.ret_img
        retardance_img = self.ret_img.copy()
        azimuth_img = self.azim_img

        if enhance_contrast:
            retardance_img = contrast_stretching(retardance_img)

        if max_retardance is None:
            max_retardance = retardance_img.max() or 1e-8

        H, W = azimuth_img.shape
        hsv_data = np.zeros((H, W, 3), dtype=float)

        # Hue
        max_azim = azimuth_img.max() or 1.0
        hsv_data[:, :, 0] = azimuth_img / max_azim

        # Saturation
        hsv_data[:, :, 1] = 0.5

        # Value
        val_channel = (retardance_img / max_retardance) ** gamma
        hsv_data[:, :, 2] = val_channel
        hsv_data[np.isnan(hsv_data)] = 0

        rgb = hsv_to_rgb(hsv_data)

        # --- Key change: Only create a new figure/axes if ax is None. ---
        created_new_figure = False
        if ax is None:
            fig, ax = plt.subplots()
            created_new_figure = True

        im = ax.imshow(rgb, cmap="hsv")
        ax.set_xticks([])
        ax.set_yticks([])

        # Add inset colorbars
        axins1 = inset_axes(
            ax, width="5%", height="45%", loc="lower left",
            bbox_to_anchor=(1.05, 0.0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cmap_hue = plt.get_cmap("hsv")
        cb1 = plt.colorbar(
            plt.cm.ScalarMappable(cmap=cmap_hue), cax=axins1, orientation="vertical"
        )
        cb1.set_label("Orientation", rotation=90, labelpad=8)
        cb1.set_ticks([0, 0.5, 1])
        cb1.set_ticklabels(["0", r"$\pi/2$", r"$\pi$"])
        axins1.set_title("Hue", fontsize=8)

        axins2 = inset_axes(
            ax, width="5%", height="45%", loc="upper left",
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
        cb2.set_label("Retardance", rotation=90, labelpad=8)
        cb2.set_ticks([0, 0.5, 1])
        max_val = max_retardance
        if not data_scale:
            cb2.set_ticklabels(["0", round(max_val / 2, 1), round(max_val, 1)])
            axins2.set_title("Value", fontsize=8)
        else:
            tick_vals = [0, 0.5 * max_val, max_val]
            tick_labels = [f"{tv * data_scale:.2f}" for tv in tick_vals]
            cb2.set_ticklabels(tick_labels)
            axins2.set_title(r"Value $(\times 10^{-3})$", fontsize=8)

        if save_path and created_new_figure:
            # If we created a new figure, we can save and show it here
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.show()
        elif created_new_figure:
            plt.show()

        # Otherwise, the caller (like a "combined" plot) controls showing the figure
        return im

    def plot_retardance_orientation(
        self, azimuth_plot_type="hsv", include_labels=True
    ):
        """
        Create a figure with three subplots:
          1) Retardance (heatmap)
          2) Azimuth/orientation (heatmap)
          3) Combined representation (orientation lines or HSV color overlay).

        Args:
            azimuth_plot_type (str): "lines" or "hsv".
            include_labels (bool): If True, add the hue map with colorbars; 
                                   otherwise just show the colorized image.
        """
        ret_image = self.ret_img
        azim_image = self.azim_img

        plt.ioff()  # Prevents interactive popup in some environments
        fig = plt.figure(figsize=(12, 3))
        plt.rcParams["image.origin"] = "upper"

        # (1) Retardance
        ax1 = plt.subplot(1, 3, 1)
        im1 = ax1.imshow(ret_image, cmap="plasma")
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label("Radians")
        ax1.set_title("Retardance")
        ax1.set_xticks([])
        ax1.set_yticks([])

        # (2) Azimuth
        ax2 = plt.subplot(1, 3, 2)
        im2 = ax2.imshow(azim_image, cmap="twilight")
        im2.set_clim(0, np.pi)
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label("Radians")
        ax2.set_title("Orientation")
        ax2.set_xticks([])
        ax2.set_yticks([])

        # (3) Combined
        ax3 = plt.subplot(1, 3, 3)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title("Retardance & Orientation")

        if azimuth_plot_type == "lines":
            self.plot_azimuth_lines(
                origin="upper",
                cmap="plasma",
                line_color="white",
                ax=ax3,
                show_colorbar=False  # Already have colorbars in subplots 1 & 2
            )
        else:
            if include_labels:
                # Use the method that includes colorbars in the same subplot
                self.plot_hue_map(ax=ax3)
            else:
                # Or simply do an HSV color overlay
                self.plot_birefringence_colorized(gamma=4)

        plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif"})
        plt.subplots_adjust(left=0.03, wspace=0.3, hspace=0)
        return fig

    def plot_intensity_images(self, cmap="gray", show_colorbar=True):
        """
        Plot intensity images stored in self.intensity_imgs with a shared colorbar.

        Args:
            cmap (str): Colormap to use for displaying intensity images.

        Returns:
            fig (matplotlib.figure.Figure): The created figure object.
        """
        num_imgs = len(self.intensity_imgs)
        plt.ioff()  # Prevent interactive popup in some environments

        fig, axes = plt.subplots(1, num_imgs, figsize=(4 * num_imgs, 4))
        plt.rcParams["image.origin"] = "upper"

        vmin = min(img.min() for img in self.intensity_imgs)
        vmax = max(img.max() for img in self.intensity_imgs) * 0.5

        ims = []
        for idx, img in enumerate(self.intensity_imgs):
            ax = axes.flatten()[idx]
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ims.append(im)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"$\\Sigma_{{{idx + 1}}}$")

        if show_colorbar:
            # Single shared colorbar
            divider = make_axes_locatable(axes[-1])
            cax = divider.append_axes('right', size='5%', pad=0.1)
            cbar = fig.colorbar(ims[-1], cax=cax, orientation='vertical')
            cbar.set_label("Intensity")

        plt.subplots_adjust(wspace=0.1, hspace=0)
        plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif"})

        # created_new_figure = True
        # if created_new_figure:
        #     plt.show()
        # plt.ion()
        # plt.show()
        return fig
