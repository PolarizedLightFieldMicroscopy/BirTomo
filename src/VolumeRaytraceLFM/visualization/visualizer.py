import matplotlib.pyplot as plt

class CombinedVisualizer:
    """
    A class that coordinates both VolumeVisualizer and ImageVisualizer 
    to produce combined or side-by-side plots.
    """

    def __init__(self, volume_visualizer, image_visualizer=None, default_dpi=300):
        """
        Args:
            volume_visualizer (VolumeVisualizer):
                An instance of your VolumeVisualizer class.
            image_visualizer (ImageVisualizer, optional):
                An instance of your ImageVisualizer class. If None, 
                some combined plots may not be available.
            default_dpi (int, optional):
                The default DPI for the plots.
        """
        self.volume_viz = volume_visualizer
        self.image_viz = image_visualizer
        self.default_dpi = default_dpi

    def save_figure(self, fig, filename: str):
        """
        Saves a given figure with the class's default DPI.
        
        Args:
            fig (matplotlib.figure.Figure): The figure to save.
            filename (str): The filename (and path) where the figure is saved.
        """
        fig.savefig(filename, dpi=self.default_dpi, bbox_inches="tight", pad_inches=0.1)

    def plot_3d_volume_and_ret_azim_colorized(
        self,
        threshold=0.0002,
        show_3d_axes=False,
        axis_order=(0, 1, 2),
        flip_dim=None,
        figsize=(11, 5),
    ):
        """
        Displays a 3D volume (with voxels above the given threshold 
        and optic-axis lines) on the left, and a colorized 
        retardance–azimuth image on the right.

        Args:
            threshold (float): Birefringence threshold for including voxels/axes.
            show_3d_axes (bool): Whether to show axis labels in the 3D subplot.
            axis_order (tuple[int,int,int]): Permutation of (z, y, x).
            flip_dim (int or None): If given, flips data along this dimension.
            figsize (tuple): The overall figure size in inches.

        Returns:
            fig (matplotlib.figure.Figure): The created matplotlib figure.
        """
        fig = plt.figure(figsize=figsize)

        # ===== Left Subplot: 3D Volume =====
        ax_3d = fig.add_subplot(1, 2, 1, projection="3d")

        vv = self.volume_viz
        vv._apply_3d_style(ax_3d)

        vv.display_3d_volume(
            threshold=threshold,
            ax=ax_3d,
            show_3d_axes=show_3d_axes,
            axis_order=axis_order,
            flip_dim=flip_dim,
        )

        # ===== Right Subplot: Retardance–Azimuth Colorized =====
        ax_right = fig.add_subplot(1, 2, 2)
        self.image_viz.plot_hue_map(ax=ax_right, enhance_contrast=True, gamma=0.8)
        ax_right.set_title("Polarized Light Field Image")

        plt.tight_layout()
        plt.show()

        return fig