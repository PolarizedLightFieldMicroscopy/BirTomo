import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa

try:
    import torch
except ImportError:
    torch = None


DEBUG = False


class VolumeVisualizer:
    """
    Utility class to visualize a BirefringentVolume object.
    Supports both NumPy and Torch backends.

    Internally, the volume has shape (4, Z, Y, X):
      - channel 0 = bir (birefringence)
      - channel 1 = az (optic axis z-component)
      - channel 2 = ay (optic axis y-component)
      - channel 3 = ax (optic axis x-component)

    By Python convention:
      - dimension 0 of the 3D shape = Z
      - dimension 1 of the 3D shape = Y
      - dimension 2 of the 3D shape = X

    Physically, you may refer to them as dim=1 (Z-axis), dim=2 (Y-axis), and dim=3 (X-axis) 
    in 1-based notation. Always be consistent and clear in your usage.
    """

    def __init__(self, birefringent_volume, crop_shape=None) -> None:
        """
        Args:
            birefringent_volume (BirefringentVolume): 
                An instance of the BirefringentVolume class.
        """
        self.bv = birefringent_volume
        self.backend = birefringent_volume.backend

        # The 3D shape is typically (Z, Y, X) in Python indexing
        self.shape_zyx = birefringent_volume.volume_shape  
        self.volume_4d = self._get_4d_volume()
        if crop_shape is not None:
            self.crop_volume_center(crop_shape)

        self._3d_style = {
            "pane_alpha": 0.1,
            "grid_alpha": 0.3,
            "grid_color": "gray",
            "linewidth": 0.3
        }

        plt.rcParams.update({
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12
        })

    def _get_4d_volume(self):
        """
        Retrieves the 4D tensor/array of shape (4, Z, Y, X):
          channel 0 = bir
          channel 1 = az
          channel 2 = ay
          channel 3 = ax
        """
        volume_4d = self.bv.get_4d_volume_representation()
        return self._to_numpy(volume_4d)

    def _get_transformed_data(
        self,
        channel: int,
        flip_dim: int | tuple[int, ...] | None,
        axis_order: tuple[int, int, int]
    ) -> np.ndarray:
        """
        Returns the data (3D) for a given channel, flipped and/or transposed
        according to flip_dim and axis_order.
        """
        data_3d = self.volume_4d[channel]  # shape: (Z, Y, X) for channel=0

        # Transpose if requested
        if axis_order != (0, 1, 2):
            data_3d = np.transpose(data_3d, axis_order)
        
        # Allow for multiple flips if flip_dim is a list/tuple
        if flip_dim is not None:
            if isinstance(flip_dim, int):
                flip_dim = [flip_dim]
            for fd in flip_dim:
                data_3d = np.flip(data_3d, axis=fd)
        
        return data_3d

    def _get_transformed_optic_axes(
        self,
        axis_order: tuple[int, int, int] = (0, 1, 2),
        flip_dim: int | tuple[int, ...] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (positions, directions) for the optic axis.
        
        The positions are computed from the transformed channel 0 (using _get_transformed_data),
        so they exactly match the ordering of the transformed data.
        
        The optic axis channels (channels 1,2,3) are also transformed via transpose and flip,
        and then their vector components are negated on any flipped axes.
        """
        # 1. Get the transformed channel 0 data. This applies the same transpose and flip as _get_transformed_data.
        transformed_data = self._get_transformed_data(channel=0, flip_dim=flip_dim, axis_order=axis_order)
        final_shape = transformed_data.shape  # This is the shape after all transformations.
        
        # Print nonzero indices from transformed channel 0 data.
        nz_trans = np.nonzero(transformed_data)
        if DEBUG:
            nz_trans_plain = list(zip(nz_trans[0].tolist(), nz_trans[1].tolist(), nz_trans[2].tolist()))
            print("Nonzero indices from transformed volume_4d[0]:", nz_trans_plain)
        
        # 2. Build the positions grid from the final shape.
        zz, yy, xx = np.indices(final_shape)
        positions = np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)
        # Shift positions to the voxel centers.
        positions = positions.astype(np.float64) #+ 0.5

        # 3. Transform the optic axis channels.
        # Extract raw optic axis channels (assumed to be channels 1,2,3 in the original volume_4d).
        ax_x = self.volume_4d[1]
        ax_y = self.volume_4d[2]
        ax_z = self.volume_4d[3]
        
        # Apply axis ordering (transpose) if needed.
        if axis_order != (0, 1, 2):
            ax_x = np.transpose(ax_x, axis_order)
            ax_y = np.transpose(ax_y, axis_order)
            ax_z = np.transpose(ax_z, axis_order)
        
        # Now apply the same flips as in _get_transformed_data.
        if flip_dim is not None:
            # Ensure flip_dim is a list.
            if isinstance(flip_dim, int):
                flip_list = [flip_dim]
            else:
                flip_list = list(flip_dim)
            for fd in flip_list:
                # pass
                ax_x = np.flip(ax_x, axis=fd)
                ax_y = np.flip(ax_y, axis=fd)
                ax_z = np.flip(ax_z, axis=fd)
        
        # 4. Stack the optic axis channels into a vector field.
        directions = np.stack([ax_x, ax_y, ax_z], axis=-1)  # shape: final_shape + (3,)
        if axis_order != (0, 1, 2):
            directions = directions.reshape(-1, 3)
            directions = directions[:, list(axis_order)]
        else:
            directions = directions.reshape(-1, 3)
        
        # 5. After flipping spatially, reverse the sign of the corresponding vector component.
        # if flip_dim is not None:
        #     if isinstance(flip_dim, int):
        #         flip_list = [flip_dim]
        #     else:
        #         flip_list = list(flip_dim)
        #     flip_list.append(0)
        #     flip_list.append(1)
        #     for fd in flip_list:
        #         directions[:, fd] = -directions[:, fd]
        
        # 6. For debugging, extract positions and directions for the nonzero voxels.
        mask_trans = (np.abs(transformed_data) != 0)
        mask_trans_flat = mask_trans.ravel()
        pos_nonzero = positions[mask_trans_flat]
        vecs_nonzero = directions[mask_trans_flat]
        
        if DEBUG:
            print("Computed positions for nonzero transformed voxels:", pos_nonzero)
            print("Optic-axis directions for nonzero transformed voxels:", vecs_nonzero)
        
        return positions, directions


    def _get_transformed_optic_axes1(
        self,
        axis_order: tuple[int, int, int] = (0, 1, 2),
        flip_dim: int | tuple[int, ...] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (positions, directions) for the optic axis.
        
        The grid positions are computed by applying the axis ordering (from the raw shape)
        and then applying flips (if any). The optic axis vector field is re-gridded in the same
        way and its vector components are re-ordered according to axis_order. Finally, flips
        (applied after axis ordering) reverse the sign of the corresponding vector components.
        
        This ensures that both positions and directions are defined in the same final coordinate system.
        """
        # 1. Compute the final spatial grid shape from the raw scalar data.
        # Assume channel 0 has shape (Z, Y, X).
        raw_shape = self.volume_4d[0].shape  # e.g. (Z, Y, X)
        # The final shape after axis ordering is:
        final_shape = tuple(raw_shape[i] for i in axis_order)
        
        # 2. Build positions from the final shape.
        # Use np.indices with indexing 'ij' so that the first axis corresponds to axis 0, etc.
        zz, yy, xx = np.indices(final_shape)
        positions = np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)
        
        # 3. Extract the raw optic axis channels (do not modify their vector component order yet).
        ax_x = self.volume_4d[1]
        ax_y = self.volume_4d[2]
        ax_z = self.volume_4d[3]
        
        # 4. Re-grid the optic axis directions to match the spatial ordering.
        # First, reorder the spatial layout by applying the same axis ordering.
        if axis_order != (0, 1, 2):
            ax_x = np.transpose(ax_x, axis_order)
            ax_y = np.transpose(ax_y, axis_order)
            ax_z = np.transpose(ax_z, axis_order)
        
        # 5. Stack the re-gridded channels.
        # At this point, each array has shape = final_shape.
        directions = np.stack([ax_x, ax_y, ax_z], axis=-1)  # shape: final_shape + (3,)
        
        # 6. Re-order the vector components according to axis_order.
        # For example, if axis_order is (2, 1, 0), then the new first component is the original third one.
        if axis_order != (0, 1, 2):
            directions = directions.reshape(-1, 3)
            directions = directions[:, list(axis_order)]
        else:
            directions = directions.reshape(-1, 3)
        
        # 7. Apply flips AFTER axis ordering to both positions and directions.
        if flip_dim is not None:
            flip_list = [flip_dim] if isinstance(flip_dim, int) else list(flip_dim)
            # For positions: adjust the voxel coordinate in the flipped axis.
            positions = positions.copy()
            for fd in flip_list:
                max_coord = final_shape[fd] - 1
                positions[:, fd] = max_coord - positions[:, fd]
            # For directions: flip the sign of the vector component in the flipped axis.
            for fd in flip_list:
                directions[:, fd] = -directions[:, fd]

        # Now, use _get_transformed_data to get the transformed channel 0 data.
        transformed_data = self._get_transformed_data(channel=0, flip_dim=flip_dim, axis_order=axis_order)
        nz_trans = np.nonzero(transformed_data)
        nz_trans_plain = list(zip(nz_trans[0].tolist(), nz_trans[1].tolist(), nz_trans[2].tolist()))
        print("Nonzero indices from transformed volume_4d[0]:", nz_trans_plain)

        # Create a boolean mask from the transformed data.
        # (Assuming nonzero means a value != 0; adjust the condition if needed.)
        mask_trans = (np.abs(transformed_data) != 0)
        mask_trans_flat = mask_trans.ravel()
        
        # Get positions and directions corresponding to nonzero voxels in the transformed data.
        pos_nonzero_preshift = positions[mask_trans_flat]
        pos_nonzero = positions[mask_trans_flat].astype(np.float64) + 0.5
        vecs_nonzero = directions[mask_trans_flat]
        
        print("Positions for nonzero transformed voxels:", pos_nonzero_preshift)
        print("Optic-axis directions for nonzero transformed voxels:", vecs_nonzero)

        # nz = np.nonzero(self.volume_4d[0])
        # nz_plain = list(zip(nz[0].tolist(), nz[1].tolist(), nz[2].tolist()))
        # print("Nonzero indices from volume_4d[0]:", nz_plain)

        # # print positions and directions of the nonzero indices from the volume_4d[0] and from the transformed volume_4d[0]
        
        return positions, directions


    def _get_transformed_optic_axes_og(
        self,
        axis_order: tuple[int, int, int] = (0, 1, 2),
        flip_dim: int | tuple[int, ...] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (positions, directions) for the optic axis lines or vectors,
        after applying the same flips and transpositions used for the voxel data.

        positions: A shape (N, 3) array of voxel coordinates (x, y, z)
                or however you store them.
        directions: A shape (N, 3) array of axis direction vectors for each voxel.
        """

        # 1) Extract the raw direction channels from volume_4d
        # Assume volume_4d[1,2,3] = X,Y,Z components, shape: (3, Z, Y, X)
        ax_x = self.volume_4d[1]  # shape (Z, Y, X)
        ax_y = self.volume_4d[2]
        ax_z = self.volume_4d[3]

        # 2) We'll build coordinate grids for the domain
        #    so we can apply the same transformations to positions.
        Z_dim, Y_dim, X_dim = ax_x.shape
        zz, yy, xx = np.meshgrid(
            np.arange(Z_dim),
            np.arange(Y_dim),
            np.arange(X_dim),
            indexing='ij'
        )
        # Now zz.shape == yy.shape == xx.shape == (Z_dim, Y_dim, X_dim)

        # Combine them into positions, shape (Z*Y*X, 3)
        positions = np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)
        # Directions, shape (Z*Y*X, 3)
        directions = np.stack([ax_x, ax_y, ax_z], axis=-1).reshape(-1, 3)

        # 3) Transpose (reorder) the domain if requested
        if axis_order != (0, 1, 2):
            # axis_order is a permutation of (0, 1, 2).
            # That means old coordinate [z, y, x] becomes e.g. [x, y, z].
            #
            # We need to reorder both positions and direction components:
            # Example: If axis_order = (2, 1, 0),
            #   old [z, y, x] => new [x, y, z]
            #   so positions[:, 0] (z) => positions[:, 2] (x)
            #   directions[:, 0] (Z-dir) => directions[:, 2] (X-dir)
            # We can do that by indexing with axis_order:

            # Re-map the positions:
            positions = positions[:, list(axis_order)]

            # Re-map the direction vectors the same way:
            directions = directions[:, list(axis_order)]

        # 4) Allow for multiple flips if flip_dim is a list/tuple
        if flip_dim is not None:
            if isinstance(flip_dim, int):
                flip_dims = [flip_dim]
            else:
                flip_dims = flip_dim
            # flip_dims = (0,) + tuple(flip_dims)
        
        # elif flip_dim is None:
        #     flip_dims = (0,)
            
            for fd in flip_dims:
                if fd == 0:
                    max_coord = Z_dim - 1
                elif fd == 1:
                    max_coord = Y_dim - 1
                elif fd == 2:
                    max_coord = X_dim - 1
                else:
                    raise ValueError(f"flip_dim must be 0, 1, or 2. Got: {fd}")

                positions[:, fd] = max_coord - positions[:, fd]
                directions[:, fd] = -directions[:, fd]

        # Return them
        return positions, directions

    def _to_numpy(self, arr):
        """
        Utility to convert a Torch tensor or NumPy array to a NumPy array.
        """
        if self.backend == "torch":
            return arr.detach().cpu().numpy()
        return arr

    def _apply_3d_style(self, ax: Axes3D) -> None:
        """
        Applies consistent 3D style settings to the given Axes3D object.
        For example, sets the alpha (transparency) of the 'pane' backgrounds,
        adjusts grid lines, etc.
        """
        style = self._3d_style
        # --- 1) Pane Backgrounds ---
        pane_color = style.get("pane_color", "white")
        pane_alpha = style.get("pane_alpha", 1.0)
        # Set facecolor and alpha (this can be an RGBA tuple or separate alpha)
        ax.xaxis.pane.set_facecolor(pane_color)
        ax.yaxis.pane.set_facecolor(pane_color)
        ax.zaxis.pane.set_facecolor(pane_color)

        ax.xaxis.pane.set_alpha(pane_alpha)
        ax.yaxis.pane.set_alpha(pane_alpha)
        ax.zaxis.pane.set_alpha(pane_alpha)

        # 2) Grid lines
        # Each axis has a dictionary of info under _axinfo
        grid_color = style.get("grid_color", "gray")
        grid_alpha = style.get("grid_alpha", None)
        if grid_alpha is not None:
            # Convert grid_color to RGBA if it's not already a tuple
            # (matplotlib will accept color names, hex codes, etc., so we must convert)
            rgba = colors.to_rgba(grid_color)  # e.g. (R, G, B, A)
            # Blend in the new alpha
            grid_color = (rgba[0], rgba[1], rgba[2], grid_alpha)

        grid_linewidth = style.get("linewidth", 1.0)
        grid_linestyle = style.get("linestyle", "-")

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["color"] = grid_color
            axis._axinfo["grid"]["linewidth"] = grid_linewidth
            axis._axinfo["grid"]["linestyle"] = grid_linestyle

        # Fully transparent background:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    def _configure_3d_axes(self, ax, axis_order: tuple[int, int, int], dims: tuple[int, int, int], show_axes: bool):
        """
        Configures the 3D axes with labels and limits based on the axis order.

        Args:
            ax: The matplotlib 3D axis object.
            axis_order: A tuple indicating the order of the axes, e.g. (0, 1, 2) or (2, 1, 0).
            dims: A tuple containing (z_dim, y_dim, x_dim) used for setting axis limits.
            show_axes: If True, sets axis labels and limits; otherwise turns the axes off.
        """
        z_dim, y_dim, x_dim = dims
        if show_axes:
            if axis_order == (0, 1, 2):
                ax.set_xlabel("$z$ (dim=0)")
                ax.set_ylabel("$y$ (dim=1)")
                ax.set_zlabel("$x$ (dim=2)") #, labelpad=0
            elif axis_order == (2, 1, 0):
                ax.set_xlabel("$x$")
                ax.set_ylabel("$y$")
                ax.set_zlabel("$z$") #, labelpad=0
            # Set limits common to both configurations.
            ax.set_xlim(0, z_dim)
            ax.set_ylim(0, y_dim)
            ax.set_zlim(0, x_dim)

            # Set integer ticks on each axis
            ax.set_xticks(np.arange(0, z_dim + 1, 1))
            ax.set_yticks(np.arange(0, y_dim + 1, 1))
            ax.set_zticks(np.arange(0, x_dim + 1, 1))
            
            # Ensure grid lines match the ticks
            ax.grid(True)
        else:
            ax.axis("off")

    def crop_volume_center(self, shape_zyx: tuple[int, int, int]) -> None:
        """
        Crop the volume in-place, keeping only a centered region of size:
          (Z_crop, Y_crop, X_crop)
        in the last three dimensions (Z, Y, X).

        shape before: (4, Z, Y, X)
        shape after:  (4, Z_crop, Y_crop, X_crop)

        Args:
            shape_zyx (tuple): (Z_crop, Y_crop, X_crop) = new center shape
        """
        z_crop, y_crop, x_crop = shape_zyx
        # Current shapes
        _, Z, Y, X = self.volume_4d.shape

        # Check for invalid requests
        if z_crop > Z or y_crop > Y or x_crop > X:
            raise ValueError(
                f"Requested crop shape {shape_zyx} exceeds current volume "
                f"shape (Z={Z}, Y={Y}, X={X})."
            )

        # Compute start/end indices for the center in Z
        z_start = (Z - z_crop) // 2
        z_end   = z_start + z_crop

        # Compute start/end indices for the center in Y
        y_start = (Y - y_crop) // 2
        y_end   = y_start + y_crop

        # Compute start/end indices for the center in X
        x_start = (X - x_crop) // 2
        x_end   = x_start + x_crop

        # Crop in-place
        self.volume_4d = self.volume_4d[
            :,
            z_start:z_end,
            y_start:y_end,
            x_start:x_end
        ]

        new_shape = self.volume_4d.shape
        print(f"Cropped volume to center shape: {new_shape}  (Z, Y, X = {z_crop}, {y_crop}, {x_crop})")

    def get_slice_2d(
        self,
        channel: str = "bir",
        slice_index: int = 0,
        axis: str = "z"
    ):
        """
        Extract a single 2D slice from the 4D volume.

        The volume_4d has shape (4, Z, Y, X):
          - channel 0: bir (birefringence)
          - channel 1: az (optic axis z-component)
          - channel 2: ay (optic axis y-component)
          - channel 3: ax (optic axis x-component)

        Args:
            channel (str): One of {"bir", "az", "ay", "ax"}.
            slice_index (int): Index along the chosen axis.
            axis (str): One of {"z", "y", "x"} 
                        - "z" → dimension 0
                        - "y" → dimension 1
                        - "x" → dimension 2

        Returns:
            slice_2d (np.ndarray or torch.Tensor):
                The 2D slice (same backend as self.volume_4d).
        """
        valid_channels = {"bir": 0, "az": 1, "ay": 2, "ax": 3}
        if channel not in valid_channels:
            raise ValueError(f"Invalid channel '{channel}'. "
                             f"Expected one of {list(valid_channels.keys())}.")

        axis_to_dim = {"z": 0, "y": 1, "x": 2}
        if axis not in axis_to_dim:
            raise ValueError("Axis must be one of 'z', 'y', or 'x'.")

        channel_idx = valid_channels[channel]
        dim = axis_to_dim[axis]
        data_3d = self.volume_4d[channel_idx]  # shape: (Z, Y, X)

        # Slice according to the chosen axis
        if dim == 0:
            slice_2d = data_3d[slice_index, :, :]
        elif dim == 1:
            slice_2d = data_3d[:, slice_index, :]
        else:  # dim == 2
            slice_2d = data_3d[:, :, slice_index]

        return slice_2d

    def plot_slice(
        self,
        channel: str = "bir",
        slice_index: int = 0,
        axis: str = "z",
        cmap: str = "viridis"
    ) -> None:
        """
        Plot a single 2D slice from the volume.

        Args:
            channel (str): One of {"bir", "az", "ay", "ax"}.
            slice_index (int): Index along the chosen axis.
            axis (str): One of {"z", "y", "x"}.
            cmap (str): Matplotlib colormap name.
        """
        slice_2d = self.get_slice_2d(channel, slice_index, axis)

        # If channel is an optic axis (az, ay, ax), fix the color range to [0, 1].
        if channel in ("az", "ay", "ax"):
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = None, None

        plt.figure(figsize=(5, 5))
        plt.imshow(slice_2d, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        plt.colorbar(label=channel)
        plt.title(f"{channel} slice @ {axis}={slice_index}")

        # Label axes more explicitly
        if axis == "z":
            plt.xlabel("X")
            plt.ylabel("Y")
        elif axis == "y":
            plt.xlabel("X")
            plt.ylabel("Z")
        else:  # axis == "x"
            plt.xlabel("Y")
            plt.ylabel("Z")

        plt.show()

    def plot_slice_montage(
        self,
        channel: str = "bir",
        axis: str = "z",
        start: int = 0,
        end: int = None,
        step: int = 1,
        cmap: str = "viridis"
    ) -> None:
        """
        Plot multiple slices in a grid (montage).

        Args:
            channel (str): One of {"bir", "az", "ay", "ax"}.
            axis (str): Which axis to slice along {"z", "y", "x"}.
            start (int): First slice index.
            end (int): Last slice index (exclusive). 
                       If None, uses the size along that axis.
            step (int): Step between slices.
            cmap (str): Matplotlib colormap name.
        """
        axis_to_dim = {"z": 0, "y": 1, "x": 2}
        if axis not in axis_to_dim:
            raise ValueError("Axis must be 'z', 'y', or 'x'.")

        dim = axis_to_dim[axis]
        size_along_axis = self.shape_zyx[dim]

        if end is None:
            end = size_along_axis

        indices = range(start, end, step)
        num_slices = len(indices)

        # Decide on a grid for plotting
        num_cols = int(np.ceil(np.sqrt(num_slices)))
        num_rows = int(np.ceil(num_slices / num_cols))

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))
        axs = np.atleast_2d(axs)

        # Pre-calculate plotting range if channel is az/ay/ax
        if channel in ("az", "ay", "ax"):
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = None, None

        for ax_i, slice_index in enumerate(indices):
            row = ax_i // num_cols
            col = ax_i % num_cols

            slice_2d = self.get_slice_2d(channel, slice_index, axis)

            im = axs[row, col].imshow(slice_2d, cmap=cmap, origin="lower",
                                      vmin=vmin, vmax=vmax)
            axs[row, col].set_title(f"{channel}: {axis}={slice_index}")
            axs[row, col].axis("off")

            # Add colorbar in the same cell
            plt.colorbar(im, ax=axs[row, col], fraction=0.046, pad=0.04)

        # Hide any unused subplots
        total_plots = num_rows * num_cols
        for ax_j in range(ax_i + 1, total_plots):
            row = ax_j // num_cols
            col = ax_j % num_cols
            axs[row, col].axis("off")

        plt.tight_layout()
        plt.show()

    def plot_bir_histogram(
        self,
        bins: int = 50,
        log: bool = False
    ) -> None:
        """
        Plot a histogram of the birefringence (channel 0) distribution.

        Args:
            bins (int): Number of histogram bins.
            log (bool): Whether to use a log scale on the y-axis.
        """
        bir_3d = self.volume_4d[0]  # channel 0 = bir
        values = bir_3d.ravel()

        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=bins, log=log, color="blue", alpha=0.7)
        plt.xlabel("birefringence")
        plt.ylabel("Count")
        if log:
            plt.yscale("log")
        plt.title("Histogram of Birefringence")
        plt.show()

    def print_nonzero_voxels(
        self,
        threshold: float = 1e-8,
        max_to_print: int = 20
    ) -> None:
        """
        Prints the indices, birefringence, and optic axis for nonzero voxels.

        Args:
            threshold (float): Minimum |bir| threshold to consider a voxel "nonzero".
            max_to_print (int): Maximum number of nonzero voxels to print.
        """
        volume_4d = self.volume_4d
        bir_3d = volume_4d[0]  # channel 0 = bir

        nonzero_mask = np.abs(bir_3d) > threshold
        nonzero_indices = np.argwhere(nonzero_mask)

        print(f"Found {len(nonzero_indices)} voxels with |bir| > {threshold}.")
        print(f"Printing up to the first {max_to_print}...")

        for i, (z, y, x) in enumerate(nonzero_indices):
            if i >= max_to_print:
                print("... (truncated) ...")
                break
            bir_val = bir_3d[z, y, x]

            # Extract optic axis (channels 1..3): az, ay, ax
            az_val = volume_4d[1, z, y, x]
            ay_val = volume_4d[2, z, y, x]
            ax_val = volume_4d[3, z, y, x]

            print(f"Voxel (z={z}, y={y}, x={x}): "
                  f"bir={bir_val:.4f}, "
                  f"optic_axis=(az={az_val:.4f}, "
                  f"ay={ay_val:.4f}, "
                  f"ax={ax_val:.4f})")

    def display_3d_birefringence(
        self,
        threshold: float = 0.0002,
        show_3d_axes: bool = True,
        axis_order: tuple[int, int, int] = (0, 1, 2),
        flip_dim: int | tuple[int, ...] | None = None,
        ax=None,
        voxel_alpha: float = 0.4,
    ) -> None:
        """
        Displays a 3D voxel plot of the birefringence data (channel 0) from the volume.
        
        Args:
            threshold (float):
                Minimum birefringence (|bir|) to display. Voxels below this are omitted.
            show_3d_axes (bool):
                Whether to show 3D axes labels. If False, axes are hidden.
            axis_order (tuple[int,int,int]):
                Order to transpose the data (default: (0, 1, 2)).
            flip_dim (int or None):
                If provided, flip the data along this axis.
            ax (matplotlib.axes.Axes):
                Axes to draw on. If None, a new figure/axes is created.
        
        Returns:
            A tuple (ax, norm) where:
                - ax is the Axes3D object with the plot.
                - norm is the normalization object for the colormap (or None if data is constant).
        """
        bir_data = self._get_transformed_data(0, flip_dim, axis_order)
        bir_mask = np.abs(bir_data) >= threshold

        if np.any(bir_mask):
            unique_vals = np.unique(bir_data[bir_mask])
        else:
            unique_vals = np.array([])

        if unique_vals.size <= 1:
            constant_value = unique_vals.item() if unique_vals.size == 1 else 0
            constant_color = "purple" if constant_value < 0 else "forestgreen"
            facecolors = np.zeros(bir_data.shape + (4,), dtype=float)
            facecolors[bir_mask] = colors.to_rgba(constant_color)
            norm = None
            display_colorbar = False
        else:
            min_val, max_val = bir_data.min(), bir_data.max()
            if min_val >= 0:
                # Data has only nonnegative values: use a sequential colormap
                # from white (0) to forest green (max).
                norm = colors.Normalize(vmin=0, vmax=max_val)
                cmap = colors.LinearSegmentedColormap.from_list(
                    "white_forestgreen", ["white", "forestgreen"]
                )
            elif max_val <= 0:
                # Data has only nonpositive values: use a sequential colormap
                # from white (0) to purple (min).
                norm = colors.Normalize(vmin=min_val, vmax=0)
                cmap = colors.LinearSegmentedColormap.from_list(
                    "white_purple", ["purple", "white"]
                )
            else:
                # Data includes negative values: use a diverging colormap that maps
                # negative values to purple, zero to white, and positive values to forest green.
                norm = colors.TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)
                cmap = colors.LinearSegmentedColormap.from_list(
                    "purple_white_forestgreen", [(0, "purple"), (0.5, "white"), (1, "forestgreen")]
                )
            facecolors = cmap(norm(bir_data))
            display_colorbar = True

        created_ax = False
        if ax is None:
            fig = plt.figure(figsize=(5, 5), constrained_layout=True)
            ax = fig.add_subplot(111, projection='3d')
            created_ax = True
        
        self._apply_3d_style(ax)

        # Plot voxels
        ax.voxels(
            bir_mask,
            facecolors=facecolors,
            edgecolor='none',
            shade=False,
            alpha=voxel_alpha
        )

        # Match aspect ratio to the volume shape
        z_dim, y_dim, x_dim = bir_data.shape
        ax.set_box_aspect((z_dim, y_dim, x_dim))

        # Optionally show axes and labels
        self._configure_3d_axes(ax, axis_order, (z_dim, y_dim, x_dim), show_3d_axes)
        # if show_3d_axes:
        #     if axis_order == (0, 1, 2):
        #         ax.set_xlabel("$z$ (dim=0)")
        #         ax.set_ylabel("$y$ (dim=1)")
        #         ax.set_zlabel("$x$ (dim=2)", labelpad=0)
        #         ax.set_xlim(0, z_dim)
        #         ax.set_ylim(0, y_dim)
        #         ax.set_zlim(0, x_dim)
        #     elif axis_order == (2, 1, 0):
        #         ax.set_xlabel("$x$")
        #         ax.set_ylabel("$y$")
        #         ax.set_zlabel("$z$", labelpad=0)
        #         ax.set_xlim(0, z_dim)
        #         ax.set_ylim(0, y_dim)
        #         ax.set_zlim(0, x_dim)
        # else:
        #     ax.axis('off')

        # Optionally add a colorbar only if the data is not constant
        NO_EXTRA = False
        if display_colorbar and not NO_EXTRA:
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array(bir_data)
            cbar = plt.colorbar(mappable, ax=ax, pad=0.07, shrink=0.7, label='Birefringence')
            
            vmin, vmax = mappable.get_clim()
            # If the data is not diverging (i.e. does not span both negative and positive values),
            # then set the ticks to three evenly spaced values.
            if not (vmin < 0 and vmax > 0):
                ticks = np.linspace(vmin, vmax, 3)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f'{t:.3f}' for t in ticks])

        if not NO_EXTRA:
            ax.set_title("3D Birefringence Distribution")

        if created_ax:
            # plt.tight_layout()
            plt.show()

        return ax, norm

    def display_3d_optic_axis(
        self,
        threshold: float = 0.005,
        stride: int = 1,
        show_3d_axes: bool = False,
        ax=None,
        cax=None,
        title: str = "Optic Axis",
        vector_color=True,
        axis_order: tuple[int, int, int] = (0, 1, 2),
        flip_dim: int | tuple[int, ...] | None = None,
    ) -> None:
        """
        Displays 3D optic-axis vectors (channels 1..3) using a quiver plot.

        The 4D volume is assumed to have shape (4, Z, Y, X):
        - channel 0: birefringence (bir)
        - channels 1,2,3: optic axis components (z, y, x)

        Any flip or axis reordering (via axis_order and flip_dim) is applied 
        consistently to both the data and the voxel coordinates.

        Voxel positions are shifted by 0.5 so that the quiver arrows are centered.
        """
        # Retrieve transformed data and optic axis voxel grid (shape: (Z, Y, X, 3))
        bir_data = self._get_transformed_data(channel=0, flip_dim=flip_dim, axis_order=axis_order)
        positions, directions = self._get_transformed_optic_axes(axis_order=axis_order, flip_dim=flip_dim)

        # If a stride is specified, subsample the data, positions, and directions.
        if stride > 1:
            bir_data = bir_data[::stride, ::stride, ::stride]
            positions = positions[::stride, ::stride, ::stride, :]
            directions = directions[::stride, ::stride, ::stride, :]

        grid_shape = bir_data.shape  # (Z, Y, X)

        # Apply threshold mask (using the birefringence channel)
        mask = np.abs(bir_data) >= threshold  # shape: (Z, Y, X)
        mask_flat = mask.ravel()

        nz = np.nonzero(mask)
        nz_indices = list(zip(nz[0].tolist(), nz[1].tolist(), nz[2].tolist()))
        if DEBUG:
            print("Nonzero indices in bir_data:", nz_indices)
        
        # Get the voxel positions and optic axis directions for voxels above threshold.
        pos = positions[mask_flat].astype(np.float64) + 0.5
        vecs = directions[mask_flat]
        if DEBUG:
            print(f"positions: {pos}")
            print(f"directions: {vecs}")

        if pos.size == 0:
            print("[display_3d_optic_axis] No voxels above threshold.")
            return

        # Determine vector colors.
        if vector_color is True:
            # Compute azimuth from the first two components (assumed to be z and y)
            azimuth = np.arctan2(vecs[:, 1], vecs[:, 0])
            az_norm = ((azimuth + np.pi) % np.pi) / np.pi
            colors_array = cm.hsv(az_norm)
        elif isinstance(vector_color, str):
            colors_array = vector_color
        else:
            colors_array = "red"

        # Create or reuse a 3D axis.
        fig_created = False
        if ax is None:
            fig_created = True
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.set_title(title)
            self._apply_3d_style(ax)

        # Plot the optic-axis vectors with the quiver plot.
        ax.quiver(
            pos[:, 0], pos[:, 1], pos[:, 2],
            vecs[:, 0], vecs[:, 1], vecs[:, 2],
            length=1.0,
            arrow_length_ratio=0.0,  # no arrowheads
            normalize=True,
            pivot="middle",
            colors=colors_array,
            linewidth=3.0
        )

        # Add a colorbar if HSV coloring was used.
        if vector_color is True:
            sm = cm.ScalarMappable(cmap=cm.hsv, norm=colors.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            if cax is not None:
                plt.colorbar(sm, cax=cax, fraction=0.03, pad=0.15, aspect=20, shrink=0.8)
            else:
                plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.15, aspect=20, shrink=0.8)

        # Configure the 3D axes (labels, limits, etc.)
        self._configure_3d_axes(ax, axis_order, grid_shape, show_3d_axes)

        if fig_created:
            plt.tight_layout()
            plt.subplots_adjust(left=0.15, right=0.65, top=0.95, bottom=0.15)
            plt.show()

    def display_3d_optic_axis_og(
        self,
        threshold: float = 0.005,
        stride: int = 1,
        show_3d_axes: bool = False,
        ax=None,
        cax=None,
        title: str = "Optic Axis",
        vector_color=True,
        axis_order: tuple[int, int, int] = (0, 1, 2),
        flip_dim: int | tuple[int, ...] | None = None,
    ) -> None:
        """
        Displays 3D optic-axis vectors (channels 1..3) using a quiver plot.
        
        The 4D volume is assumed to have shape (4, Z, Y, X):
        - channel 0: bir (birefringence)
        - channels 1,2,3: optic axis components (z, y, x)
        
        This method applies any flip or axis reordering (via axis_order and flip_dim) consistently
        to both the data and the voxel coordinates.
        """
        # --- Retrieve and transform data ---
        bir_data = self._get_transformed_data(channel=0, flip_dim=flip_dim, axis_order=axis_order)
        positions, directions = self._get_transformed_optic_axes(axis_order=axis_order, flip_dim=flip_dim)
        grid_shape = bir_data.shape  # (Z, Y, X)
    
        
        # --- Apply threshold mask ---
        if False:
            # --- Compute voxel centers ---
            # Create a grid of voxel indices, shift by 0.5 for centers, and apply stride.
            centers = (np.stack(np.indices(grid_shape), axis=-1).astype(float) + 0.5)[::stride, ::stride, ::stride, :]
            mask = (np.abs(bir_data) >= threshold)[::stride, ::stride, ::stride]
            pos = centers.reshape(-1, 3)[mask.ravel()]
            # Reshape directions to match grid, apply stride and then mask.
            dirs_strided = dirs.reshape(grid_shape + (3,))[::stride, ::stride, ::stride, :]
            vecs = dirs_strided.reshape(-1, 3)[mask.ravel()]

        mask_3d = (np.abs(bir_data) >= threshold)
        mask_1d = mask_3d.ravel()  # Flatten

        pos = positions[mask_1d]
        vecs = directions[mask_1d]
        print(f"pos.shape: {pos.shape}, vecs.shape: {vecs.shape}")
        print(f"positions: {pos}")
        print(f"directions: {vecs}")
        

        
        if pos.size == 0:
            print("[display_3d_optic_axis] No voxels above threshold.")
            return
        
        # --- Determine vector colors ---
        if isinstance(vector_color, bool) and vector_color:
            # Compute azimuth angle from the first two components (Z, Y)
            az_norm = (((np.arctan2(vecs[:, 1], vecs[:, 0]) + np.pi) % np.pi) / np.pi)
            colors_array = cm.hsv(az_norm)
        elif isinstance(vector_color, str):
            colors_array = vector_color
        else:
            colors_array = "red" # "steelblue"
        
        # --- Create or reuse a 3D axis ---
        fig_created = False
        if ax is None:
            fig_created = True
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.set_title(title)
            self._apply_3d_style(ax)
        
        # --- Plot the optic-axis vectors ---
        ax.quiver(
            pos[:, 0], pos[:, 1], pos[:, 2],
            vecs[:, 0], vecs[:, 1], vecs[:, 2],
            length=1.0,
            arrow_length_ratio=0.0,  # no arrowheads
            normalize=True,
            pivot="middle",
            colors=colors_array,
            linewidth=3.0
        )
        
        # --- Add colorbar if using HSV coloring ---
        if isinstance(vector_color, bool) and vector_color:
            sm = cm.ScalarMappable(cmap=cm.hsv, norm=colors.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            if cax is not None:
                plt.colorbar(sm, cax=cax, fraction=0.03, pad=0.15, aspect=20, shrink=0.8)
            else:
                plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.15, aspect=20, shrink=0.8)
        
        # --- Configure 3D axes ---
        self._configure_3d_axes(ax, axis_order, grid_shape, show_3d_axes)
        # if show_3d_axes:
        #     if axis_order == (0, 1, 2):
        #         ax.set_xlabel("$z$ (dim=0)")
            #     ax.set_ylabel("$y$ (dim=1)")
            #     ax.set_zlabel("$x$ (dim=2)")
            #     ax.set_xlim(0, grid_shape[0])
            #     ax.set_ylim(0, grid_shape[1])
            #     ax.set_zlim(0, grid_shape[2])
            #     ax.set_box_aspect(grid_shape)
            # elif axis_order == (2, 1, 0):
            #     ax.set_xlabel("$x$")
            #     ax.set_ylabel("$y$")
            #     ax.set_zlabel("$z$")
            #     ax.set_xlim(0, grid_shape[2])
            #     ax.set_ylim(0, grid_shape[1])
        #         ax.set_zlim(0, grid_shape[0])
        # else:
        #     ax.axis("off")
        
        # Only show the plot if a new figure was created.
        if fig_created:
            plt.tight_layout()
            plt.subplots_adjust(left=0.15, right=0.65, top=0.95, bottom=0.15)
            plt.show()

    def display_3d_volume(
        self,
        threshold=0.0002,
        show_3d_axes=False,
        ax=None,
        axis_order: tuple[int, int, int] = (0, 1, 2),
        flip_dim: int | tuple[int, ...] | None = None
    ):
        """
        Plot a combined view of 3D volume (birefringence voxels) + optic axis lines.
        
        Args:
            threshold (float): min |birefringence| to show a voxel or axis.
            show_3d_axes (bool): if True, show labeled axes on the 3D plot.
        """
        fig_created = False
        if ax is None:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
            fig_created = True

        ax_3d, norm = self.display_3d_birefringence(
            threshold=threshold,
            show_3d_axes=show_3d_axes,
            axis_order=axis_order,
            flip_dim=flip_dim,
            ax=ax,
            voxel_alpha=0.2 #0.14
        )
        # 5) Overlay the optic axis lines, single color
        self.display_3d_optic_axis(
            threshold=threshold,
            ax=ax_3d,
            show_3d_axes=show_3d_axes,
            vector_color="steelblue",
            axis_order=axis_order,
            flip_dim=flip_dim
        )

        NO_EXTRA = False
        if not NO_EXTRA:
            ax_3d.set_title("3D Volume: Birefringence & Optic Axis")

        # ax_3d.xaxis.pane.fill = False
        # ax_3d.yaxis.pane.fill = False
        # ax_3d.zaxis.pane.fill = False
        # ax_3d.xaxis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 0.3)
        # ax_3d.yaxis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 0.3)
        # ax_3d.zaxis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 0.3)

        camera_elev = 30
        camera_azim = -50#-70
        ax_3d.view_init(elev=camera_elev, azim=camera_azim)

        if fig_created:
            plt.tight_layout()
            plt.show()
            return fig, ax_3d

        return ax_3d
