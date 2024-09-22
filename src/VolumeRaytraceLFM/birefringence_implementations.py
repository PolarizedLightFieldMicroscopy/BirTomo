"""Birefringence implementations for VolumeRaytraceLFM.
This module contains the BirefringentVolume class
and BirefringentRaytraceLFM class
"""

from math import floor
from tqdm import tqdm
import time
from collections import Counter
from VolumeRaytraceLFM.abstract_classes import *
from VolumeRaytraceLFM.birefringence_base import BirefringentElement
from VolumeRaytraceLFM.nerf import (
    ImplicitRepresentationMLP,
    ImplicitRepresentationMLPSpherical,
)
from VolumeRaytraceLFM.file_manager import VolumeFileManager
from VolumeRaytraceLFM.volumes.modification import (
    pad_to_region_shape,
    crop_to_region_shape,
)
from VolumeRaytraceLFM.volumes.generation import (
    generate_single_voxel_volume,
    generate_random_volume,
    generate_planes_volume,
    generate_ellipsoid_volume,
)
from VolumeRaytraceLFM.volumes.optic_axis import (
    spherical_to_unit_vector_torch,
    unit_vector_to_spherical,
    fill_vector_based_on_nonaxial,
    adjust_optic_axis_positive_axial,
)
from VolumeRaytraceLFM.jones.jones_calculus import (
    JonesMatrixGenerators,
    JonesVectorGenerators,
)
from VolumeRaytraceLFM.jones.eigenanalysis import (
    retardance_from_su2,
    retardance_from_su2_numpy,
    azimuth_from_jones_torch,
    azimuth_from_jones_numpy,
)
from VolumeRaytraceLFM.jones import jones_matrix
from VolumeRaytraceLFM.utils.dict_utils import filter_keys_by_count, convert_to_tensors
from VolumeRaytraceLFM.utils.error_handling import check_for_negative_values_dict
from VolumeRaytraceLFM.combine_lenslets import (
    gather_voxels_of_rays_pytorch_batch,
    calculate_offsets_vectorized,
)
from VolumeRaytraceLFM.utils.mask_utils import get_bool_mask_for_ray_indices


DEBUG = False
if DEBUG:
    from VolumeRaytraceLFM.utils.error_handling import check_for_inf_or_nan
    from utils import errors


######################################################################
class BirefringentVolume(BirefringentElement):
    """Stores a 3D array of voxels with birefringence properties,
    either with a numpy or pytorch back-end."""

    def __init__(
        self,
        backend=BackEnds.NUMPY,
        torch_args=None,
        optical_info=None,
        Delta_n=0.0,
        optic_axis=None,
        volume_creation_args=None,
    ):
        """BirefringentVolume
        Args:
            backend (BackEnd):
                A computation BackEnd (Numpy vs Pytorch). If Pytorch is used,
                    torch_args are required to initialize the head class OpticBlock.
            torch_args (dict):
                Required for PYTORCH backend. Contains optic_config object and members_to_learn.
                Ex: {'optic_config' : None, 'members_to_learn' : []}
            optical_info (dict):
                volume_shape ([3]:[sz,sy,sz]):
                                        Shape of the volume in voxel numbers per dimension.
                voxel_size_um ([3]):    Size of a voxel in micrometers.
                pixels_per_ml (int):    Number of pixels covered by a microlens
                                            in a light-field system
                na_obj (float):         Numerical aperture of the objective.
                n_medium (float):       Refractive index of immersion medium.
                wavelength (float):     Wavelength of light used.
                Ex: {'volume_shape' : [11,11,11], 'voxel_size_um' : 3*[1.0],
                'pixels_per_ml' : 17, 'na_obj' : 1.2, 'n_medium' : 1.52,
                'wavelength' : 0.550, 'n_micro_lenses' : 1}
            Delta_n (float or [sz,sy,sz] array):
                Defines the birefringence magnitude of the voxels.
                If a float is passed, all the voxels will have the same Delta_n.
            optic_axis ([3] or [3,sz,sy,sz]:
                Defines the optic axis per voxel.
                If a single 3D vector is passed all the voxels will share this optic axis.
            volume_creation_args (dict):
                Containing information on how to init a volume, such as:
                    init_type (str): zeros, nplanes, where n is a number, ellipsoid...
                    init_args (dic): see self.init_volume function for specific arguments
                    per init_type.
        """
        if torch_args is None:
            torch_args = {}
        if optical_info is None:
            optical_info = {}
        if optic_axis is None:
            optic_axis = [1.0, 0.0, 0.0]

        super().__init__(
            backend=backend, torch_args=torch_args, optical_info=optical_info
        )
        self._initialize_volume_attributes(optical_info, Delta_n, optic_axis)
        self.indices_active = None
        self.optic_axis_planar = None

        if volume_creation_args is not None:
            self.init_volume(
                volume_creation_args["init_mode"],
                volume_creation_args.get("init_args", {}),
            )

    def _initialize_volume_attributes(self, optical_info, Delta_n, optic_axis):
        self.volume_shape = optical_info["volume_shape"]
        if self.backend == BackEnds.NUMPY:
            self._initialize_numpy_backend(Delta_n, optic_axis)
        elif self.backend == BackEnds.PYTORCH:
            self._initialize_pytorch_backend(Delta_n, optic_axis)
        else:
            raise ValueError(f"Unsupported backend type: {self.backend}")

    def _initialize_numpy_backend(self, Delta_n, optic_axis):
        # In the case when an optic axis per voxel of a 3D volume is provided, e.g. [3,nz,ny,nx]
        if isinstance(optic_axis, np.ndarray) and len(optic_axis.shape) == 4:
            self._handle_3d_optic_axis_numpy(optic_axis)
            self.Delta_n = Delta_n
            assert (
                len(self.Delta_n.shape) == 3
            ), "3D Delta_n expected, as the optic_axis was provided as a 3D array"
        # Single optic axis, replicate for all voxels
        elif isinstance(optic_axis, list) or isinstance(optic_axis, np.ndarray):
            self._handle_single_optic_axis_numpy(optic_axis)
            # Create Delta_n 3D volume
            self.Delta_n = Delta_n * np.ones(self.volume_shape)

        self.Delta_n[np.isnan(self.Delta_n)] = 0
        self.optic_axis[np.isnan(self.optic_axis)] = 0

    def _initialize_pytorch_backend(self, Delta_n, optic_axis):
        # Normalization of optical axis, depending on input
        if not isinstance(optic_axis, list) and optic_axis.ndim == 4:
            self._handle_3d_optic_axis_torch(optic_axis)
            assert (
                len(Delta_n.shape) == 3
            ), "3D Delta_n expected, as the optic_axis was provided as a 3D torch tensor"
            self.Delta_n = Delta_n
            if not torch.is_tensor(Delta_n):
                self.Delta_n = torch.from_numpy(Delta_n).type(torch.get_default_dtype())
        else:
            self._handle_single_optic_axis_torch(optic_axis)
            self.Delta_n = Delta_n * torch.ones(self.volume_shape)

        # Check for not a number, for when the voxel optic_axis is all zeros
        self.Delta_n[torch.isnan(self.Delta_n)] = 0
        self.optic_axis[torch.isnan(self.optic_axis)] = 0
        # Store the data as pytorch parameters
        self.optic_axis = nn.Parameter(self.optic_axis.reshape(3, -1)).type(
            torch.get_default_dtype()
        )
        self.Delta_n = nn.Parameter(self.Delta_n.flatten()).type(
            torch.get_default_dtype()
        )

    def _handle_3d_optic_axis_numpy(self, optic_axis):
        """Normalize and reshape a 3D optic axis array for Numpy backend."""
        self.volume_shape = optic_axis.shape[1:]
        # Flatten all the voxels in order to normalize them
        optic_axis = optic_axis.reshape(
            3, self.volume_shape[0] * self.volume_shape[1] * self.volume_shape[2]
        ).astype(np.float64)
        for n_voxel in range(len(optic_axis[0, ...])):
            oa_norm = np.linalg.norm(optic_axis[:, n_voxel])
            if oa_norm > 0:
                optic_axis[:, n_voxel] /= oa_norm
        # Set 4D shape again
        self.optic_axis = optic_axis.reshape(3, *self.volume_shape)

    def _handle_single_optic_axis_numpy(self, optic_axis):
        """Set a single optic axis for all voxels for Numpy backend."""
        optic_axis = np.array(optic_axis)
        oa_norm = np.linalg.norm(optic_axis)
        if oa_norm != 0:
            optic_axis /= oa_norm
        self.optic_axis = (
            np.expand_dims(optic_axis, [1, 2, 3])
            .repeat(self.volume_shape[0], 1)
            .repeat(self.volume_shape[1], 2)
            .repeat(self.volume_shape[2], 3)
        )

    def _handle_3d_optic_axis_torch(self, optic_axis):
        """Normalize and reshape a 3D optic axis array for PyTorch backend."""
        if isinstance(optic_axis, np.ndarray):
            optic_axis = torch.from_numpy(optic_axis).type(torch.get_default_dtype())
        oa_norm = torch.sqrt(torch.sum(optic_axis**2, dim=0))
        self.optic_axis = optic_axis / oa_norm.repeat(3, 1, 1, 1)

    def _handle_single_optic_axis_torch(self, optic_axis):
        """Set a single optic axis for all voxels for PyTorch backend."""
        optic_axis = np.array(optic_axis).astype(np.float32)
        oa_norm = np.linalg.norm(optic_axis)
        if oa_norm != 0:
            optic_axis /= oa_norm
        optic_axis_tensor = (
            torch.from_numpy(optic_axis).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        )
        self.optic_axis = optic_axis_tensor.repeat(1, *self.volume_shape)

    def get_delta_n(self):
        """Retrieves the birefringence as a 3D array"""
        if self.backend == BackEnds.PYTORCH:
            return self.Delta_n.view(self.optical_info["volume_shape"])
        else:
            return self.Delta_n

    def get_optic_axis(self):
        """Retrieves the optic axis as a 4D array"""
        if self.backend == BackEnds.PYTORCH:
            return self.optic_axis.view(
                3,
                self.optical_info["volume_shape"][0],
                self.optical_info["volume_shape"][1],
                self.optical_info["volume_shape"][2],
            )
        else:
            return self.optic_axis

    def normalize_optic_axis(self):
        """Normalize the optic axis per voxel."""
        if self.backend == BackEnds.PYTORCH:
            with torch.no_grad():
                self.optic_axis.requires_grad = False
                mags = torch.linalg.norm(self.optic_axis, axis=0)
                valid_mask = mags > 0
                self.optic_axis[:, valid_mask].data /= mags[valid_mask]
                self.optic_axis.requires_grad = True
        elif self.backend == BackEnds.NUMPY:
            mags = np.linalg.norm(self.optic_axis, axis=0)
            valid_mask = mags > 0
            self.optic_axis[:, valid_mask] /= mags[valid_mask]

    def __iadd__(self, other):
        """Overload the += operator to sum volumes."""
        # Ensure shapes are compatible
        delta_n_shape = self.get_delta_n().shape
        optic_axis_shape = self.get_optic_axis().shape
        assert (
            delta_n_shape == other.get_delta_n().shape
        ), f"Shape mismatch for Delta_n: {delta_n_shape} vs {other.get_delta_n().shape}"
        assert (
            optic_axis_shape == other.get_optic_axis().shape
        ), f"Shape mismatch for optic_axis: {optic_axis_shape} vs {other.get_optic_axis().shape}"

        # Disable gradients if using PyTorch
        requires_grad = getattr(self.Delta_n, "requires_grad", False)
        if requires_grad:
            torch.set_grad_enabled(False)
            self.Delta_n.requires_grad = False
            self.optic_axis.requires_grad = False

        # Perform the addition
        self.Delta_n += other.Delta_n
        self.optic_axis += other.optic_axis
        # Maybe normalize axis again?

        # Normalize the optic axis
        norm = (
            torch.linalg.norm(self.optic_axis, axis=0)
            if requires_grad
            else np.linalg.norm(self.optic_axis)
        )
        self.optic_axis /= norm

        # Re-enable gradients if they were disabled
        if requires_grad:
            self.Delta_n.requires_grad = True
            self.optic_axis.requires_grad = True
            torch.set_grad_enabled(True)
        return self

    def plot_lines_plotly(
        self,
        colormap="Bluered_r",
        size_scaler=5,
        fig=None,
        draw_spheres=True,
        delta_n_ths=0.5,
        use_ticks=False,
    ):
        """Plots the optic axis as lines and the birefringence as sphere
        at the ends of the lines. Other parameters could be opacity=0.5 or mode='lines'
        Args:
            delta_n_ths (float): proportion of birefringence values to set to zero
                                    after the birefringence has been normalized
        """
        # Fetch local data
        delta_n = self.get_delta_n() * 1
        optic_axis = self.get_optic_axis() * 1
        optical_info = self.optical_info
        # Check if this is a torch tensor
        if not isinstance(delta_n, np.ndarray):
            try:
                delta_n = delta_n.cpu().detach().numpy()
                optic_axis = optic_axis.cpu().detach().numpy()
            except:
                pass
        delta_n /= np.max(np.abs(delta_n))
        delta_n[np.abs(delta_n) < delta_n_ths] = 0

        import plotly.graph_objects as go

        volume_shape = optical_info["volume_shape"]
        if "voxel_size_um" not in optical_info:
            optical_info["voxel_size_um"] = [1, 1, 1]
            print(
                "Notice: 'voxel_size_um' was not found in optical_info. Size of [1, 1, 1] assigned."
            )
        volume_size_um = [
            optical_info["voxel_size_um"][i] * optical_info["volume_shape"][i]
            for i in range(3)
        ]
        [dz, dxy, dxy] = optical_info["voxel_size_um"]
        use_microns = False
        if use_microns:
            volume_size = volume_size_um
        else:
            volume_size = optical_info["volume_shape"]

        # Sometimes the volume_shape is causing an error when being used as the nticks parameter
        if use_ticks:
            scene_dict = dict(
                xaxis={"nticks": volume_shape[0], "range": [0, volume_size[0]]},
                yaxis={"nticks": volume_shape[1], "range": [0, volume_size[1]]},
                zaxis={"nticks": volume_shape[2], "range": [0, volume_size[2]]},
                xaxis_title="Axial dimension",
                aspectratio={
                    "x": volume_size_um[0],
                    "y": volume_size_um[1],
                    "z": volume_size_um[2],
                },
                aspectmode="manual",
            )
        else:
            scene_dict = dict(
                xaxis_title="Axial dimension",
                aspectratio={
                    "x": volume_size_um[0],
                    "y": volume_size_um[1],
                    "z": volume_size_um[2],
                },
                aspectmode="manual",
            )

        # Define grid
        coords = np.indices(np.array(delta_n.shape)).astype(float)
        if use_microns:
            voxel_length = optical_info["voxel_size_um"]
        else:
            voxel_length = [1, 1, 1]
        coords_base = [
            (coords[i] + 0.5) * voxel_length[i] for i in range(3)
        ]
        coords_tip = [
            (coords[i] + 0.5 + optic_axis[i, ...] * delta_n * 0.75)
            * voxel_length[i]
            for i in range(3)
        ]

        # Plot single line per voxel, where it's length is delta_n
        z_base, y_base, x_base = coords_base
        z_tip, y_tip, x_tip = coords_tip

        # Don't plot zero values
        mask = delta_n == 0
        x_base[mask] = np.nan
        y_base[mask] = np.nan
        z_base[mask] = np.nan
        x_tip[mask] = np.nan
        y_tip[mask] = np.nan
        z_tip[mask] = np.nan

        # Gather all rays in single arrays, to plot them all at once, placing NAN in between them
        array_size = 3 * len(x_base.flatten())
        # Prepare colormap
        all_x = np.empty((array_size))
        all_x[::3] = x_base.flatten()
        all_x[1::3] = x_tip.flatten()
        all_x[2::3] = np.nan
        all_y = np.empty((array_size))
        all_y[::3] = y_base.flatten()
        all_y[1::3] = y_tip.flatten()
        all_y[2::3] = np.nan
        all_z = np.empty((array_size))
        all_z[::3] = z_base.flatten()
        all_z[1::3] = z_tip.flatten()
        all_z[2::3] = np.nan
        # Compute colors
        all_color = np.empty((array_size))
        all_color[::3] = (
            (x_base - x_tip).flatten() ** 2
            + (y_base - y_tip).flatten() ** 2
            + (z_base - z_tip).flatten() ** 2
        )
        # all_color[::3] =  delta_n.flatten() * 1.0
        all_color[1::3] = all_color[::3]
        all_color[2::3] = 0
        all_color[np.isnan(all_color)] = 0

        err = (
            "The BirefringentVolume is expected to have non-zeros values. If the "
            + "BirefringentVolume was cropped to fit into a region, the non-zero values "
            + "may no longer be included."
        )
        assert any(all_color != 0), err

        all_color[all_color != 0] -= all_color[all_color != 0].min()
        all_color += 0.5
        all_color /= all_color.max()

        if fig is None:
            fig = go.Figure()
        fig.add_scatter3d(
            z=all_x,
            y=all_y,
            x=all_z,
            marker={"color": all_color, "colorscale": colormap, "size": 4},
            line={"color": all_color, "colorscale": colormap, "width": size_scaler},
            connectgaps=False,
            mode="lines",
        )
        if draw_spheres:
            fig.add_scatter3d(
                z=x_base.flatten(),
                y=y_base.flatten(),
                x=z_base.flatten(),
                marker={
                    "color": all_color[::3] - 0.5,
                    "colorscale": colormap,
                    "size": size_scaler * 5 * all_color[::3],
                },
                line={
                    "color": all_color[::3] - 0.5,
                    "colorscale": colormap,
                    "width": 5,
                },
                mode="markers",
            )
        camera = {"eye": {"x": 50, "y": 0, "z": 0}}
        fig.update_layout(
            scene=scene_dict,
            scene_camera=camera,
            margin={"r": 0, "l": 0, "b": 0, "t": 0},
            showlegend=False,
        )
        # fig.data = fig.data[::-1]
        # fig.show()
        return fig

    @staticmethod
    def plot_volume_plotly(
        optical_info, voxels_in=None, opacity=0.5, colormap="gray", fig=None
    ):
        """Plots a 3D array with the non-zero voxels shaded."""
        voxels = voxels_in * 1.0
        # Check if this is a torch tensor
        if not isinstance(voxels_in, np.ndarray):
            try:
                voxels = voxels.detach()
                voxels = voxels.cpu().abs().numpy()
            except:
                pass
        voxels = np.abs(voxels)
        err = (
            "The set of voxels are expected to have non-zeros values. If the "
            + "BirefringentVolume was cropped to fit into a region, the non-zero values "
            + "may no longer be included."
        )
        assert voxels.any(), err

        import plotly.graph_objects as go

        volume_shape = optical_info["volume_shape"]
        if "voxel_size_um" not in optical_info:
            optical_info["voxel_size_um"] = [1, 1, 1]
            print(
                "Notice: 'voxel_size_um' was not found in optical_info. Size of [1, 1, 1] assigned."
            )
        volume_size_um = [
            optical_info["voxel_size_um"][i] * optical_info["volume_shape"][i]
            for i in range(3)
        ]
        # Define grid
        coords = np.indices(np.array(voxels.shape)).astype(float)
        # Shift by half a voxel and multiply by voxel size
        coords = [
            (coords[i] + 0.5) * optical_info["voxel_size_um"][i] for i in range(3)
        ]
        if fig is None:
            fig = go.Figure()
        fig.add_volume(
            x=coords[0].flatten(),
            y=coords[1].flatten(),
            z=coords[2].flatten(),
            value=voxels.flatten() / voxels.max(),
            isomin=0,
            isomax=0.1,
            opacity=opacity,  # needs to be small to see through all surfaces
            surface_count=20,  # needs to be a large number for good volume rendering
            colorscale=colormap,
        )
        camera = {"eye": {"x": 50, "y": 0, "z": 0}}
        fig.update_layout(
            scene=dict(
                xaxis={"nticks": volume_shape[0], "range": [0, volume_size_um[0]]},
                yaxis={"nticks": volume_shape[1], "range": [0, volume_size_um[1]]},
                zaxis={"nticks": volume_shape[2], "range": [0, volume_size_um[2]]},
                xaxis_title="Axial dimension",
                aspectratio={
                    "x": volume_size_um[0],
                    "y": volume_size_um[1],
                    "z": volume_size_um[2],
                },
                aspectmode="manual",
            ),
            scene_camera=camera,
            margin={"r": 0, "l": 0, "b": 0, "t": 0},
            autosize=True,
        )
        # fig.data = fig.data[::-1]
        # fig.show()
        return fig

    def get_vox_params(self, vox_idx):
        """vox_idx is a tuple"""
        if isinstance(vox_idx, tuple) and len(vox_idx) == 3:
            axis = self.optic_axis[:, vox_idx[0], vox_idx[1], vox_idx[2]]
        else:
            axis = self.optic_axis[:, vox_idx]
        return self.Delta_n[vox_idx], axis

    @staticmethod
    def init_from_file(h5_file_path, backend=BackEnds.NUMPY, optical_info=None):
        """Loads a birefringent volume from an h5 file and places it in the center of the volume.
        Requires:
            optical_info/volume_shape [3]: shape of the volume in voxels [nz,ny,nx]
            data/delta_n [nz,ny,nx]: Birefringence volumetric information.
            data/optic_axis [3,nz,ny,nx]: Optical axis per voxel.
        """
        file_manager = VolumeFileManager()
        delta_n, optic_axis = file_manager.extract_data_from_h5(h5_file_path)
        region_shape = np.array(optical_info["volume_shape"])
        if (delta_n.shape == region_shape).all():
            pass
        elif (delta_n.shape >= region_shape).all():
            delta_n, optic_axis = crop_to_region_shape(
                delta_n, optic_axis, delta_n.shape, region_shape
            )
        elif (delta_n.shape <= region_shape).all():
            delta_n, optic_axis = pad_to_region_shape(
                delta_n, optic_axis, delta_n.shape, region_shape
            )
        else:
            raise ValueError(
                f"BirefringentVolume has dimensions ({delta_n.shape}) that are not all greater "
                f"than or less than the volume region dimensions ({region_shape}) set for the microscope"
            )
        volume = BirefringentVolume(
            backend=backend,
            optical_info=optical_info,
            Delta_n=delta_n,
            optic_axis=optic_axis,
        )
        return volume

    @staticmethod
    def load_from_file(h5_file_path, backend_type="numpy"):
        """Loads a birefringent volume from an h5 file and places it in the center of the volume.
        Requires:
            data/delta_n [nz,ny,nx]: Birefringence volumetric information.
            data/optic_axis [3,nz,ny,nx]: Optical axis per voxel.
        """
        backend = {"torch": BackEnds.PYTORCH, "numpy": BackEnds.NUMPY}.get(backend_type)

        if backend is None:
            raise ValueError(f"Backend type {backend_type} is not an option.")

        file_manager = VolumeFileManager()
        delta_n, optic_axis, volume_shape, voxel_size_um = (
            file_manager.extract_all_data_from_h5(h5_file_path)
        )
        # Create optical info dictionary
        # TODO: add the remaining variables, notably the voxel size and the cube voxels boolean
        optical_info = dict(
            {
                "volume_shape": volume_shape,
                "voxel_size_um": voxel_size_um,
                "cube_voxels": True,
            }
        )
        # Create volume
        volume_out = BirefringentVolume(
            backend=backend,
            optical_info=optical_info,
            Delta_n=delta_n,
            optic_axis=optic_axis,
        )
        return volume_out

    def save_as_file(
        self, h5_file_path, description="Temporary description", optical_all=False
    ):
        """Store this volume into an h5 file"""
        tqdm.write(f"Saving volume to h5 file: {h5_file_path}")
        self._save_volume(h5_file_path, description, optical_all, "h5")

    def save_as_numpy_arrays(self, filename):
        """Store this volume into a npy file"""
        self._save_volume(filename, optical_all=False, file_format="npz")

    def save_as_tiff(self, filename):
        """Store this volume into a tiff file"""
        self._save_volume(filename, file_format="tiff")

    def _save_volume(
        self,
        file_path,
        description="Temporary description",
        optical_all=False,
        file_format="h5",
    ):
        """Helper method to save volume data in different formats"""
        delta_n, optic_axis = self._get_data_as_numpy_arrays()
        file_manager = VolumeFileManager()

        if file_format == "h5":
            file_manager.save_as_h5(
                file_path,
                delta_n,
                optic_axis,
                self.optical_info,
                description,
                optical_all,
            )
        elif file_format == "npz":
            file_manager.save_as_npz(file_path, delta_n, optic_axis)
        elif file_format == "tiff":
            file_manager.save_as_channel_stack_tiff(file_path, delta_n, optic_axis)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    def _get_data_as_numpy_arrays(self):
        """Converts delta_n and optic_axis based on backend"""
        delta_n = self.get_delta_n()
        optic_axis = self.get_optic_axis()
        if self.backend == BackEnds.PYTORCH:
            delta_n = delta_n.detach().cpu().numpy()
            optic_axis = optic_axis.detach().cpu().numpy()
        return delta_n, optic_axis

    def _get_backend_str(self):
        """Returns the string representation of the backend type."""
        backend_map = {BackEnds.PYTORCH: "pytorch", BackEnds.NUMPY: "numpy"}
        if self.backend in backend_map:
            return backend_map[self.backend]
        else:
            raise ValueError(f"Backend type {self.backend} is not supported.")

    ########### Generate different birefringent volumes ############
    def init_volume(self, init_mode="zeros", init_args={}):
        """This function creates predefined volumes and shapes."""
        volume_shape = self.optical_info["volume_shape"]
        if init_mode == "zeros":
            self._init_zeros(volume_shape)
        elif init_mode == "single_voxel":
            self._init_single_voxel(volume_shape, init_args)
        elif init_mode == "random":
            self._init_random(volume_shape, init_args)
        elif "planes" in init_mode:
            self._init_planes(volume_shape, init_mode, init_args)
        elif init_mode in ["ellipsoid", "shell"]:
            self._init_ellipsoid_or_shell(volume_shape, init_mode, init_args)
        else:
            raise ValueError(f"The init mode {init_mode} has not been created yet.")
        self._set_volume_ref()

    def _init_zeros(self, volume_shape):
        shape = [4] + volume_shape
        if self.backend == BackEnds.NUMPY:
            self.voxel_parameters = np.zeros(shape)
        elif self.backend == BackEnds.PYTORCH:
            self.voxel_parameters = torch.zeros(shape)

    def _init_single_voxel(self, volume_shape, init_args):
        delta_n = init_args.get("delta_n", 0.01)
        optic_axis = init_args.get("optic_axis", [1, 0, 0])
        offset = init_args.get("offset", [0, 0, 0])
        self.voxel_parameters = self.generate_single_voxel_volume(
            volume_shape, delta_n, optic_axis, offset
        )

    def _init_random(self, volume_shape, init_args):
        my_init_args = (
            init_args if init_args else {"Delta_n_range": [0, 1], "axes_range": [-1, 1]}
        )
        self.voxel_parameters = self.generate_random_volume(
            volume_shape, init_args=my_init_args
        )

    def _init_planes(self, volume_shape, init_mode, init_args):
        n_planes = int(init_mode[0])
        z_offset = init_args.get("z_offset", 0)
        delta_n = init_args.get("delta_n", 0.01)
        self.voxel_parameters = self.generate_planes_volume(
            volume_shape, n_planes, z_offset=z_offset, delta_n=delta_n
        )

    def _init_ellipsoid_or_shell(self, volume_shape, init_mode, init_args):
        """Initialize the volume with an ellipsoid or shell shape.
        Args:
            volume_shape (list): Shape of the volume.
            init_mode (str): Initialization mode, either 'ellipsoid' or 'shell'.
            init_args (dict): Arguments for initialization, including
                radius, center, delta_n, and border_thickness.
        """
        radius = init_args.get("radius", [5.5, 5.5, 3.5])
        center = init_args.get("center", [0.5, 0.5, 0.5])
        delta_n = init_args.get("delta_n", 0.01)
        alpha = init_args.get("border_thickness", 1)
        
        self.voxel_parameters = self.generate_ellipsoid_volume(
            volume_shape, center=center, radius=radius, alpha=alpha, delta_n=delta_n
        )
        
        if init_mode == "shell":
            # how tall is the shell top to botom? 
            # the tallness is size like? and must have a -1 when doing index maths
            shell_tallness = int(radius[0]//2) # a good tallness
            shell_tallness = 4
            # how high is the shell flying above the bottom of the volume
            shell_highness = int((volume_shape[0]-shell_tallness)//2) # centered in the volume.
            #shell_highness = 0
            
            # flip = True
            # adjust the center of the elipse so the shell is centered at max_index/2
            center = [(shell_tallness-1+shell_highness-radius[0])/(volume_shape[0]-1)+10*np.finfo(float).eps, center[1], center[2]]  # add some epsilons to encourage the tip of the elipse to show up
            self.voxel_parameters = self.generate_ellipsoid_volume(
                volume_shape, center=center, radius=radius, alpha=alpha, delta_n=delta_n
            )
            # expanded_shape = [2 * volume_shape[0], volume_shape[1], volume_shape[2]]
            # self.voxel_parameters = self.generate_ellipsoid_volume(
            #     expanded_shape, center=center, radius=radius, alpha=alpha, delta_n=delta_n
            # )
            self._apply_shell_modification(radius, shell_highness)

    def _apply_shell_modification(self, radius, shell_highness):
        vol_shape = self.optical_info["volume_shape"]
        # removal_amount = vol_shape[0] + int(radius[0] // 2)
        # removal_amount = 0
        removal_amount = shell_highness

        # set all voxels to zero that are below the 
        # removal_amount hight on the optical axis
        self.voxel_parameters[0, ...][
            : removal_amount, ...
            ] = 0
        # Shift the nonzero contents along the z-axis
        # shift = - vol_shape[0] // 4 - int(radius[0] // 2)
        # shift = - round((radius[0] + radius[0] / 2) / 2)
        # shifted_voxel_parameters = np.roll(self.voxel_parameters, shift=shift, axis=1)
        # self.voxel_parameters = shifted_voxel_parameters
        # # volume_shape = self.optical_info["volume_shape"]
        # # z_center = volume_shape[0] // 2
        
        # # # Ensure the shift does not go out of bounds
        # # z_start = max(0, z_center - volume_shape[0] // 2)
        # # z_end = min(volume_shape[0], z_center + volume_shape[0] // 2 + 1)
        # start_z = vol_shape[0] // 2
        # end_z = start_z + vol_shape[0]
        # self.voxel_parameters = self.voxel_parameters[:, start_z:end_z, ...]

    def _set_volume_ref(self):
        volume_ref = BirefringentVolume(
            backend=self.backend,
            optical_info=self.optical_info,
            Delta_n=self.voxel_parameters[0, ...],
            optic_axis=self.voxel_parameters[1:, ...],
        )
        self.Delta_n = volume_ref.Delta_n
        self.optic_axis = volume_ref.optic_axis
        self.optic_axis = adjust_optic_axis_positive_axial(self.optic_axis)

    @staticmethod
    def generate_single_voxel_volume(
        volume_shape: list[int, int, int],
        delta_n: float = 0.01,
        optic_axis: list = [1, 0, 0],
        offset: list[int, int, int] = [0, 0, 0],
    ):
        """Generates a single voxel volume."""
        # Identity the center of the volume after the shifts
        vox_idx = [s // 2 + o for s, o in zip(volume_shape, offset)]
        # Create a volume of all zeros.
        vol = np.zeros((4,) + tuple(volume_shape))
        # Set the birefringence and optic axis
        vol[0, vox_idx[0], vox_idx[1], vox_idx[2]] = delta_n
        vol[1:, vox_idx[0], vox_idx[1], vox_idx[2]] = np.array(optic_axis)
        return vol

    @staticmethod
    def generate_random_volume(
        volume_shape: list[int, int, int],
        init_args: dict = {"Delta_n_range": [0, 1], "axes_range": [-1, 1]},
    ):
        """Generates a random volume."""
        np.random.seed(42)
        Delta_n = np.random.uniform(*init_args["Delta_n_range"], volume_shape)
        axes = [
            np.random.uniform(*init_args["axes_range"], volume_shape) for _ in range(3)
        ]
        norm_A = np.linalg.norm(axes, axis=0)
        return np.concatenate(
            [np.expand_dims(Delta_n, axis=0)]
            + [np.expand_dims(a / norm_A, axis=0) for a in axes],
            axis=0,
        )

    @staticmethod
    def generate_planes_volume(
        volume_shape: list[int, int, int],
        n_planes: int = 1,
        z_offset: int = 0,
        delta_n: float = 0.01,
    ):
        """Generates a volume with planes."""
        vol = np.zeros((4,) + tuple(volume_shape))
        z_size = volume_shape[0]
        z_ranges = np.linspace(0, z_size - 1, n_planes * 2).astype(int)
        optic_axis = np.random.uniform(-1, 1, (3, *volume_shape))
        vol[1:, ...] = optic_axis / np.linalg.norm(optic_axis, axis=0)

        if n_planes == 1:
            # Birefringence
            vol[0, z_size // 2 + z_offset, :, :] = delta_n
            # Axis
            vol[1, z_size // 2 + z_offset, :, :] = 1
            vol[2, z_size // 2 + z_offset, :, :] = 0
            vol[3, z_size // 2 + z_offset, :, :] = 0
            return vol

        random_data = BirefringentVolume.generate_random_volume([n_planes])
        for z_ix in range(n_planes):
            slice_range = z_ranges[z_ix * 2 : z_ix * 2 + 1]
            expanded_data = np.expand_dims(random_data[:, z_ix], axis=(1, 2, 3))
            vol[:, slice_range] = expanded_data.repeat(volume_shape[1], axis=2).repeat(
                volume_shape[2], axis=3
            )
        return vol

    @staticmethod
    def generate_ellipsoid_volume(
        volume_shape, center=[0.5, 0.5, 0.5], radius=[10, 10, 10], alpha=1, delta_n=0.01
    ):
        """Creates an ellipsoid with optical axis normal to the ellipsoid surface.
        Args:
            center [3]: [cz,cy,cx] from 0 to 1 where 0.5 is the center of the volume_shape.
            radius [3]: in voxels, the radius in z,y,x for this ellipsoid.
            alpha (float): Border thickness.
            delta_n (float): Delta_n value of birefringence in the volume
        Returns:
            vol (np.array): 4D array where the first dimension represents the
                birefringence and optic axis properties, and the last three
                dims represents the 3D spatial locations.
        """
        # Originally grabbed from https://math.stackexchange.com/questions/2931909/normal-of-a-point-on-the-surface-of-an-ellipsoid,
        #   then modified to do the subtraction of two ellipsoids instead.
        vol = np.zeros((4,) + tuple(volume_shape))
        kk, jj, ii = np.meshgrid(
            np.arange(volume_shape[0]),
            np.arange(volume_shape[1]),
            np.arange(volume_shape[2]),
            indexing="ij",
        )
        # shift to center
        kk = (center[0] * (volume_shape[0]-1)) - kk.astype(float)
        jj = (center[1] * (volume_shape[1]-1)) - jj.astype(float)
        ii = (center[2] * (volume_shape[2]-1)) - ii.astype(float)

        # DEBUG: checking the indices
        # np.argwhere(ellipsoid_border == np.min(ellipsoid_border))
        # plt.imshow(ellipsoid_border_mask[int(volume_shape[0] / 2),:,:])
        ellipsoid_border = (
            (kk**2) / (radius[0] ** 2)
            + (jj**2) / (radius[1] ** 2)
            + (ii**2) / (radius[2] ** 2)
        )
        hollow_inner = True
        if hollow_inner:
            ellipsoid_border_mask = np.abs(ellipsoid_border) <= 1
            # The inner radius could also be defined as a scaled version of the outer radius.
            # inner_radius = [0.9 * r for r in radius]
            inner_radius = [r - alpha for r in radius]
            inner_ellipsoid_border = (
                (kk**2) / (inner_radius[0] ** 2)
                + (jj**2) / (inner_radius[1] ** 2)
                + (ii**2) / (inner_radius[2] ** 2)
            )
            inner_mask = np.abs(inner_ellipsoid_border) < 1  # this one is less than so the inverse is >=
        else:
            ellipsoid_border_mask = np.abs(ellipsoid_border - alpha) <= 1 # this line feels wierd and maybe should not have the -alpha

        vol[0, ...] = ellipsoid_border_mask.astype(float)
        # Compute normals
        kk_normal = 2 * kk / radius[0]
        jj_normal = 2 * jj / radius[1]
        ii_normal = 2 * ii / radius[2]
        norm_factor = np.sqrt(kk_normal**2 + jj_normal**2 + ii_normal**2)
        # Avoid division by zero
        norm_factor[norm_factor == 0] = 1
        vol[1, ...] = (kk_normal / norm_factor) * vol[0, ...]
        vol[2, ...] = (jj_normal / norm_factor) * vol[0, ...]
        vol[3, ...] = (ii_normal / norm_factor) * vol[0, ...]
        vol[0, ...] *= delta_n
        # vol = vol.permute(0,2,1,3)
        if hollow_inner:
            # Hollowing out the ellipsoid
            combined_mask = np.logical_and(ellipsoid_border_mask, ~inner_mask)
            vol[0, ...] = vol[0, ...] * combined_mask.astype(float)
        return vol

    @staticmethod
    def create_dummy_volume(
        backend=BackEnds.NUMPY,
        optical_info=None,
        vol_type="shell",
        volume_axial_offset=0,
    ):
        """Create different volumes, some of them randomized. Feel free to add
        your volumes here.
        Args:
            backend: BackEnds.NUMPY or BackEnds.PYTORCH
            optical_info (dict): Stores optical properties, primarily the volume shape.
            vol_type (str): Type of volume to generate. Options include "single_voxel", "zeros",
                            "ellipsoid", and "shell".
            volume_axial_offset (int): A potential offset for the volume on the axial direction.
        Returns:
            volume (BirefringentVolume)
        """
        volume_shape = optical_info["volume_shape"]

        if vol_type in ["single_voxel", "zeros"]:
            if backend == BackEnds.NUMPY:
                raise NotImplementedError(
                    "There is not a NUMPY single_voxel or"
                    + "zeros volume method implemented. Use PYTORCH instead."
                )
            voxel_delta_n = 0.01 if vol_type == "single_voxel" else 0
            return BirefringentVolume.generate_single_voxel_volume(
                volume_shape,
                delta_n=voxel_delta_n,
                optic_axis=[1, 0, 0],
                offset=[volume_axial_offset, 0, 0],
            )
        elif vol_type in ["ellipsoid", "shell"]:
            ellipsoid_args = {
                "radius": [5.5, 9.5, 5.5],
                "center": [
                    volume_axial_offset / optical_info["volume_shape"][0],
                    0.50,
                    0.5,
                ],  # from 0 to 1
                "delta_n": 0.01,
                "border_thickness": 1,
            }
            volume = BirefringentVolume(
                backend=backend,
                optical_info=optical_info,
                volume_creation_args={
                    "init_mode": "ellipsoid",
                    "init_args": ellipsoid_args,
                },
            )
            # Do we want a shell? Let's remove some of the volume
            if vol_type == "shell":
                if backend == BackEnds.PYTORCH:
                    with torch.no_grad():
                        volume.get_delta_n()[
                            : optical_info["volume_shape"][0] // 2 + 2, ...
                        ] = 0
                else:
                    volume.get_delta_n()[
                        : optical_info["volume_shape"][0] // 2 + 2, ...
                    ] = 0
        elif vol_type == "sphere_oct13":
            sphere_args = {
                "radius": [4.5, 4.5, 4.5],
                "center": [
                    volume_axial_offset / optical_info["volume_shape"][0],
                    0.50,
                    0.5,
                ],  # from 0 to 1
                "delta_n": 0.01,
                "border_thickness": 1,
            }
            volume = BirefringentVolume(
                backend=backend,
                optical_info=optical_info,
                volume_creation_args={
                    "init_mode": "ellipsoid",
                    "init_args": sphere_args,
                },
            )
        elif vol_type[-10:] == "ellipsoids":
            n_ellipsoids = int(vol_type[:-10])
            volume = BirefringentVolume(
                backend=backend,
                optical_info=optical_info,
                volume_creation_args={"init_mode": "zeros"},
            )
            for _ in range(n_ellipsoids):
                ellipsoid_args = {
                    "radius": np.random.uniform(0.5, 3.5, [3]),
                    "center": [
                        np.random.uniform(0.35, 0.65),
                    ]
                    + list(np.random.uniform(0.3, 0.70, [2])),
                    "delta_n": np.random.uniform(-0.01, -0.001),
                    "border_thickness": 1,
                }
                new_vol = BirefringentVolume(
                    backend=backend,
                    optical_info=optical_info,
                    volume_creation_args={
                        "init_mode": "ellipsoid",
                        "init_args": ellipsoid_args,
                    },
                )
                volume += new_vol
        elif vol_type == "ellipsoids_random":
            n_ellipsoids = np.random.randint(1, 5)
            volume = BirefringentVolume(
                backend=backend,
                optical_info=optical_info,
                volume_creation_args={"init_mode": "zeros"},
            )
            for _ in range(n_ellipsoids):
                ellipsoid_args = {
                    "radius": np.random.uniform(0.5, 3.5, [3]) * 10,
                    "center": [
                        np.random.uniform(0.35, 0.65),
                    ]
                    + list(np.random.uniform(0.3, 0.70, [2])),
                    "delta_n": np.random.uniform(-0.01, -0.001),
                    "border_thickness": 1 * 3,
                }
                new_vol = BirefringentVolume(
                    backend=backend,
                    optical_info=optical_info,
                    volume_creation_args={
                        "init_mode": "ellipsoid",
                        "init_args": ellipsoid_args,
                    },
                )
                volume += new_vol
        elif vol_type == "sphere":
            sphere_args = {
                "radius": [np.random.uniform(3, 6)] * 3,
                "center": [
                    np.random.uniform(0.35, 0.65),
                ]
                + list(np.random.uniform(0.3, 0.70, [2])),
                "delta_n": -0.01,
                "border_thickness": 1,
            }
            volume = BirefringentVolume(
                backend=backend,
                optical_info=optical_info,
                volume_creation_args={
                    "init_mode": "ellipsoid",
                    "init_args": sphere_args,
                },
            )
        elif vol_type == "small_sphere":
            sphere_args = {
                "radius": [3] * 3,
                "center": [0.5] * 3,
                "delta_n": -0.01,
                "border_thickness": 1,
            }
            volume = BirefringentVolume(
                backend=backend,
                optical_info=optical_info,
                volume_creation_args={
                    "init_mode": "ellipsoid",
                    "init_args": sphere_args,
                },
            )
        elif vol_type == "small_sphere_pos":
            min_x = 0.5 - 0.125
            max_x = 0.5 + 0.124
            sphere_args = {
                "radius": [np.random.uniform(1, 2)] * 3,
                "center": [
                    np.random.uniform(min_x, max_x),
                ]
                + list(np.random.uniform(0.42, 0.55, [2])),
                "delta_n": 0.01,
                "border_thickness": 1,
            }
            volume = BirefringentVolume(
                backend=backend,
                optical_info=optical_info,
                volume_creation_args={
                    "init_mode": "ellipsoid",
                    "init_args": sphere_args,
                },
            )
        elif vol_type == "small_sphere_rand_bir":
            min_x = 0.5 - 0.125
            max_x = 0.5 + 0.124
            sphere_args = {
                "radius": [np.random.uniform(1, 2)] * 3,
                "center": [
                    np.random.uniform(min_x, max_x),
                ]
                + list(np.random.uniform(0.42, 0.55, [2])),
                "delta_n": np.random.uniform(0.005, 0.015),
                "border_thickness": 1,
            }
            volume = BirefringentVolume(
                backend=backend,
                optical_info=optical_info,
                volume_creation_args={
                    "init_mode": "ellipsoid",
                    "init_args": sphere_args,
                },
            )
        else:
            raise NotImplementedError
        return volume


############ Implementations ############
class BirefringentRaytraceLFM(RayTraceLFM, BirefringentElement):
    """This class extends RayTraceLFM, and implements the forward function,
    where voxels contribute to ray's Jones matrices with a retardance and axis
    in a non-commutative matter"""

    def __init__(
        self, backend: BackEnds = BackEnds.NUMPY, torch_args={}, optical_info={}
    ):
        """Initialize the class with attibriutes, including those from RayTraceLFM."""
        super().__init__(
            backend=backend, torch_args=torch_args, optical_info=optical_info
        )

        # Initialize voxel indices
        self.vox_indices_ml_shifted = {}
        self.vox_indices_ml_shifted_all = []
        self.vox_indices_by_mla_idx = {}
        self.vox_indices_by_mla_idx_tensors = {}

        # Initialize ray-related attributes
        self.ray_valid_indices_all = None
        self.MLA_volume_geometry_ready = False
        self.verbose = True
        self.only_nonzero_for_jones = False
        self.mla_execution_times = {}

        # Initialize timing dictionary
        self.times = {
            "ray_trace_through_volume": 0,
            "cummulative_jones": 0,
            "prep_for_cummulative_jones": 0,
            "mask_voxels_of_segs": 0,
            "loop_through_vox_collisions": 0,
            "gather_params_for_voxRayJM": 0,
            "jones_matrix_multiplication": 0,
            "voxRayJM": 0,
            "calc_ret_azim_for_jones": 0,
            "calc_jones": 0,
            "retardance_from_jones": 0,
            "azimuth_from_jones": 0,
            "Diag-Offdiag": 0,
            "Stacking": 0,
        }
        self.check_errors = False
        self.use_nerf = False
        self.inr_model = None
    
    def initialize_nerf_mode(self, use_nerf=True):
        """Initialize the NeRF mode based on the user's preference.
        Args:
            use_nerf (bool): Flag to enable or disable NeRF mode. Default is True.
        """
        self.use_nerf = use_nerf
        if self.use_nerf:
            self.inr_model = ImplicitRepresentationMLP(3, 4, [256, 128, 64])
            # self.inr_model = ImplicitRepresentationMLP(3, 4, [256, 256, 256, 256, 256])
            self.inr_model = ImplicitRepresentationMLPSpherical(3, 3, [256, 256, 256])
            self.inr_model = torch.nn.DataParallel(self.inr_model)
            print("NeRF mode initialized.")
        else:
            self.inr_model = None
            print("NeRF mode is disabled.")

    def save_nerf_model(self, filepath):
        """Save the NeRF model to a file."""
        if self.use_nerf:
            torch.save(self.inr_model.state_dict(), filepath)
            print(f"Saved the NeRF model to {filepath}")
        else:
            print("NERF is not enabled, no model to save.")

    def load_nerf_model(self, filepath, eval_mode=False):
        """Load the NeRF model from a file.
        Args:
            filepath (str): Path to the saved model file.
            eval_mode (bool): Whether to set the model to evaluation mode. Default is False.
        """
        if self.use_nerf:
            self.inr_model.load_state_dict(torch.load(filepath))
            if eval_mode:
                self.inr_model.eval()  # Set the model to evaluation mode if needed
            print(f"Loaded the NeRF model from {filepath}")
        else:
            print("NERF is not enabled, no model to load.")

    def __str__(self):
        info = [
            f"BirefringentRaytraceLFM(backend={self.backend}, optical_info={self.optical_info})",
            f"vox_indices_ml_shifted={self.vox_indices_ml_shifted}",
            f"vox_indices_ml_shifted_all={self.vox_indices_ml_shifted_all}",
            f"MLA_volume_geometry_ready={self.MLA_volume_geometry_ready}",
            f"vox_ctr_idx={self.vox_ctr_idx}",
            f"voxel_span_per_ml={self.voxel_span_per_ml}",
            f"ray_valid_indices[:, 0:3]={self.ray_valid_indices[:, 0:3]}",
            f"ray_valid_indices_all={self.ray_valid_indices_all}",
            f"ray_direction_basis[0][0]={self.ray_direction_basis[0][0]}",
            f"ray_vol_colli_indices[0]={self.ray_vol_colli_indices[0]}",
            f"ray_vol_colli_lengths[0]={self.ray_vol_colli_lengths[0]}",
            f"nonzero_pixels_dict[(0, 0)].shape={self.nonzero_pixels_dict[(0, 0)].shape}",
            f"use_lenslet_based_filtering={self.use_lenslet_based_filtering}",
        ]
        return "\n".join(info)

    def save(self, filepath):
        """Save the BirefringentRaytraceLFM instance to a file"""
        time0 = time.time()
        with open(filepath, "wb") as file:
            pickle.dump(self, file)
        print(f"Rays saved in {time.time() - time0:.0f} seconds to {filepath}")

    def print_timing_info(self, precision=2, unit="ms"):
        rays_times = self.times
        multiplier = 1000 if unit == "ms" else 1
        unit_str = "ms" if unit == "ms" else "s"
        fmt_str = f"{{:,.{precision}f}}"
        print("Time spent in each part of the forward model:")
        print(
            "Raytrace through volume:",
            fmt_str.format(rays_times["ray_trace_through_volume"] * multiplier),
            unit_str,
        )
        print(
            "Generating MLA images (sum of indiv lenslets times):",
            fmt_str.format(sum(self.mla_execution_times.values()) * multiplier),
            unit_str,
        )
        print(
            "\tCummulative Jones matrix:",
            fmt_str.format(rays_times["cummulative_jones"] * multiplier),
            unit_str,
        )
        print(
            "\t\tPrepping section of cumulative Jones matrix:",
            fmt_str.format(rays_times["prep_for_cummulative_jones"] * multiplier),
            unit_str,
        )
        print(
            "\t\tMasking voxels of segments:",
            fmt_str.format(rays_times["mask_voxels_of_segs"] * multiplier),
            unit_str,
        )
        print(
            "\t\tGather params for voxRayJM:",
            fmt_str.format(rays_times["gather_params_for_voxRayJM"] * multiplier),
            unit_str,
        )
        print(
            "\t\tvoxRayJM:",
            fmt_str.format(rays_times["voxRayJM"] * multiplier),
            unit_str,
        )
        print(
            "\t\t\tret & azim for JM:",
            fmt_str.format(rays_times["calc_ret_azim_for_jones"] * multiplier),
            unit_str,
        )
        print(
            "\t\t\tJones matrix calculation:",
            fmt_str.format(rays_times["calc_jones"] * multiplier),
            unit_str,
        )
        print(
            "\t\t\t\tMatrix element calculation:",
            fmt_str.format(rays_times["Diag-Offdiag"] * multiplier),
            unit_str,
        )
        print(
            "\t\t\t\tFilling the Matrix:",
            fmt_str.format(rays_times["Stacking"] * multiplier),
            unit_str,
        )
        print(
            "\t\tLoop through across collisions:",
            fmt_str.format(rays_times["loop_through_vox_collisions"] * multiplier),
            unit_str,
        )
        print(
            "\t\t\tJones matrix multiplication:",
            fmt_str.format(rays_times["jones_matrix_multiplication"] * multiplier),
            unit_str,
        )
        print(
            "\tRetardance from Jones:",
            fmt_str.format(rays_times["retardance_from_jones"] * multiplier),
            unit_str,
        )
        print(
            "\tAzimuth from Jones:",
            fmt_str.format(rays_times["azimuth_from_jones"] * multiplier),
            unit_str,
        )

    def reset_timing_info(self):
        self.mla_execution_times = {key: 0 for key in self.mla_execution_times}
        self.times = {key: 0 for key in self.times}

    def to_device(self, device):
        """Move the BirefringentRaytraceLFM to a device"""
        self.ray_valid_indices = self.ray_valid_indices.to(device)
        self.ray_direction_basis = self.ray_direction_basis.to(device)
        self.ray_vol_colli_lengths = self.ray_vol_colli_lengths.to(device)
        if self.use_nerf:
            self.inr_model = self.inr_model.to(device)

    def get_volume_reachable_region(self):
        """Returns a binary mask where the MLA's can reach into the volume"""
        n_micro_lenses = self.optical_info["n_micro_lenses"]
        n_voxels_per_ml = self.optical_info["n_voxels_per_ml"]
        mask = torch.zeros(self.optical_info["volume_shape"])
        ctr_idx = self.vox_ctr_idx
        if True:
            # Include the ray angle reach
            vox_span_half = int(
                self.voxel_span_per_ml + (n_micro_lenses * n_voxels_per_ml) / 2
            )
            mask[
                :,
                ctr_idx[1] - vox_span_half + 1 : ctr_idx[1] + vox_span_half,
                ctr_idx[2] - vox_span_half + 1 : ctr_idx[2] + vox_span_half,
            ] = 1.0
        else:
            # Do not include the ray angle reach
            n_ml_half = floor(n_micro_lenses * n_voxels_per_ml / 2.0)
            mask[
                :,
                ctr_idx[1] - n_ml_half + 1 : ctr_idx[1] + n_ml_half,
                ctr_idx[2] - n_ml_half + 1 : ctr_idx[2] + n_ml_half,
            ] = 1.0
        return mask.detach()

    def prepare_for_all_rays_at_once(self):
        if self.MLA_volume_geometry_ready:
            print("The geometry for the MLA is already prepared.")
        else:
            self.use_lenslet_based_filtering = False
            if self.vox_indices_by_mla_idx == {}:
                self.store_shifted_vox_indices()
                # self.store_shifted_vox_indices_all() # in progress
            self.store_vox_indices_by_mla_idx()
            self.create_colli_indices_all()
            self.create_ray_valid_indices_all()
            self.replicate_ray_info_each_microlens()
            self.MLA_volume_geometry_ready = True
            print(f"Prepared geometry for all rays at once.")
        self.del_arr_unnecessary_for_all_rays_at_once()

    def del_arr_unnecessary_for_all_rays_at_once(self):
        """Delete unnecessary attributes if they exist."""
        if hasattr(self, "vox_indices_by_mla_idx"):
            self.vox_indices_by_mla_idx = None
        if hasattr(self, "vox_indices_ml_shifted"):
            self.vox_indices_ml_shifted = None
        if hasattr(self, "ray_vol_colli_indices"):
            self.ray_vol_colli_indices = None

    def store_shifted_vox_indices(self):
        """Store the shifted voxel indices for each microlens in a
        dictionary. The shape of the volume is taken from the
        optical_info, which should be the same as the volume used in the
        ray tracing process.

        Returns:
            dict: contains the shifted voxel indices for each microlens

        Class attributes accessed:
        (directly)
        - self.optical_info['n_micro_lenses']
        - self.optical_info['n_voxels_per_ml']
        - self.ray_vol_colli_indices
        (indirectly)
        - self.backend
        - self.optical_info['volume_shape']

        Notes:
        - vox_list may be equivalent to self.vox_indices_ml_shifted[str(current_offset)]
        """
        n_micro_lenses = self.optical_info["n_micro_lenses"]
        n_voxels_per_ml = self.optical_info["n_voxels_per_ml"]
        n_ml_half = floor(n_micro_lenses / 2.0)
        collision_indices = self.ray_vol_colli_indices
        if self.verbose:
            print(f"Storing shifted voxel indices for each microlens:")
            row_iterable = tqdm(
                range(n_micro_lenses),
                desc=f"Computing rows of microlenses for storing voxel indices",
                position=1,
                leave=True,
            )
        else:
            row_iterable = range(n_micro_lenses)
        for ml_ii_idx in row_iterable:
            ml_ii = ml_ii_idx - n_ml_half
            for ml_jj_idx in range(n_micro_lenses):
                ml_jj = ml_jj_idx - n_ml_half
                current_offset = self._calculate_current_offset(
                    ml_ii, ml_jj, n_voxels_per_ml, n_micro_lenses
                )
                mla_index = (ml_jj_idx, ml_ii_idx)
                vox_list = self._gather_voxels_of_rays_pytorch(
                    current_offset, collision_indices
                )
                if mla_index not in self.vox_indices_by_mla_idx.keys():
                    self.vox_indices_by_mla_idx[mla_index] = vox_list
        check_for_negative_values_dict(self.vox_indices_by_mla_idx)
        return self.vox_indices_by_mla_idx

    def store_shifted_vox_indices_all(self):
        """In progress: Store the shifted voxel indices for all
        microlenses at once to be a more computationally efficient
        version of store_shifted_vox_indices()."""
        n_micro_lenses = self.optical_info["n_micro_lenses"]
        n_voxels_per_ml = self.optical_info["n_voxels_per_ml"]
        collision_indices = self.ray_vol_colli_indices
        offsets, mla_indices = calculate_offsets_vectorized(
            n_micro_lenses, n_voxels_per_ml, self.vox_ctr_idx
        )
        vox_lists = gather_voxels_of_rays_pytorch_batch(
            offsets, collision_indices, self.optical_info["volume_shape"], self.backend
        )
        for idx, mla_index in enumerate(map(tuple, mla_indices)):
            self.vox_indices_by_mla_idx[mla_index] = vox_lists[idx]
        check_for_negative_values_dict(self.vox_indices_by_mla_idx)
        return self.vox_indices_by_mla_idx

    def create_colli_indices_all(self):
        """Gather the collision indices for all microlenses at once."""
        vox_indices_by_mla_idx = self.vox_indices_by_mla_idx_tensors
        tensors_to_combine = []
        for key in vox_indices_by_mla_idx:
            tensors_to_combine.extend(vox_indices_by_mla_idx[key])
        if tensors_to_combine:
            giant_tensor = torch.stack(tensors_to_combine, dim=0)
        else:
            giant_tensor = torch.tensor([])
        self.vox_indices_ml_shifted_all = giant_tensor

    def create_ray_valid_indices_all(self):
        """Gather the valid ray indices for all microlenses at once."""
        n_micro_lenses = self.optical_info["n_micro_lenses"]
        n_pixels_per_ml = self.optical_info["pixels_per_ml"]
        self.ray_valid_indices_all = None
        device = self.ray_valid_indices.device
        for ml_ii_idx in range(n_micro_lenses):
            for ml_jj_idx in range(n_micro_lenses):
                if self.ray_valid_indices_all is None:
                    self.ray_valid_indices_all = self.ray_valid_indices.clone()
                else:
                    offset = torch.tensor(
                        [ml_jj_idx * n_pixels_per_ml, ml_ii_idx * n_pixels_per_ml],
                        device=device,
                    ).unsqueeze(1)
                    updated_ray_valid_indices = self.ray_valid_indices + offset
                    self.ray_valid_indices_all = torch.cat(
                        (self.ray_valid_indices_all, updated_ray_valid_indices), dim=1
                    )

    def replicate_ray_info_each_microlens(self):
        """Replicate ray info for all the microlenses"""
        n_micro_lenses = self.optical_info["n_micro_lenses"]
        self.ray_vol_colli_lengths = self.ray_vol_colli_lengths.repeat(
            n_micro_lenses**2, 1
        )
        self.ray_direction_basis = self.ray_direction_basis.repeat(
            1, n_micro_lenses**2, 1
        )

    def store_vox_indices_by_mla_idx(self):
        self.vox_indices_by_mla_idx_tensors = convert_to_tensors(
            self.vox_indices_by_mla_idx
        )

    def filter_from_radiometry(self, radiometry: torch.Tensor):
        """Filter out invalid rays based on radiometry image."""
        err_msg = "The geometry for the entire MLA must be prepared first."
        if not self.MLA_volume_geometry_ready:
            raise RuntimeError(err_msg)
        # Get current ray and voxel data
        ray_indices = self.ray_valid_indices_all
        voxel_indices = self.vox_indices_ml_shifted_all
        collision_lengths = self.ray_vol_colli_lengths
        ray_dir_basis = self.ray_direction_basis
        # Create a mask for valid rays based on radiometry
        radiomask = get_bool_mask_for_ray_indices(ray_indices, radiometry)
        # Filter out invalid rays and update attributes
        self.ray_valid_indices_all = ray_indices[:, radiomask]
        self.vox_indices_ml_shifted_all = voxel_indices[radiomask, :]
        self.ray_vol_colli_lengths = collision_lengths[radiomask, :]
        self.ray_direction_basis = ray_dir_basis[:, radiomask, :]

    def ray_trace_through_volume(
        self,
        volume_in: BirefringentVolume = None,
        all_rays_at_once=False,
        intensity=False,
    ):
        """This function forward projects a whole volume, by iterating through
        the volume in front of each microlens in the system. We compute an offset
        (current_offset) that shifts the volume indices reached by each ray.
        Then we accumulate the images generated by each microlens,
        and concatenate in a final image.

        Args:
            volume_in (BirefringentVolume): The volume to be processed.
            all_rays_at_once (bool): Flag to indicate whether all rays should be processed at once.
            intensity (bool): Flag to indicate whether to generate intensity images.
        Returns:
            list[ImageType]: A list of images resulting from the ray tracing process.
        """
        if all_rays_at_once != self.MLA_volume_geometry_ready:
            raise ValueError(
                "The geometry for the MLA is not prepared appropriately "
                + "for the all_rays_at_once flag. If the flag is True, "
                + "call prepare_for_all_rays_at_once() first."
            )
        start_time_raytrace = time.perf_counter()
        # volume_shape defines the size of the workspace
        # the number of microlenses defines the valid volume inside the workspace
        volume_shape = volume_in.optical_info["volume_shape"]
        n_micro_lenses = self.optical_info["n_micro_lenses"]
        n_voxels_per_ml = self.optical_info["n_voxels_per_ml"]
        n_ml_half = floor(n_micro_lenses / 2.0)

        # Check if the volume_size can fit these microlenses.
        # # considering that some rays go beyond the volume in front of the microlenses
        min_required_volume_size = self._calculate_min_volume_size(
            n_micro_lenses, n_voxels_per_ml
        )
        self._validate_volume_size(min_required_volume_size, volume_shape)

        # Traverse volume for every ray, and generate intensity images or retardance and azimuth images
        if all_rays_at_once:
            if intensity:
                raise NotImplementedError(
                    "Intensity images are not supported for all rays at once."
                )
            full_img_list = self.ret_and_azim_images_mla_torch(volume_in)
        else:
            full_img_list = [None] * 5
            odd_mla_shift = n_micro_lenses % 2  # shift for odd number of microlenses
            row_iterable = self._get_row_iterable(n_ml_half, odd_mla_shift)
            # Iterate over each row of microlenses (y direction)
            for ml_ii_idx, ml_ii in enumerate(row_iterable):
                # Initialize a list for storing concatenated images of the current row
                full_img_row_list = [None] * 5
                # Iterate over each column of microlenses in the current row (x direction)
                for ml_jj_idx, ml_jj in enumerate(
                    range(-n_ml_half, n_ml_half + odd_mla_shift)
                ):
                    current_offset = self._calculate_current_offset(
                        ml_ii, ml_jj, n_voxels_per_ml, n_micro_lenses
                    )
                    img_list = self.generate_images(
                        volume_in,
                        current_offset,
                        intensity,
                        mla_index=(ml_jj_idx, ml_ii_idx),
                    )
                    # Concatenate the generated images with the images of the current row
                    full_img_row_list = self._concatenate_images(
                        full_img_row_list, img_list, axis=0
                    )
                # Concatenate the row images with the full image list
                full_img_list = self._concatenate_images(
                    full_img_list, full_img_row_list, axis=1
                )
        end_time_raytrace = time.perf_counter()
        self.times["ray_trace_through_volume"] += (
            end_time_raytrace - start_time_raytrace
        )
        return full_img_list

    def _get_row_iterable(self, n_ml_half, odd_mla_shift):
        range_iterable = range(-n_ml_half, n_ml_half + odd_mla_shift)
        if self.verbose:
            return tqdm(
                range_iterable,
                desc=f"Computing rows of microlenses {self.backend}",
                position=1,
                leave=False,
            )
        return range_iterable

    def _calculate_min_volume_size(self, num_microlenses, num_voxels_per_ml):
        return int(self.voxel_span_per_ml + (num_microlenses * num_voxels_per_ml))

    def _validate_volume_size(self, min_required_volume_size, volume_shape):
        if (
            min_required_volume_size > volume_shape[1]
            or min_required_volume_size > volume_shape[2]
        ):
            warning_msg = (
                f"WARNING: The required volume size ({min_required_volume_size}) "
                f"exceeds the provided volume shape {volume_shape[1:]}."
            )
            print(warning_msg)
            raise_error = False  # DEBUG: set to False to avoid raising error
            if raise_error:
                raise ValueError(warning_msg)

    def _calculate_current_offset(
        self, row_index, col_index, num_voxels_per_ml, num_microlenses
    ):
        """Maps the position of a microlens in its array to the corresponding
        position in the volumetric data, identified by its row and column
        indices. This function calculates the offset to the top corner of the
        volume in front of the current microlens (row_index, col_index).

        Args:
            row_index (int): The row index of the current microlens in the
                             microlens array.
            col_index (int): The column index of the current microlens in the
                             microlens array.
            num_voxels_per_ml (int): The number of voxels per microlens,
                indicating the size of the voxel area each microlens covers.
            num_microlenses (int): The total number of microlenses in one
                                   dimension of the microlens array.
        Returns:
            np.array: An array representing the calculated offset in the
                      volumetric data for the current microlens.
        """
        # Scale row and column indices to voxel space. This is important when using supersampling.
        scaled_indices = np.array(
            [num_voxels_per_ml * row_index, num_voxels_per_ml * col_index]
        )

        # Add central indices of the volume. This shifts the focus to the relevant part of the volume
        # based on the predefined central indices (vox_ctr_idx).
        central_offset = np.array(self.vox_ctr_idx[1:])

        # Compute the midpoint of the total voxel space covered by the
        #   microlenses. This value is subtracted to center the offset around
        #   the middle of the microlens array.
        half_voxel_span = floor(num_voxels_per_ml * num_microlenses / 2.0)

        # Calculate and return the final offset for the current microlens
        return scaled_indices + central_offset - half_voxel_span

    def generate_images(self, volume, offset, intensity, mla_index=(0, 0)):
        """Generates images for a single microlens, by passing an offset
        to the ray tracing process. The offset shifts the volume indices
        reached by each ray, depending on the microlens position and the
        supersampling factor."""
        start_time = time.time()
        if intensity:
            image_list = self.intensity_images(
                volume, microlens_offset=offset, mla_index=mla_index
            )
        else:
            image_list = self.ret_and_azim_images(
                volume, microlens_offset=offset, mla_index=mla_index
            )
        self._update_mla_execution_time(mla_index, time.time() - start_time)
        return image_list

    def _update_mla_execution_time(self, mla_index, execution_time):
        if mla_index not in self.mla_execution_times:
            self.mla_execution_times[mla_index] = 0
        self.mla_execution_times[mla_index] += execution_time

    def _concatenate_images(self, img_list1, img_list2, axis):
        if img_list1[0] is None:
            return img_list2
        if self.backend == BackEnds.NUMPY:
            return [
                np.concatenate((img1, img2), axis)
                for img1, img2 in zip(img_list1, img_list2)
            ]
        elif self.backend == BackEnds.PYTORCH:
            return [
                torch.concatenate((img1, img2), axis)
                for img1, img2 in zip(img_list1, img_list2)
            ]

    def _measure_time(self, func, *args, **kwargs):
        """Helper function to measure execution time of a function."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time

    def retardance(self, jones):
        """Phase delay introduced between the fast and slow axis in a Jones matrix"""
        if self.backend == BackEnds.NUMPY:
            retardance, duration = self._measure_time(retardance_from_su2_numpy, jones)
        elif self.backend == BackEnds.PYTORCH:
            retardance, duration = self._measure_time(retardance_from_su2, jones)
            if DEBUG:
                assert not torch.isnan(
                    retardance
                ).any(), "Retardance contains NaN values."
        else:
            raise NotImplementedError("Unsupported backend")

        self.times["retardance_from_jones"] += duration
        return retardance

    def azimuth(self, jones):
        """Rotation angle of the fast axis (neg phase)"""
        if self.backend == BackEnds.NUMPY:
            azimuth, duration = self._measure_time(azimuth_from_jones_numpy, jones)
        elif self.backend == BackEnds.PYTORCH:
            azimuth, duration = self._measure_time(azimuth_from_jones_torch, jones)
        else:
            raise NotImplementedError("Unsupported backend")

        self.times["azimuth_from_jones"] += duration
        return azimuth

    def calc_cummulative_JM_of_ray(
        self, volume_in: BirefringentVolume, microlens_offset=[0, 0], mla_index=(0, 0)
    ):
        if self.backend == BackEnds.NUMPY:
            # TODO: check if the function calling is appropriate
            return self.calc_cummulative_JM_of_ray_numpy(volume_in, microlens_offset)
        elif self.backend == BackEnds.PYTORCH:
            return self.calc_cummulative_JM_of_ray_torch(
                volume_in, microlens_offset, mla_index=mla_index
            )

    def calc_cummulative_JM_of_ray_numpy(
        self, i, j, volume_in: BirefringentVolume, microlens_offset=[0, 0]
    ):
        """For the (i,j) pixel behind a single microlens"""
        # Fetch precomputed Siddon parameters
        voxels_of_segs, ell_in_voxels = (
            self.ray_vol_colli_indices,
            self.ray_vol_colli_lengths,
        )
        # rays are stored in a 1D array, let's look for index i,j
        n_ray = j + i * self.optical_info["pixels_per_ml"]
        rayDir = self.ray_direction_basis[n_ray][:]

        jones_list = []
        try:
            for m, vox in enumerate(voxels_of_segs[n_ray]):
                ell = ell_in_voxels[n_ray][m]
                # Check if indices are within bounds
                y_index = vox[1] + microlens_offset[0]
                z_index = vox[2] + microlens_offset[1]
                if not (
                    0 <= y_index < volume_in.Delta_n.shape[1]
                    and 0 <= z_index < volume_in.Delta_n.shape[2]
                ):
                    raise IndexError(
                        f"Cumulative Jones Matrix computation failed. "
                        f"Index out of bounds: Attempted to access Delta_n at index "
                        f"[{vox[0]}, {y_index}, {z_index}], but this is outside "
                        f"the valid range of Delta_n's shape {volume_in.Delta_n.shape}."
                    )

                Delta_n = volume_in.Delta_n[vox[0], y_index, z_index]
                opticAxis = volume_in.optic_axis[:, vox[0], y_index, z_index]
                jones = self.voxRayJM(
                    Delta_n, opticAxis, rayDir, ell, self.optical_info["wavelength"]
                )
                jones_list.append(jones)
        except Exception as e:
            raise Exception(
                "Cumulative Jones Matrix computation failed. "
                + "Error accessing the volume, try increasing the volume size in Y-Z"
            ) from e
        material_jones = BirefringentRaytraceLFM.rayJM_numpy(jones_list)
        return material_jones

    def calc_cummulative_JM_of_ray_torch(
        self,
        volume_in: BirefringentVolume,
        microlens_offset=[0, 0],
        all_rays_at_once=False,
        mla_index=(0, 0),
    ):
        """Computes the cumulative Jones Matrices (JM) for all rays defined in
        a BirefringentVolume object using PyTorch. This function can process
        rays either all at once or individually based on the `all_rays_at_once`
        flag. It uses pytorch's batch dimension to store each ray, and process
        them in parallel.

        Args:
            volume_in (BirefringentVolume): The volume through which rays pass.
            microlens_offset (list, optional): Offset [x, y] for the microlens.
                Defaults to [0, 0].
            all_rays_at_once (bool, optional): If True, processes all rays
                simultaneously. Defaults to False.

        Returns:
            torch.Tensor: The cumulative Jones Matrices for the rays.
                            torch.Size([n_rays_with_voxels, 2, 2])
        """
        if False:  # DEBUG
            assert not all(element == voxels_of_segs[0] for element in voxels_of_segs)
            # Note: if all elements of voxels_of_segs are equal, then all of
            #   self.ray_vol_colli_indices may be equal
        if False:  # DEBUG
            print("DEBUG: making the optical info of volume and self the same")
            print("vol in: ", volume_in.optical_info)
            print("self in: ", self.optical_info)
            print(
                {
                    k: self.optical_info[k] - volume_in.optical_info[k]
                    for k in self.optical_info
                }
            )
            volume_in.optical_info = self.optical_info
            try:
                errors.compare_dicts(self.optical_info, volume_in.optical_info)
            except ValueError:
                print(
                    "Optical info mismatch between ray-tracer and volume. This might cause issues on the border microlenses."
                )

        start_time_cummulative_jones = time.perf_counter()
        material_jones = None
        start_time_prep = time.perf_counter()
        # Get the voxel indices for the provided microlens
        if self.use_lenslet_based_filtering:
            # Mask out the rays that lead to zero pixels
            if False:  # DEBUG
                err_message = f"mla_index {mla_index} is not in nonzero_pixels_dict"
                assert mla_index in self.nonzero_pixels_dict, err_message
            ell_in_voxels, ray_dir_basis, collision_indices = self._filter_ray_data(
                mla_index
            )
            if all_rays_at_once:
                err_message = (
                    "all_rays_at_once not implemented "
                    + "for lenslet-based filtering of rays"
                )
                raise NotImplementedError(err_message)
            voxels_of_segs = self._update_vox_indices_shifted(
                microlens_offset, collision_indices
            )
            err_message = (
                "The list of voxels of segments should be the same "
                + "length as the list of filtered ray volume collision indices."
            )
            assert len(voxels_of_segs) == len(collision_indices), err_message
            if not voxels_of_segs:
                # print("The list 'voxels_of_segs' is empty.")
                max_length = 0
                padded_voxels_of_segs = []
            else:
                max_length = max(len(inner_list) for inner_list in voxels_of_segs)
                # Pad each list to the maximum length and create a tensor
                padded_voxels_of_segs = [
                    inner_list + [-1] * (max_length - len(inner_list))
                    for inner_list in voxels_of_segs
                ]
            voxels_of_segs = torch.tensor(padded_voxels_of_segs, dtype=torch.int)
        else:
            ell_in_voxels = self.ray_vol_colli_lengths
            ray_dir_basis = self.ray_direction_basis
            # Determine voxel indices based on the processing mode. The voxel
            #    indices correspond to the voxels that each ray segment traverses.
            if all_rays_at_once:
                voxels_of_segs = self.vox_indices_ml_shifted_all
            else:
                if mla_index not in self.vox_indices_by_mla_idx_tensors.keys():
                    if mla_index not in self.vox_indices_by_mla_idx.keys():
                        vox_list = self._gather_voxels_of_rays_pytorch(
                            microlens_offset, self.ray_vol_colli_indices
                        )
                        self.vox_indices_by_mla_idx[mla_index] = vox_list
                    voxels_of_segs = self.vox_indices_by_mla_idx[mla_index]
                    max_length = max(len(inner_list) for inner_list in voxels_of_segs)
                    # Pad shorter lists with a specific value (e.g., -1 if -1 is not a valid data point)
                    padded_voxels_of_segs = [
                        inner_list + [-1] * (max_length - len(inner_list))
                        for inner_list in voxels_of_segs
                    ]
                    voxels_of_segs_tensor = torch.tensor(padded_voxels_of_segs)
                    self.vox_indices_by_mla_idx_tensors[mla_index] = (
                        voxels_of_segs_tensor
                    )
                voxels_of_segs = self.vox_indices_by_mla_idx_tensors[mla_index]
        end_time_prep = time.perf_counter()
        self.times["prep_for_cummulative_jones"] += end_time_prep - start_time_prep

        voxels_of_segs_tensor = voxels_of_segs
        if voxels_of_segs_tensor.numel() == 0:
            print("The tensor is empty.")
            valid_voxels_count = torch.tensor([], dtype=torch.int)
        else:
            valid_voxels_mask = voxels_of_segs_tensor != -1
            valid_voxels_count = valid_voxels_mask.sum(dim=1)

        if "mask_voxels_of_segs" not in self.times:
            self.times["mask_voxels_of_segs"] = 0
        self.times["mask_voxels_of_segs"] += time.perf_counter() - end_time_prep

        # Process interactions of all rays with each voxel
        # Iterate the interactions of all rays with the m-th voxel
        # Some rays interact with less voxels,
        #   so we mask the rays valid with rays_with_voxels.

        alt_props = False
        if volume_in.indices_active is not None:
            alt_props = True
        try:
            start_time_gather_params = time.perf_counter()
            # Extract the birefringence and optic axis information from the volume
            if self.use_nerf:
                Delta_n, opticAxis = self.retrieve_properties_from_vox_idx_mlp(
                    volume_in, voxels_of_segs_tensor.long()
                )
            else:
                Delta_n, opticAxis = self.retrieve_properties_from_vox_idx(
                    volume_in, voxels_of_segs_tensor.long(), active_props_only=alt_props
                )
            end_time_gather_params = time.perf_counter()
            self.times["gather_params_for_voxRayJM"] += (
                end_time_gather_params - start_time_gather_params
            )

            # Compute the interaction from the rays with their corresponding voxels
            jones = self.voxRayJM(
                Delta_n=Delta_n,
                opticAxis=opticAxis,
                rayDir=ray_dir_basis,
                ell=ell_in_voxels,
                wavelength=self.optical_info["wavelength"],
            )

            start_time_mloop = time.perf_counter()
            material_jones = jones[:, 0]
            for m in range(1, ell_in_voxels.shape[1]):
                if DEBUG:
                    # Determine which rays have remaining voxels to traverse
                    rays_with_voxels = valid_voxels_count > m
                    assert rays_with_voxels.all(), "Rays with voxels not found."
                # Combine the current Jones Matrix with the cumulative one
                start_time_jones_mult = time.perf_counter()
                material_jones = material_jones @ jones[:, m]
                end_time_jones_mult = time.perf_counter()
                self.times["jones_matrix_multiplication"] += (
                    end_time_jones_mult - start_time_jones_mult
                )
        except IndexError:
            raise IndexError(
                f"Cumulative Jones Matrix computation failed. "
                f"Voxel index out of bounds for Delta_n of shape {volume_in.Delta_n.shape}."
            )
        except (AssertionError, RuntimeError) as e:
            raise type(e)(
                f"{type(e).__name__} in cumulative Jones Matrix computation: {e}"
            )
        except Exception as e:
            raise Exception(f"Cumulative Jones Matrix computation failed: {e}")
        end_time_mloop = time.perf_counter()
        self.times["loop_through_vox_collisions"] += end_time_mloop - start_time_mloop
        end_time_cummulative_jones = time.perf_counter()
        self.times["cummulative_jones"] += (
            end_time_cummulative_jones - start_time_cummulative_jones
        )
        return material_jones

    def retrieve_properties_from_vox_idx(
        self, volume: BirefringentVolume, vox: torch.Tensor, active_props_only=False
    ):
        """Retrieves the birefringence and optic axis from the volume based on the
        provided voxel indices. This function is used to retrieve the properties
        of the voxels that each ray segment interacts with."""
        if active_props_only:
            device = volume.birefringence_active.device
            idx_tensor = volume.active_idx2spatial_idx_tensor  # .to(device)
            indices = idx_tensor[vox]
            safe_indices = torch.clamp(indices, min=0)
            mask = indices >= 0
            Delta_n = torch.where(
                mask,
                volume.birefringence_active[safe_indices],
                torch.tensor(0.0, device=device),
            )

            if volume.optic_axis_planar is not None:
                opticAxis = torch.empty(
                    (3, *indices.shape), dtype=torch.get_default_dtype(), device=device
                )
                opticAxis[0, :, :] = torch.where(
                    mask,
                    volume.optic_axis_active[0, safe_indices],
                    torch.tensor(0.0, device=device),
                )
                opticAxis[1:, :, :] = torch.where(
                    mask.unsqueeze(0),
                    volume.optic_axis_planar[:, safe_indices],
                    torch.tensor(0.0, device=device),
                )
            else:
                opticAxis = torch.where(
                    mask.unsqueeze(0),
                    volume.optic_axis_active[:, safe_indices],
                    torch.tensor(0.0, device=device),
                )
        else:
            Delta_n = volume.Delta_n[vox]
            opticAxis = volume.optic_axis[:, vox]

        return Delta_n, opticAxis.permute(1, 0, 2)

    def retrieve_properties_from_vox_idx_mlp(self, volume, vox):
        """Retrieves the birefringence and optic axis from the volume
        based on the provided voxel indices using an MLP. This function
        is used to retrieve the properties of the voxels that each ray
        segment interacts with.

        Args:
            volume (BirefringentVolume): Birefringent volume object.
            vox (torch.Tensor): Voxel indices in 1D.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Birefringence and optic axis.
        """
        vol_shape = self.optical_info["volume_shape"]
        filtered_vox = vox[self.mask[vox]]
        vox_copy = filtered_vox.clone()
        vox_3d = RayTraceLFM.unravel_index(vox_copy, vol_shape)
        vox_3d_float = vox_3d.float().to(volume.Delta_n.device)

        # Normalize the input coordinates based on volume shape
        vol_shape_tensor = torch.tensor(
            vol_shape, dtype=vox_3d_float.dtype, device=vox_3d_float.device
        )
        vox_3d_float = vox_3d_float / vol_shape_tensor

        # Pass the input through the MLP
        properties_at_3d_position = self.inr_model(vox_3d_float)

        # Retrieve Delta_n and opticAxis from the MLP output
        Delta_n_filtered = properties_at_3d_position[..., 0]
        if properties_at_3d_position.shape[-1] == 3:
            spherical_angles = properties_at_3d_position[..., 1:]
            opticAxis_filtered = spherical_to_unit_vector_torch(spherical_angles)
        else:
            opticAxis_filtered = properties_at_3d_position[..., 1:]

        # Initialize with zeros and fill in with the filtered values
        Delta_n = torch.zeros(
            vox.shape, dtype=Delta_n_filtered.dtype, device=Delta_n_filtered.device
        )
        opticAxis = torch.zeros(
            (*vox.shape, 3),
            dtype=opticAxis_filtered.dtype,
            device=opticAxis_filtered.device,
        )
        Delta_n[self.mask[vox]] = Delta_n_filtered
        opticAxis[self.mask[vox], :] = opticAxis_filtered
        return Delta_n, opticAxis.permute(0, 2, 1)

    def _get_default_jones(self):
        """Returns the default Jones Matrix for a ray that does not
        interact with any voxels. This is the identity matrix.
        """
        if self.backend == BackEnds.NUMPY:
            return np.array([[1, 0], [0, 1]])
        elif self.backend == BackEnds.PYTORCH:
            return torch.eye(2, dtype=torch.complex64)
        else:
            raise ValueError("Unsupported backend")

    def _update_vox_indices_shifted(self, microlens_offset, collision_indices):
        """
        Updates or retrieves the shifted voxel indices based on the microlens offset.
        The 3D voxel indices are converted to 1D indices for faster access.
        Args:
            microlens_offset (list): Offset [x, y] for the microlens.
            collision_indices (list): The indices of the voxels that each ray
                                        segment traverses.
        Returns:
            list: The shifted voxel indices in 1D.
        """
        # Compute the 1D index for each microlens and store for later use
        #   Accessing 1D arrays increases training speed by 25%
        key = str(microlens_offset)
        if key not in self.vox_indices_ml_shifted:
            self.vox_indices_ml_shifted[key] = [
                [
                    RayTraceLFM.ravel_index(
                        (
                            vox[ix][0],
                            vox[ix][1] + microlens_offset[0],
                            vox[ix][2] + microlens_offset[1],
                        ),
                        self.optical_info["volume_shape"],
                    )
                    for ix in range(len(vox))
                ]
                for vox in collision_indices
            ]

        return self.vox_indices_ml_shifted[key]

    def _gather_voxels_of_rays_pytorch(self, microlens_offset, collision_indices):
        """Gathers the shifted voxel indices based on the microlens offset.

        Args:
            microlens_offset (list): Offset [y, z] in volume space due to the
                                     microlens position.
            collision_indices (list): The indices of the voxels that each ray
                                      segment traverses.

        Returns:
            list: The shifted voxel indices in 1D.

        Class attributes accessed:
        - self.backend: The backend used for computation.
        - self.optical_info['volume_shape']: The shape of the volume.
        """
        err_msg = "This function is for PyTorch backend only."
        assert self.backend == BackEnds.PYTORCH, err_msg
        vol_shape = self.optical_info["volume_shape"]

        # TODO: test tensor_method to see if equivalent
        tensor_method = False
        if tensor_method:
            # perform original method for comparison
            list_of_voxel_lists_og = [
                [
                    RayTraceLFM.ravel_index(
                        (
                            vox[ix][0],
                            vox[ix][1] + microlens_offset[0],
                            vox[ix][2] + microlens_offset[1],
                        ),
                        vol_shape,
                    )
                    for ix in range(len(vox))
                ]
                for vox in collision_indices
            ]

            def ravel_index_tensor(x, dims):
                """Convert multi-dimensional indices to a 1D index using PyTorch operations."""
                dims = torch.tensor(dims, dtype=torch.long, device=x.device)
                c = torch.cumprod(
                    torch.cat((torch.tensor([1], device=x.device), dims.flip(0))), 0
                )[:-1].flip(0)
                if x.ndim == 1:
                    return torch.sum(c * x)  # torch.dot(c, x)
                elif x.ndim == 2:
                    return torch.sum(c * x, dim=1)
                else:
                    raise ValueError("Input tensor x must be 1D or 2D")

            # Convert microlens_offset and collision_indices to tensors
            microlens_offset = torch.tensor(
                microlens_offset,
                dtype=torch.long,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            collision_indices = [
                torch.tensor(
                    vox,
                    dtype=torch.long,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                for vox in collision_indices
            ]

            # Create offsets tensor
            offsets = torch.zeros(
                (len(collision_indices), 3),
                dtype=torch.long,
                device=microlens_offset.device,
            )
            offsets[:, 1] = microlens_offset[0]
            offsets[:, 2] = microlens_offset[1]

            # Shift voxel indices
            shifted_vox = [vox + offsets[i] for i, vox in enumerate(collision_indices)]

            # Check for out-of-bounds indices
            for shifted in shifted_vox:
                if torch.any(shifted >= torch.tensor(vol_shape, device=shifted.device)):
                    print("WARNING: Index out of bounds. Skipping this voxel.")
                    continue

            # Convert the shifted voxel indices to 1D using ravel_index_tensor
            raveled_indices = [
                ravel_index_tensor(shifted, vol_shape) for shifted in shifted_vox
            ]
            assert (
                list_of_voxel_lists == list_of_voxel_lists_og
            ), "The tensor method does not match the original method."
            return raveled_indices
        else:
            if DEBUG:
                list_of_voxel_lists = [
                    [
                        RayTraceLFM.safe_ravel_index(
                            vox[ix], microlens_offset, vol_shape
                        )
                        for ix in range(len(vox))
                    ]
                    for vox in collision_indices
                ]
            else:
                list_of_voxel_lists = [
                    [
                        RayTraceLFM.ravel_index(
                            (
                                vox[ix][0],
                                vox[ix][1] + microlens_offset[0],
                                vox[ix][2] + microlens_offset[1],
                            ),
                            vol_shape,
                        )
                        for ix in range(len(vox))
                    ]
                    for vox in collision_indices
                ]
        return list_of_voxel_lists

    def _count_vox_raytrace_occurrences(
        self,
        zero_ret_voxels=False,
        nonzero_ret_voxels=False,
        zero_ret_entire_lenslet_voxels=False,
    ):
        """Counts occurances of voxels from ray tracing.
        Iterates over micro-lenses, aggregates voxel indices,
        and counts occurrences. Optionally filters for voxels
        leading to zero retardance.

        Args:
            zero_retardance_voxels (bool): If True, filters for
            voxels not contributing to nonzero retardance.

        Returns:
            Counter: Counts of voxel indices.

        Class attributes accessed:
        (directly)
        - self.optical_info['n_micro_lenses']
        - self.vox_indices_by_mla_idx
        (indirectly)
        - self.ray_valid_indices
        - self.nonzero_pixels_dict
        """
        assert (
            self.vox_indices_by_mla_idx
        ), "Voxel indices data must be populated first."
        n_micro_lenses = self.optical_info["n_micro_lenses"]
        count = Counter()
        for ml_ii_idx in range(n_micro_lenses):
            for ml_jj_idx in range(n_micro_lenses):
                mla_index = (ml_jj_idx, ml_ii_idx)
                vox_indices = self.vox_indices_by_mla_idx[mla_index]

                if zero_ret_voxels:
                    # Get the boolean mask for the pixels that are not zero
                    nonzero_mask = self._form_mask_from_nonzero_pixels_dict(mla_index)
                    tensor_method = False
                    if tensor_method:
                        # In progress method to use tensors for faster computation
                        vox_indices_tensor = self.vox_indices_by_mla_idx_tensors[
                            mla_index
                        ]
                        vox_indices = np.array(vox_indices_tensor)
                        zero_ret_vox_indices = vox_indices[~nonzero_mask].tolist()
                    else:
                        # Find the voxels that lead to a zero pixel in the retardance image
                        zero_ret_vox_indices = [
                            vox_indices[i]
                            for i, nonzero_bool in enumerate(nonzero_mask)
                            if not nonzero_bool
                        ]
                    vox_indices = zero_ret_vox_indices
                elif zero_ret_entire_lenslet_voxels:
                    # Get the boolean mask for the pixels that are not zero
                    nonzero_mask = self._form_mask_from_nonzero_pixels_dict(mla_index)
                    # Find the voxels that lead to a zero retardance lenslet image
                    if np.all(~nonzero_mask):
                        # Find the voxels that lead to a zero pixel in the retardance image
                        zero_ret_vox_indices = [
                            vox_indices[i]
                            for i, nonzero_bool in enumerate(nonzero_mask)
                            if not nonzero_bool
                        ]
                        vox_indices = zero_ret_vox_indices
                    else:
                        vox_indices = [[]]
                elif nonzero_ret_voxels:
                    nonzero_mask = self._form_mask_from_nonzero_pixels_dict(mla_index)
                    # Find the voxels that lead to a non-zero pixel in the retardance image
                    nonzero_ret_vox_indices = [
                        vox_indices[i]
                        for i, nonzero_bool in enumerate(nonzero_mask)
                        if nonzero_bool
                    ]
                    vox_indices = nonzero_ret_vox_indices

                flat_list = [item for sublist in vox_indices for item in sublist]
                count.update(flat_list)
                # print(f"mla_index: {mla_index}, counter: {count}")
        return count

    def identify_voxels_repeated_zero_ret(self):
        counts = self._count_vox_raytrace_occurrences(zero_ret_voxels=True)
        if False:  # DEBUG
            print("DEBUG: ", sorted([(key, count) for key, count in counts.items()]))
        vox_list = filter_keys_by_count(counts, 2)
        return vox_list

    def identify_voxels_zero_ret_lenslet(self):
        counts = self._count_vox_raytrace_occurrences(
            zero_ret_entire_lenslet_voxels=True
        )
        if False:  # DEBUG
            print("DEBUG: ", sorted([(key, count) for key, count in counts.items()]))
        vox_list = filter_keys_by_count(counts, 2)
        return vox_list

    def identify_voxels_at_least_one_nonzero_ret(self):
        counts = self._count_vox_raytrace_occurrences(nonzero_ret_voxels=True)
        vox_list = filter_keys_by_count(counts, 1)
        return vox_list

    def _filter_ray_data(self, mla_index: tuple[int, int]) -> tuple:
        """
        Extract the ray tracing variables that contribute to the image.
        This is done by applying a mask to the ray tracing variables. No class
        attributes are modified, but several are accessed.

        Args:
            mla_index: Index to identify the relevant non-zero pixel grid from
                       the class attribute `nonzero_pixels_dict`.

        Returns:
            tuple: A tuple containing filtered ell_in_voxels, ray direction
                   basis, and ray volume collision indices.

        Class attributes accessed:
        (directly)
        - self.ray_vol_colli_lengths: Contains lengths of rays through voxels.
        - self.ray_direction_basis: Contains the directions of the rays.
        - self.ray_vol_colli_indices: Contains ray volume collision indices.
        (indirectly)
        - self.ray_valid_indices: Contains the rays that reach the detector.
        - self.nonzero_pixels_dict: A dictionary containing grids of non-zero
                                    pixels, accessed using `mla_index`.
        """
        mask = self._form_mask_from_nonzero_pixels_dict(mla_index)

        assert (
            self.ray_vol_colli_lengths is not None
        ), "Ray data must be populated first."
        assert self.ray_direction_basis is not None, "Ray data must be populated first."
        assert (
            self.ray_vol_colli_indices is not None
        ), "Ray data must be populated first."

        if not mask.any():
            # Return empty tensors with desired shapes
            # first dim would be self.ray_vol_colli_lengths.shape[0]
            ell_in_voxels_filtered = torch.empty(0, 0)
            ray_dir_basis_filtered = torch.empty(3, 0, 3)
            ray_vol_colli_indices_filtered = []
        else:
            # Apply mask to ray data
            ell_in_voxels_filtered = self.ray_vol_colli_lengths[mask]
            ray_dir_basis_filtered = self.ray_direction_basis[:, mask, :]
            colli_indices = self.ray_vol_colli_indices
            ray_vol_colli_indices_filtered = [
                idx for idx, mask_val in zip(colli_indices, mask) if mask_val
            ]

        return (
            ell_in_voxels_filtered,
            ray_dir_basis_filtered,
            ray_vol_colli_indices_filtered,
        )

    def ret_and_azim_images(
        self, volume_in: BirefringentVolume, microlens_offset=[0, 0], mla_index=(0, 0)
    ):
        """Calculate retardance and azimuth values for a ray with a Jones Matrix."""
        if self.backend == BackEnds.NUMPY:
            # TODO: pass mla_index argument into the numpy function
            return self.ret_and_azim_images_numpy(volume_in, microlens_offset)
        elif self.backend == BackEnds.PYTORCH:
            return self.ret_and_azim_images_torch(
                volume_in, microlens_offset, mla_index=mla_index
            )

    def ret_and_azim_images_numpy(
        self, volume_in: BirefringentVolume, microlens_offset=[0, 0]
    ):
        """Calculate retardance and azimuth values for a ray with a Jones Matrix."""
        pixels_per_ml = self.optical_info["pixels_per_ml"]
        ret_image = np.zeros((pixels_per_ml, pixels_per_ml))
        azim_image = np.zeros((pixels_per_ml, pixels_per_ml))
        for i in range(pixels_per_ml):
            for j in range(pixels_per_ml):
                if np.isnan(self.ray_entry[0, i, j]):
                    ret_image[i, j] = 0
                    azim_image[i, j] = 0
                else:
                    effective_jones = self.calc_cummulative_JM_of_ray_numpy(
                        i, j, volume_in, microlens_offset
                    )
                    ret_image[i, j] = self.retardance(effective_jones)
                    if np.isclose(ret_image[i, j], 0.0):
                        azim_image[i, j] = 0
                    else:
                        azim_image[i, j] = self.azimuth(effective_jones)
        return [ret_image, azim_image]

    def ret_and_azim_images_mla_torch(self, volume_in: BirefringentVolume):
        """This function computes the retardance and azimuth images
        of the precomputed rays going through a volume for all rays at once."""

        # Calculate the number of pixels in the microlens array
        pix_per_lenslet = self.optical_info["pixels_per_ml"]
        num_micro_lenses = self.optical_info["n_micro_lenses"]
        pixels_per_mla = pix_per_lenslet * num_micro_lenses

        # Calculate Jones Matrices for all rays
        effective_jones = self.calc_cummulative_JM_of_ray_torch(
            volume_in, all_rays_at_once=True
        )
        # Calculate retardance and azimuth
        retardance = self.retardance(effective_jones).to(torch.float32)
        azimuth = self.azimuth(effective_jones).to(torch.float32)

        # Create output images
        ret_image = torch.zeros(
            (pixels_per_mla, pixels_per_mla),
            dtype=torch.float32,
            device=retardance.device,
        )
        azim_image = torch.zeros(
            (pixels_per_mla, pixels_per_mla),
            dtype=torch.float32,
            device=azimuth.device,
        )

        # Fill the values in the images
        ray_indices_all = self.ray_valid_indices_all
        ret_image[ray_indices_all[0, :], ray_indices_all[1, :]] = retardance
        azim_image[ray_indices_all[0, :], ray_indices_all[1, :]] = azimuth
        return [ret_image, azim_image]

    def ret_and_azim_images_torch(
        self, volume_in: BirefringentVolume, microlens_offset=[0, 0], mla_index=(0, 0)
    ):
        """Computes the retardance and azimuth images for a given volume
        and microlens offset using PyTorch.

        This function calculates the retardance and azimuth values for the
        (precomputed) rays passing through a specific region of the volume,
        as determined by the microlens offset.
        It generates two images: one for retardance and one for azimuth,
        for a single microlens. This offset is included to move the center of
        the volume, as the ray collisions are computed only for a single microlens.
        Args:
            volume_in (BirefringentVolume): The volume through which rays pass.
            microlens_offset (list): The offset [x, y] to the center of the
                                     volume for the specific microlens.
            mla_index (tuple, optional): The index of the microlens.
                                         Defaults to (0, 0).
        Returns:
            list: A list containing two PyTorch tensors, one for the retardance
                    image and one for the azimuth image.
        """
        pixels_per_ml = self.optical_info["pixels_per_ml"]

        # Calculate effective Jones Matrices
        effective_jones = self.calc_cummulative_JM_of_ray(
            volume_in, microlens_offset, mla_index=mla_index
        )

        # Calculate retardance and azimuth from the effective Jones Matrices
        retardance = self.retardance(effective_jones).to(torch.float32)
        azimuth = self.azimuth(effective_jones).to(torch.float32)

        # Initialize output images
        ret_image = torch.zeros(
            (pixels_per_ml, pixels_per_ml),
            dtype=torch.float32,
            device=retardance.device,
        )
        azim_image = torch.zeros(
            (pixels_per_ml, pixels_per_ml),
            dtype=torch.float32,
            device=azimuth.device,
        )

        # Retrieve the ray indices specific to the lenslet
        current_lenslet_indices = self._retrieve_lenslet_indices(mla_index)

        # Fill the calculated values into the images at the lenslet indices
        ret_image[current_lenslet_indices[0, :], current_lenslet_indices[1, :]] = (
            retardance
        )
        azim_image[current_lenslet_indices[0, :], current_lenslet_indices[1, :]] = (
            azimuth
        )
        return [ret_image, azim_image]

    def _retrieve_lenslet_indices(self, mla_index):
        """
        Retrieves the indices of the rays that reach the detector for a given
        microlens. This function will filter out the rays specific to the
        microlens if `use_lenslet_based_filtering` is True.

        Args:
            mla_index (tuple): The index of the microlens.

        Returns:
            current_lenslet_indices (array): The indices of the rays.

        Class attributes accessed:
        - self.ray_valid_indices: Contains the rays that reach the detector.
        - self.nonzero_pixels_dict: A dictionary containing grids of non-zero
                                    pixels, accessed using `mla_index`.
        - self.use_lenslet_based_filtering: A flag to indicate whether to
                                            filter out rays specific to the
                                            microlens.
        """
        if self.use_lenslet_based_filtering:
            # Collect the valid indices specific to the lenslet
            mask = self._form_mask_from_nonzero_pixels_dict(mla_index)
            current_lenslet_indices = self.ray_valid_indices[:, mask]
        else:
            current_lenslet_indices = self.ray_valid_indices
        return current_lenslet_indices

    def _form_mask_from_nonzero_pixels_dict(self, mla_index):
        """Create a boolean mask based on whether or not the image value at
        each index is zero.
        Debugging tips: If the grid is all True, then a subset of the grid
            can be set to False to test the mask.
            Ex 1: nonzero_pixels_grid[0:2, :] = False
            Ex 2: self.nonzero_pixels_dict[(0, 0)][0:2, :] = False

        Args:
            mla_index (tuple): The index of the microlens.
        Returns:
            mask (array): A boolean mask to filter out rays that do not reach
                          the detector or lead to nonzero pixels.
        Class attributes accessed:
        - self.ray_valid_indices: Contains the rays that reach the detector.
        - self.nonzero_pixels_dict: A dictionary containing Boolean grids that
                specify which pixles are nonzero, accessed using `mla_index`.
        """
        err_message = f"mla_index {mla_index} is not in nonzero_pixels_dict"
        assert mla_index in self.nonzero_pixels_dict, err_message
        reshaped_indices = self.ray_valid_indices.T
        nonzero_pixels_grid = self.nonzero_pixels_dict[mla_index]
        mask = np.array(
            [nonzero_pixels_grid[idx[0], idx[1]] for idx in reshaped_indices]
        )
        return mask

    def intensity_images(
        self, volume_in: BirefringentVolume, microlens_offset=[0, 0], mla_index=(0, 0)
    ):
        """Calculate intensity images using Jones Calculus. The polarizer and
        analyzer are applied to the cummulated Jones matrices."""
        analyzer = self.optical_info["analyzer"]
        swing = self.optical_info["polarizer_swing"]
        pixels_per_ml = self.optical_info["pixels_per_ml"]
        lenslet_jones = self.calc_cummulative_JM_lenslet(
            volume_in, microlens_offset, mla_index=mla_index
        )
        intensity_image_list = [np.zeros((pixels_per_ml, pixels_per_ml))] * 5

        for setting in range(5):
            polarizer = JonesMatrixGenerators.universal_compensator_modes(
                setting=setting, swing=swing
            )
            pol_hor = polarizer @ JonesVectorGenerators.horizonal()
            if self.backend == BackEnds.NUMPY:
                E_out = analyzer @ lenslet_jones @ pol_hor
                intensity = np.linalg.norm(E_out, axis=2) ** 2
                intensity_image_list[setting] = intensity
            else:
                intensity_image_list[setting] = torch.zeros(
                    (pixels_per_ml, pixels_per_ml),
                    dtype=torch.float32,
                    device=lenslet_jones.device,
                )
                pol_torch = torch.from_numpy(pol_hor).type(torch.complex64)
                ana_torch = torch.from_numpy(analyzer).type(torch.complex64)
                E_out = ana_torch @ lenslet_jones @ pol_torch
                intensity = torch.linalg.norm(E_out, axis=1) ** 2
                intensity_image_list[setting][
                    self.ray_valid_indices[0, :], self.ray_valid_indices[1, :]
                ] = intensity

        return intensity_image_list

    def calc_cummulative_JM_lenslet(
        self, volume_in: BirefringentVolume, microlens_offset=[0, 0], mla_index=(0, 0)
    ):
        """Calculate the Jones matrix associated with each pixel behind a lenslet."""
        pixels_per_ml = self.optical_info["pixels_per_ml"]
        lenslet = np.zeros((pixels_per_ml, pixels_per_ml, 2, 2), dtype=np.complex128)
        if self.backend == BackEnds.PYTORCH:
            lenslet = torch.from_numpy(lenslet).to(volume_in.Delta_n.device)
            is_nan = torch.isnan
            lenslet = self.calc_cummulative_JM_of_ray_torch(
                volume_in, microlens_offset, mla_index=mla_index
            )
        else:
            is_nan = np.isnan
            for i in range(pixels_per_ml):
                for j in range(pixels_per_ml):
                    if not is_nan(self.ray_entry[0, i, j]):
                        # Due to the optics, no light reaches the pixel
                        # TODO: verify that the Jones matrix should be zeros instead of identity
                        lenslet[i, j, :, :] = self.calc_cummulative_JM_of_ray_numpy(
                            i, j, volume_in, microlens_offset
                        )
        return lenslet

    def voxRayJM(self, Delta_n, opticAxis, rayDir, ell, wavelength):
        """Compute Jones matrix associated with a particular ray and voxel combination"""
        start_time_voxRayJM = time.perf_counter()
        ret, azim = self.vox_ray_ret_azim(Delta_n, opticAxis, rayDir, ell, wavelength)
        jones = self.vox_ray_matrix(ret, azim)
        end_time_voxRayJM = time.perf_counter()
        self.times["voxRayJM"] += end_time_voxRayJM - start_time_voxRayJM
        return jones

    def vox_ray_ret_azim(self, Delta_n, opticAxis, rayDir, ell, wavelength):
        """Calculate the effective retardance and azimuth of a ray
        passing through a voxel.
        Azimuth is the angle of the slow axis of retardance.
        Note: The numpy and pytorch method differ by a factor of 2 in the
        retardance calculation, because the Jones matrix expressions
        diff by a factor of 2.
        """
        start_time = time.perf_counter()
        if self.backend == BackEnds.NUMPY:
            ret, azim = jones_matrix.vox_ray_ret_azim_numpy(
                Delta_n, opticAxis, rayDir, ell, wavelength
            )
        else:
            if DEBUG:
                if not torch.is_tensor(opticAxis):
                    opticAxis = torch.from_numpy(opticAxis).to(Delta_n.device)
            ret, azim = jones_matrix.calculate_vox_ray_ret_azim_torch(
                Delta_n,
                opticAxis,
                rayDir,
                ell,
                wavelength,
                nonzeros_only=self.only_nonzero_for_jones,
            )

        end_time = time.perf_counter()
        self.times["calc_ret_azim_for_jones"] += end_time - start_time
        return ret, azim

    def vox_ray_matrix(self, ret, azim):
        """Calculate the Jones matrix from a given retardance and
        azimuth angle."""
        start_time = time.perf_counter()
        if DEBUG:
            check_for_inf_or_nan(ret)
            check_for_inf_or_nan(azim)
        if self.backend == BackEnds.NUMPY:
            jones = JonesMatrixGenerators.linear_retarder(ret, azim)
        elif self.backend == BackEnds.PYTORCH:
            jones_in_2_steps = False
            if jones_in_2_steps:
                diag_time_start = time.perf_counter()
                diag, offdiag = jones_matrix._get_diag_offdiag_jones(ret, azim)
                diag_time_end = time.perf_counter()
                jones = jones_matrix.jones_torch_from_diags(diag, offdiag)
                jones_time_end = time.perf_counter()
                self.times["Diag-Offdiag"] += diag_time_end - diag_time_start
                self.times["Stacking"] += jones_time_end - diag_time_end
            else:
                jones = jones_matrix.calculate_jones_torch(
                    ret, azim, nonzeros_only=self.only_nonzero_for_jones
                )
                # self.times["Diag-Offdiag"] = 0
                # self.times["Stacking"] = 0
            if DEBUG:
                assert not torch.isnan(
                    jones
                ).any(), "A Jones matrix contains NaN values."
        end_time = time.perf_counter()
        self.times["calc_jones"] += end_time - start_time
        return jones

    def clone(self):
        # Code to create a copy of this instance
        new_instance = BirefringentVolume(...)
        return new_instance

    @staticmethod
    def rayJM_numpy(JMlist):
        """Computes product of Jones matrix sequence
        Equivalent method: np.linalg.multi_dot([JM1, JM2])
        """
        product = np.identity(2)
        for JM in JMlist:
            product = product @ JM
        return product

    @staticmethod
    def rayJM_torch(JMlist, voxels_of_segs):
        """Computes product of Jones matrix sequence
        Equivalent method: torch.linalg.multi_dot([JM1, JM2])
        """
        n_rays = len(JMlist[0])
        product = (
            torch.tensor(
                [[1.0, 0], [0, 1.0]], dtype=torch.complex64, device=JMlist[0].device
            )
            .unsqueeze(0)
            .repeat(n_rays, 1, 1)
        )
        for ix, JM in enumerate(JMlist):
            rays_with_voxels = [len(vx) > ix for vx in voxels_of_segs]
            product[rays_with_voxels, ...] = product[rays_with_voxels, ...] @ JM
        return product

    def apply_polarizers(self, material_jones):
        """Apply the polarizer and analyzer to a product of Jones matrices representing the
        material. material_jones can be a 2x2 array or probably a list/array of 2x2 array.
        """
        if self.backend == BackEnds.PYTORCH:
            # Possibly need to attach .to(Delta_n.device)
            polarizer = torch.from_numpy(self.optical_info["polarizer"]).type(
                torch.complex64
            )
            analyzer = torch.from_numpy(self.optical_info["analyzer"]).type(
                torch.complex64
            )
        elif self.backend == BackEnds.NUMPY:
            polarizer = self.optical_info["polarizer"]
            analyzer = self.optical_info["analyzer"]
        effective_jones = analyzer @ material_jones @ polarizer
        return effective_jones
