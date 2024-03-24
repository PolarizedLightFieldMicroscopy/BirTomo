"""Birefringence implementations for VolumeRaytraceLFM.
This module contains the BirefringentVolume class
and BirefringentRaytraceLFM class
"""
from math import floor
from tqdm import tqdm
from timeit import default_timer as timer
import time
from collections import Counter
from VolumeRaytraceLFM.abstract_classes import *
from VolumeRaytraceLFM.birefringence_base import BirefringentElement
from VolumeRaytraceLFM.file_manager import VolumeFileManager
from VolumeRaytraceLFM.jones_calculus import JonesMatrixGenerators, JonesVectorGenerators
from VolumeRaytraceLFM.utils.dict_utils import filter_keys_by_count
from utils import errors

NORM_PROJ = False   # normalize the projection of the ray onto the optic axis
OPTIMIZING_MODE = False # use the birefringence stored in Delta_n_combined

######################################################################
class BirefringentVolume(BirefringentElement):
    '''This class stores a 3D array of voxels with birefringence properties,
    either with a numpy or pytorch back-end.'''
    def __init__(self, backend=BackEnds.NUMPY, torch_args={},
        optical_info={}, #,
        Delta_n=0.0,
        optic_axis=[1.0, 0.0, 0.0],
        volume_creation_args=None):
        '''BirefringentVolume
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
            '''
        super(BirefringentVolume, self).__init__(backend=backend,
                                                 torch_args=torch_args,
                                                 optical_info=optical_info
                                                 )
        self._initialize_volume_attributes(optical_info, Delta_n, optic_axis)

        # Check if a volume creation was requested
        if volume_creation_args is not None:
            self.init_volume(volume_creation_args['init_mode'], volume_creation_args.get('init_args', {}))

    def _initialize_volume_attributes(self, optical_info, Delta_n, optic_axis):
        self.volume_shape = optical_info['volume_shape']
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
            assert len(self.Delta_n.shape) == 3, '3D Delta_n expected, as the optic_axis was provided as a 3D array'
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
            assert len(Delta_n.shape) == 3, \
                    '3D Delta_n expected, as the optic_axis was provided as a 3D torch tensor'
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
        self.optic_axis = nn.Parameter(self.optic_axis.reshape(3, -1)).type(torch.get_default_dtype())
        self.Delta_n = nn.Parameter(self.Delta_n.flatten()).type(torch.get_default_dtype())

    def _handle_3d_optic_axis_numpy(self, optic_axis):
        """Normalize and reshape a 3D optic axis array for Numpy backend."""
        self.volume_shape = optic_axis.shape[1:]
        # Flatten all the voxels in order to normalize them
        optic_axis = optic_axis.reshape(
                        3,
                        self.volume_shape[0] * self.volume_shape[1] * self.volume_shape[2]
                        ).astype(np.float64)
        for n_voxel in range(len(optic_axis[0,...])):
            oa_norm = np.linalg.norm(optic_axis[:,n_voxel])
            if oa_norm > 0:
                optic_axis[:,n_voxel] /= oa_norm
        # Set 4D shape again
        self.optic_axis = optic_axis.reshape(3, *self.volume_shape)

    def _handle_single_optic_axis_numpy(self, optic_axis):
        """Set a single optic axis for all voxels for Numpy backend."""
        optic_axis = np.array(optic_axis)
        oa_norm = np.linalg.norm(optic_axis)
        if oa_norm != 0:
            optic_axis /= oa_norm
        self.optic_axis = np.expand_dims(optic_axis,[1,2,3]).repeat(self.volume_shape[0],1).repeat(self.volume_shape[1],2).repeat(self.volume_shape[2],3)
        # self.optic_axis = np.expand_dims(optic_axis, axis=(1, 2, 3))
        # self.optic_axis = np.repeat(self.optic_axis, self.volume_shape, axis=(1, 2, 3))

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
        optic_axis_tensor = torch.from_numpy(optic_axis).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        self.optic_axis = optic_axis_tensor.repeat(1, *self.volume_shape)

    def get_delta_n(self):
        '''Retrieves the birefringence as a 3D array'''
        if self.backend == BackEnds.PYTORCH:
            return self.Delta_n.view(self.optical_info['volume_shape'])
        else:
            return self.Delta_n

    def get_optic_axis(self):
        '''Retrieves the optic axis as a 4D array'''
        if self.backend == BackEnds.PYTORCH:
            return self.optic_axis.view(3, self.optical_info['volume_shape'][0],
                                            self.optical_info['volume_shape'][1],
                                            self.optical_info['volume_shape'][2])
        else:
            return self.optic_axis

    def normalize_optic_axis(self):
        """Normalize the optic axis per voxel."""
        if self.backend == BackEnds.PYTORCH:
            with torch.no_grad():
                self.optic_axis.requires_grad = False
                mags = torch.linalg.norm(self.optic_axis, axis=0)
                valid_mask = mags>0
                self.optic_axis[:, valid_mask].data /= mags[valid_mask]
                self.optic_axis.requires_grad = True
        elif self.backend == BackEnds.NUMPY:
            mags = np.linalg.norm(self.optic_axis, axis=0)
            valid_mask = mags>0
            self.optic_axis[:, valid_mask] /= mags[valid_mask]

    def __iadd__(self, other):
        ''' Overload the += operator to be able to sum volumes'''
        # Check that shapes are the same
        assert (self.get_delta_n().shape == other.get_delta_n().shape) \
                and (self.get_optic_axis().shape == other.get_optic_axis().shape)
        # Check if it's pytorch and need to have the grads disabled before modification
        has_grads = False
        if hasattr(self.Delta_n, 'requires_grad'):
            torch.set_grad_enabled(False)
            has_grads = True
            self.Delta_n.requires_grad = False
            self.optic_axis.requires_grad = False

        self.Delta_n += other.Delta_n
        self.optic_axis += other.optic_axis
        # Maybe normalize axis again?

        if has_grads:
            self.optic_axis = self.optic_axis / nn.Parameter(torch.linalg.norm(self.optic_axis, axis=0))
            torch.set_grad_enabled(has_grads)
            self.Delta_n.requires_grad = True
            self.optic_axis.requires_grad = True
        else:
            self.optic_axis = self.optic_axis / np.linalg.norm(self.optic_axis)
        return self

    def plot_lines_plotly(self, colormap='Bluered_r', size_scaler=5,
                          fig=None, draw_spheres=True, delta_n_ths=0.5,
                          use_ticks=False
                          ):
        '''Plots the optic axis as lines and the birefringence as sphere
        at the ends of the lines. Other parameters could be opacity=0.5 or mode='lines'
        Args:
            delta_n_ths (float): proportion of birefringence values to set to zero
                                    after the birefringence has been normalized
        '''
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
        delta_n[np.abs(delta_n)<delta_n_ths] = 0

        import plotly.graph_objects as go
        volume_shape = optical_info['volume_shape']
        volume_size_um = [optical_info['voxel_size_um'][i] * optical_info['volume_shape'][i] for i in range(3)]
        [dz, dxy, dxy] = optical_info['voxel_size_um']

        # Sometimes the volume_shape is causing an error when being used as the nticks parameter
        if use_ticks:
            scene_dict = dict(
                xaxis = {"nticks": volume_shape[0], "range": [0, volume_size_um[0]]},
                yaxis = {"nticks": volume_shape[1], "range": [0, volume_size_um[1]]},
                zaxis = {"nticks": volume_shape[2], "range": [0, volume_size_um[2]]},
                xaxis_title = 'Axial dimension',
                aspectratio = {"x": volume_size_um[0], "y": volume_size_um[1], "z": volume_size_um[2]},
                aspectmode = 'manual'
                )
        else:
            scene_dict = dict(
                xaxis_title = 'Axial dimension',
                aspectratio = {"x": volume_size_um[0], "y": volume_size_um[1], "z": volume_size_um[2]},
                aspectmode = 'manual'
                )

        # Define grid
        coords = np.indices(np.array(delta_n.shape)).astype(float)

        coords_base = [(coords[i] + 0.5) * optical_info['voxel_size_um'][i] for i in range(3)]
        coords_tip =  [(coords[i] + 0.5 + optic_axis[i,...] * delta_n * 0.75) * optical_info['voxel_size_um'][i] for i in range(3)]

        # Plot single line per voxel, where it's length is delta_n
        z_base, y_base, x_base = coords_base
        z_tip, y_tip, x_tip = coords_tip

        # Don't plot zero values
        mask = delta_n==0
        x_base[mask] = np.NaN
        y_base[mask] = np.NaN
        z_base[mask] = np.NaN
        x_tip[mask] = np.NaN
        y_tip[mask] = np.NaN
        z_tip[mask] = np.NaN

        # Gather all rays in single arrays, to plot them all at once, placing NAN in between them
        array_size = 3 * len(x_base.flatten())
        # Prepare colormap
        all_x = np.empty((array_size))
        all_x[::3] = x_base.flatten()
        all_x[1::3] = x_tip.flatten()
        all_x[2::3] = np.NaN
        all_y = np.empty((array_size))
        all_y[::3] = y_base.flatten()
        all_y[1::3] = y_tip.flatten()
        all_y[2::3] = np.NaN
        all_z = np.empty((array_size))
        all_z[::3] = z_base.flatten()
        all_z[1::3] = z_tip.flatten()
        all_z[2::3] = np.NaN
        # Compute colors
        all_color = np.empty((array_size))
        all_color[::3] =    (x_base-x_tip).flatten() ** 2 + \
                            (y_base-y_tip).flatten() ** 2 + \
                            (z_base-z_tip).flatten() ** 2
        # all_color[::3] =  delta_n.flatten() * 1.0
        all_color[1::3] = all_color[::3]
        all_color[2::3] = 0
        all_color[np.isnan(all_color)] = 0

        err = ("The BirefringentVolume is expected to have non-zeros values. If the " +
            "BirefringentVolume was cropped to fit into a region, the non-zero values " +
            "may no longer be included.")
        assert any(all_color != 0), err

        all_color[all_color!=0] -= all_color[all_color!=0].min()
        all_color += 0.5
        all_color /= all_color.max()

        if fig is None:
            fig = go.Figure()
        fig.add_scatter3d(z=all_x, y=all_y, x=all_z,
            marker={"color": all_color, "colorscale": colormap, "size": 4},
            line={"color": all_color, "colorscale": colormap, "width": size_scaler},
            connectgaps=False,
            mode='lines'
            )
        if draw_spheres:
            fig.add_scatter3d(z=x_base.flatten(), y=y_base.flatten(), x=z_base.flatten(),
                marker={"color": all_color[::3] - 0.5,
                        "colorscale": colormap,
                        "size": size_scaler * 5 * all_color[::3]},
                line={"color": all_color[::3] - 0.5, "colorscale": colormap, "width": 5},
                mode='markers')
        camera = {'eye': {'x': 50, 'y': 0, 'z': 0}}
        fig.update_layout(
            scene=scene_dict,
            scene_camera=camera,
            margin={'r': 0, 'l': 0, 'b': 0, 't': 0},
            showlegend=False,
            )
        # fig.data = fig.data[::-1]
        # fig.show()
        return fig

    @staticmethod
    def plot_volume_plotly(optical_info, voxels_in=None, opacity=0.5,
                           colormap='gray', fig=None):
        '''Plots a 3D array with the non-zero voxels shaded.'''
        voxels = voxels_in * 1.0
        # Check if this is a torch tensor
        if not isinstance(voxels_in, np.ndarray):
            try:
                voxels = voxels.detach()
                voxels = voxels.cpu().abs().numpy()
            except:
                pass
        voxels = np.abs(voxels)
        err = ("The set of voxels are expected to have non-zeros values. If the " +
            "BirefringentVolume was cropped to fit into a region, the non-zero values " +
            "may no longer be included.")
        assert voxels.any(), err

        import plotly.graph_objects as go
        volume_shape = optical_info['volume_shape']
        volume_size_um = [optical_info['voxel_size_um'][i] * optical_info['volume_shape'][i] for i in range(3)]
        # Define grid
        coords = np.indices(np.array(voxels.shape)).astype(float)
        # Shift by half a voxel and multiply by voxel size
        coords = [(coords[i]+0.5) * optical_info['voxel_size_um'][i] for i in range(3)]
        if fig is None:
            fig = go.Figure()
        fig.add_volume(
            x=coords[0].flatten(),
            y=coords[1].flatten(),
            z=coords[2].flatten(),
            value=voxels.flatten() / voxels.max(),
            isomin=0,
            isomax=0.1,
            opacity=opacity, # needs to be small to see through all surfaces
            surface_count=20, # needs to be a large number for good volume rendering
            colorscale=colormap
            )
        camera = {'eye': {'x': 50, 'y': 0, 'z': 0}}
        fig.update_layout(
            scene=dict(
                xaxis = {"nticks": volume_shape[0], "range": [0, volume_size_um[0]]},
                yaxis = {"nticks": volume_shape[1], "range": [0, volume_size_um[1]]},
                zaxis = {"nticks": volume_shape[2], "range": [0, volume_size_um[2]]},
                xaxis_title='Axial dimension',
                aspectratio = {"x": volume_size_um[0], "y": volume_size_um[1], "z": volume_size_um[2]},
                aspectmode = 'manual'
                ),
            scene_camera=camera,
            margin={'r': 0, 'l': 0, 'b': 0, 't': 0},
            autosize=True
            )
        # fig.data = fig.data[::-1]
        # fig.show()
        return fig

    def get_vox_params(self, vox_idx):
        '''vox_idx is a tuple'''
        if isinstance(vox_idx, tuple) and len(vox_idx) == 3:
            axis = self.optic_axis[:, vox_idx[0], vox_idx[1], vox_idx[2]]
        else:
            axis = self.optic_axis[:, vox_idx]
        return self.Delta_n[vox_idx], axis

    @staticmethod
    def crop_to_region_shape(delta_n, optic_axis, volume_shape, region_shape):
        '''
        Parameters:
            delta_n (np.array): 3D array with dimension volume_shape
            optic_axis (np.array): 4D array with dimension (3, *volume_shape)
            volume_shape (np.array): dimensions of object volume
            region_shape (np.array): dimensions of the region fitting the object,
                                        values must be greater than volume_shape
        Returns:
            cropped_delta_n (np.array): 3D array with dimension region_shape
            cropped_optic_axis (np.array): 4D array with dimension (3, *region_shape)
        '''
        assert (volume_shape >= region_shape).all(), "Error: volume_shape must be greater than region_shape"
        crop_start = (volume_shape - region_shape) // 2
        crop_end = crop_start + region_shape
        cropped_delta_n = delta_n[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], crop_start[2]:crop_end[2]]
        cropped_optic_axis = optic_axis[:, crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], crop_start[2]:crop_end[2]]
        return cropped_delta_n, cropped_optic_axis

    @staticmethod
    def pad_to_region_shape(delta_n, optic_axis, volume_shape, region_shape):
        '''
        Parameters:
            delta_n (np.array): 3D array with dimension volume_shape
            optic_axis (np.array): 4D array with dimension (3, *volume_shape)
            volume_shape (np.array): dimensions of object volume
            region_shape (np.array): dimensions of the region fitting the object,
                                        values must be less than volume_shape
        Returns:
            padded_delta_n (np.array): 3D array with dimension region_shape
            padded_optic_axis (np.array): 4D array with dimension (3, *region_shape)
        '''
        assert (volume_shape <= region_shape).all(), "Error: volume_shape must be less than region_shape"
        z_,y_, x_ = region_shape
        z, y, x = volume_shape
        z_pad = abs(z_-z)
        y_pad = abs(y_-y)
        x_pad = abs(x_-x)
        padded_delta_n = np.pad(delta_n,(
                        (z_pad//2, z_pad//2 + z_pad%2),
                        (y_pad//2, y_pad//2 + y_pad%2),
                        (x_pad//2, x_pad//2 + x_pad%2)),
                    mode = 'constant').astype(np.float64)
        padded_optic_axis = np.pad(optic_axis,((0,0),
                    (z_pad//2, z_pad//2 + z_pad%2),
                    (y_pad//2, y_pad//2 + y_pad%2),
                    (x_pad//2, x_pad//2 + x_pad%2)),
                mode = 'constant', constant_values=np.sqrt(3)).astype(np.float64)
        return padded_delta_n, padded_optic_axis

    @staticmethod
    def init_from_file(h5_file_path, backend=BackEnds.NUMPY, optical_info=None):
        ''' Loads a birefringent volume from an h5 file and places it in the center of the volume
            It requires to have:
                optical_info/volume_shape [3]: shape of the volume in voxels [nz,ny,nx]
                data/delta_n [nz,ny,nx]: Birefringence volumetric information.
                data/optic_axis [3,nz,ny,nx]: Optical axis per voxel.
        '''
        file_manager = VolumeFileManager()
        delta_n, optic_axis = file_manager.extract_data_from_h5(h5_file_path)
        region_shape = np.array(optical_info['volume_shape'])
        if (delta_n.shape == region_shape).all():
            pass
        elif (delta_n.shape >= region_shape).all():
            delta_n, optic_axis = BirefringentVolume.crop_to_region_shape(delta_n, optic_axis, delta_n.shape, region_shape)
        elif (delta_n.shape <= region_shape).all():
            delta_n, optic_axis = BirefringentVolume.pad_to_region_shape(delta_n, optic_axis, delta_n.shape, region_shape)
        else:
            err = (f"BirefringentVolume has dimensions ({delta_n.shape}) that are not all greater " +
                    f"than or less than the volume region dimensions ({region_shape}) set for the microscope")
            raise ValueError(err)
        volume = BirefringentVolume(backend=backend, optical_info=optical_info, Delta_n=delta_n, optic_axis=optic_axis)
        return volume

    @staticmethod
    def load_from_file(h5_file_path, backend_type='numpy'):
        ''' Loads a birefringent volume from an h5 file and places it in the center of the volume
            It requires to have:
                data/delta_n [nz,ny,nx]: Birefringence volumetric information.
                data/optic_axis [3,nz,ny,nx]: Optical axis per voxel.'''
        if backend_type == 'torch':
            backend = BackEnds.PYTORCH
        elif backend_type == 'numpy':
            backend = BackEnds.NUMPY
        else:
            raise ValueError(f"Backend type {backend_type} is not an option.")

        file_manager = VolumeFileManager()
        delta_n, optic_axis, volume_shape, voxel_size_um = file_manager.extract_all_data_from_h5(h5_file_path)
        cube_voxels = True
        # Create optical info dictionary
        # TODO: add the remaining variables, notably the voxel size and the cube voxels boolean
        optical_info = dict({'volume_shape': volume_shape,
                             'voxel_size_um': voxel_size_um,
                             'cube_voxels': cube_voxels}
                             )
        # Create volume
        volume_out = BirefringentVolume(backend=backend, optical_info=optical_info, Delta_n=delta_n, optic_axis=optic_axis)
        return volume_out

    def save_as_file(self, h5_file_path, description="Temporary description", optical_all=False):
        '''Store this volume into an h5 file'''
        print(f'Saving volume to h5 file: {h5_file_path}')

        delta_n, optic_axis = self._get_data_as_numpy_arrays()
        file_manager = VolumeFileManager()
        file_manager.save_as_h5(h5_file_path, delta_n, optic_axis, self.optical_info, description, optical_all)

    def save_as_numpy_arrays(self, filename):
        '''Store this volume into a npy file'''
        delta_n, optic_axis = self._get_data_as_numpy_arrays()
        file_manager = VolumeFileManager()
        file_manager.save_as_npz(filename, delta_n, optic_axis)

    def _get_data_as_numpy_arrays(self):
        '''Converts delta_n and optic_axis based on backend'''
        delta_n = self.get_delta_n()
        optic_axis = self.get_optic_axis()

        if self.backend == BackEnds.PYTORCH:
            delta_n = delta_n.detach().cpu().numpy()
            optic_axis = optic_axis.detach().cpu().numpy()

        return delta_n, optic_axis

    def save_as_numpy_arrays(self, filename):
        '''Store this volume into a numpy file'''
        delta_n, optic_axis = self._get_data_as_numpy_arrays()
        np.savez(filename, birefringence=delta_n, optic_axis=optic_axis)

    def save_as_tiff(self, filename):
        '''Store this volume into a tiff file'''
        delta_n, optic_axis = self._get_data_as_numpy_arrays()
        file_manager = VolumeFileManager()
        file_manager.save_as_channel_stack_tiff(filename, delta_n, optic_axis)

    def _get_backend_str(self):
        if self.backend == BackEnds.PYTORCH:
            return 'pytorch'
        elif self.backend == BackEnds.NUMPY:
            return 'numpy'
        else:
            raise ValueError(f"Backend type {self.backend} is not supported.")

########### Generate different birefringent volumes ############
    def init_volume(self, init_mode='zeros', init_args={}):
        ''' This function creates predefined volumes and shapes, such as
        planes, ellipsoids, random, etc
        TODO: use init_args for random and planes
        '''
        volume_shape = self.optical_info['volume_shape']
        if init_mode=='zeros':
            if self.backend == BackEnds.NUMPY:
                voxel_parameters = np.zeros([4,] + volume_shape)
            if self.backend == BackEnds.PYTORCH:
                voxel_parameters = torch.zeros([4,] + volume_shape)
        elif init_mode=='single_voxel':
            delta_n = init_args['delta_n'] if 'delta_n' in init_args.keys() else 0.01
            optic_axis = init_args['optic_axis'] if 'optic_axis' in init_args.keys() else [1,0,0]
            offset = init_args['offset'] if 'offset' in init_args.keys() else [0,0,0]
            voxel_parameters = self.generate_single_voxel_volume(volume_shape, delta_n, optic_axis, offset)
        elif init_mode=='random':
            if init_args == {}:
                my_init_args = {'Delta_n_range' : [0,1], 'axes_range' : [-1,1]}
            else:
                my_init_args = init_args
            voxel_parameters = self.generate_random_volume(volume_shape, init_args=my_init_args)
        elif 'planes' in init_mode:
            n_planes = int(init_mode[0])
            z_offset = init_args['z_offset'] if 'z_offset' in init_args.keys() else 0
            delta_n = init_args['delta_n'] if 'delta_n' in init_args.keys() else 0.01
            # Perpendicular optic axes each with constant birefringence and orientation
            voxel_parameters = self.generate_planes_volume(
                                        volume_shape, n_planes,
                                        z_offset=z_offset, delta_n=delta_n
                                        )
        elif init_mode=='ellipsoid':
            # Look for variables in init_args, else init with something
            radius = init_args['radius'] if 'radius' in init_args.keys() else [5.5,5.5,3.5]
            center = init_args['center'] if 'center' in init_args.keys() else [0.5,0.5,0.5]
            delta_n = init_args['delta_n'] if 'delta_n' in init_args.keys() else 0.01
            alpha = init_args['border_thickness'] if 'border_thickness' in init_args.keys() else 1
            voxel_parameters = self.generate_ellipsoid_volume(
                                        volume_shape, center=center,
                                        radius=radius, alpha=alpha, delta_n=delta_n
                                        )
        else:
            print(f"The init mode {init_mode} has not been created yet.")
        volume_ref = BirefringentVolume(backend=self.backend,
                                        optical_info=self.optical_info,
                                        Delta_n=voxel_parameters[0,...],
                                        optic_axis=voxel_parameters[1:,...])
        self.Delta_n = volume_ref.Delta_n
        self.optic_axis = volume_ref.optic_axis

    @staticmethod
    def generate_single_voxel_volume(volume_shape, delta_n=0.01,
                                     optic_axis=[1,0,0], offset=[0,0,0]):
        # Identity the center of the volume after the shifts
        vox_idx = [
            volume_shape[0] // 2 + offset[0],
            volume_shape[1] // 2 + offset[1],
            volume_shape[2] // 2 + offset[2]
        ]
        # Create a volume of all zeros.
        vol = np.zeros([4,] + volume_shape)
        # Set the birefringence and optic axis
        vol[0, vox_idx[0], vox_idx[1], vox_idx[2]] = delta_n
        vol[1:, vox_idx[0], vox_idx[1], vox_idx[2]] = np.array(optic_axis)
        return vol

    @staticmethod
    def generate_random_volume(volume_shape,
                               init_args={'Delta_n_range' : [0,1], 'axes_range' : [-1,1]}):
        Delta_n = np.random.uniform(init_args['Delta_n_range'][0],
                                    init_args['Delta_n_range'][1],
                                    volume_shape)
        # Random axis
        min_axis = init_args['axes_range'][0]
        max_axis = init_args['axes_range'][1]
        a_0 = np.random.uniform(min_axis, max_axis, volume_shape)
        a_1 = np.random.uniform(min_axis, max_axis, volume_shape)
        a_2 = np.random.uniform(min_axis, max_axis, volume_shape)
        norm_A = np.sqrt(a_0**2 + a_1**2 + a_2**2)
        return np.concatenate(
                        (np.expand_dims(Delta_n, axis=0), np.expand_dims(a_0/norm_A, axis=0),
                        np.expand_dims(a_1/norm_A, axis=0), np.expand_dims(a_2/norm_A, axis=0)),
                        0)

    @staticmethod
    def generate_planes_volume(volume_shape, n_planes=1, z_offset=0,
                               delta_n=0.01):
        vol = np.zeros([4,] + volume_shape)
        z_size = volume_shape[0]
        z_ranges = np.linspace(0, z_size-1, n_planes*2).astype(int)

        # Set random optic axis
        optic_axis = np.random.uniform(-1, 1, [3, *volume_shape])
        norms = np.linalg.norm(optic_axis, axis=0)
        vol[1:, ...] = optic_axis / norms

        if n_planes==1:
            # Birefringence
            vol[0, z_size//2+z_offset, :, :] = delta_n
            # Axis
            # vol[1, z_size//2, :, :] = 0.5
            vol[1, z_size//2+z_offset, :, :] = 1
            vol[2, z_size//2+z_offset, :, :] = 0
            vol[3, z_size//2+z_offset, :, :] = 0
            return vol
        random_data = BirefringentVolume.generate_random_volume([n_planes])
        for z_ix in range(0,n_planes):
            vol[:,z_ranges[z_ix*2] : z_ranges[z_ix*2+1]] = np.expand_dims(random_data[:,z_ix],[1,2,3]).repeat(1,1).repeat(volume_shape[1],2).repeat(volume_shape[2],3)
        return vol

    @staticmethod
    def generate_ellipsoid_volume(volume_shape, center=[0.5,0.5,0.5],
                                  radius=[10,10,10], alpha=1, delta_n=0.01):
        '''Creates an ellipsoid with optical axis normal to the ellipsoid surface.
        Args:
            center [3]: [cz,cy,cx] from 0 to 1 where 0.5 is the center of the volume_shape.
            radius [3]: in voxels, the radius in z,y,x for this ellipsoid.
            alpha (float): Border thickness.
            delta_n (float): Delta_n value of birefringence in the volume
        Returns:
            vol (np.array): 4D array where the first dimension represents the
                birefringence and optic axis properties, and the last three
                dims represents the 3D spatial locations.
            '''
        # Originally grabbed from https://math.stackexchange.com/questions/2931909/normal-of-a-point-on-the-surface-of-an-ellipsoid,
        #   then modified to do the subtraction of two ellipsoids instead.
        vol = np.zeros([4,] + volume_shape)
        kk, jj, ii = np.meshgrid(np.arange(volume_shape[0]),
                                 np.arange(volume_shape[1]),
                                 np.arange(volume_shape[2]),
                                 indexing='ij'
                                 )
        # shift to center
        kk = floor(center[0]*volume_shape[0]) - kk.astype(float)
        jj = floor(center[1]*volume_shape[1]) - jj.astype(float)
        ii = floor(center[2]*volume_shape[2]) - ii.astype(float)

        # DEBUG: checking the indices
        # np.argwhere(ellipsoid_border == np.min(ellipsoid_border))
        # plt.imshow(ellipsoid_border_mask[int(volume_shape[0] / 2),:,:])
        ellipsoid_border = (kk**2) / (radius[0]**2) + (jj**2) / (radius[1]**2) + (ii**2) / (radius[2]**2)
        hollow_inner = True
        if hollow_inner:
            ellipsoid_border_mask = np.abs(ellipsoid_border) <= 1
            # The inner radius could also be defined as a scaled version of the outer radius.
            # inner_radius = [0.9 * r for r in radius]
            inner_radius = [r - alpha for r in radius]
            inner_ellipsoid_border = (kk**2) / (inner_radius[0]**2) + (jj**2) / (inner_radius[1]**2) + (ii**2) / (inner_radius[2]**2)
            inner_mask = np.abs(inner_ellipsoid_border) <= 1
        else:
            ellipsoid_border_mask = np.abs(ellipsoid_border-alpha) <= 1

        vol[0,...] = ellipsoid_border_mask.astype(float)
        # Compute normals
        kk_normal = 2 * kk / radius[0]
        jj_normal = 2 * jj / radius[1]
        ii_normal = 2 * ii / radius[2]
        norm_factor = np.sqrt(kk_normal**2 + jj_normal**2 + ii_normal**2)
        # Avoid division by zero
        norm_factor[norm_factor==0] = 1
        vol[1,...] = (kk_normal / norm_factor) * vol[0,...]
        vol[2,...] = (jj_normal / norm_factor) * vol[0,...]
        vol[3,...] = (ii_normal / norm_factor) * vol[0,...]
        vol[0,...] *= delta_n
        # vol = vol.permute(0,2,1,3)
        if hollow_inner:
            # Hollowing out the ellipsoid
            combined_mask = np.logical_and(ellipsoid_border_mask, ~inner_mask)
            vol[0,...] = vol[0,...] * combined_mask.astype(float)
        return vol

    @staticmethod
    def create_dummy_volume(backend=BackEnds.NUMPY, optical_info=None,
                            vol_type="shell", volume_axial_offset=0):
        '''Create different volumes, some of them randomized. Feel free to add
        your volumes here.
        Parameters:
            backend: BackEnds.NUMPY or BackEnds.PYTORCH
            optical_info (dict): Stores optical properties, primarily the volume shape.
            vol_type (str): Type of volume to generate. Options include "single_voxel", "zeros",
                            "ellipsoid", and "shell".
            volume_axial_offset (int): A potential offset for the volume on the axial direction.
        Returns:
            volume (BirefringentVolume)
        '''
        # Where is the center of the volume?
        vox_ctr_idx = np.array([optical_info['volume_shape'][0] / 2,
                                optical_info['volume_shape'][1] / 2,
                                optical_info['volume_shape'][2] / 2]).astype(int)
        if vol_type in ["single_voxel", "zeros"]:
            if backend == BackEnds.NUMPY:
                raise NotImplementedError("There is not a NUMPY single_voxel or" +
                                          "zeros volume method implemented. Use PYTORCH instead.")
            voxel_delta_n = 0.01
            if vol_type == 'zeros':
                voxel_delta_n = 0
            # TODO: make numpy version of birefringence axis
            voxel_birefringence_axis = torch.tensor([1,0.0,0])
            voxel_birefringence_axis /= voxel_birefringence_axis.norm()
            # Create empty volume
            volume = BirefringentVolume(backend=backend,
                                        optical_info=optical_info,
                                        volume_creation_args={'init_mode' : 'zeros'})
            # Set delta_n
            volume.Delta_n.requires_grad = False
            volume.optic_axis.requires_grad = False
            volume.get_delta_n()[volume_axial_offset,
                                            vox_ctr_idx[1],
                                            vox_ctr_idx[2]] = voxel_delta_n
            # set optical_axis
            volume.get_optic_axis()[:, volume_axial_offset,
                                    vox_ctr_idx[1],
                                    vox_ctr_idx[2]] \
                                    = voxel_birefringence_axis
            volume.Delta_n.requires_grad = True
            volume.optic_axis.requires_grad = True
        elif vol_type in ["ellipsoid", "shell"]:    # whole plane
            ellipsoid_args = {  'radius' : [5.5, 9.5, 5.5],
                        'center' : [volume_axial_offset / optical_info['volume_shape'][0], \
                                        0.50, 0.5],  # from 0 to 1
                        'delta_n' : 0.01,
                        'border_thickness' : 1}
            volume = BirefringentVolume(
                backend=backend,
                optical_info=optical_info,
                volume_creation_args={'init_mode' : 'ellipsoid', 'init_args' : ellipsoid_args}
                )
            # Do we want a shell? Let's remove some of the volume
            if vol_type == 'shell':
                if backend == BackEnds.PYTORCH:
                    with torch.no_grad():
                        volume.get_delta_n()[:optical_info['volume_shape'][0] // 2 + 2,...] = 0
                else:
                    volume.get_delta_n()[:optical_info['volume_shape'][0] // 2 + 2,...] = 0
        elif vol_type == 'sphere_oct13':
            sphere_args = {
                'radius' : [4.5, 4.5, 4.5],
                'center' : [volume_axial_offset / optical_info['volume_shape'][0], \
                                0.50, 0.5],  # from 0 to 1
                'delta_n' : 0.01,
                'border_thickness' : 1
                }
            volume = BirefringentVolume(
                backend=backend,
                optical_info=optical_info,
                volume_creation_args={'init_mode' : 'ellipsoid', 'init_args' : sphere_args}
                )
        elif vol_type[-10:] == 'ellipsoids':
            n_ellipsoids = int(vol_type[:-10])
            volume = BirefringentVolume(backend=backend,
                                        optical_info=optical_info,
                                        volume_creation_args={'init_mode' : 'zeros'})
            for _ in range(n_ellipsoids):
                ellipsoid_args = {
                    'radius' : np.random.uniform(.5, 3.5, [3]),
                    'center' : [np.random.uniform(0.35, 0.65),] + list(np.random.uniform(0.3, 0.70, [2])),
                    'delta_n' : np.random.uniform(-0.01, -0.001),
                    'border_thickness' : 1
                    }
                new_vol = BirefringentVolume(
                    backend=backend,
                    optical_info=optical_info,
                    volume_creation_args={'init_mode' : 'ellipsoid', 'init_args' : ellipsoid_args}
                    )
                volume += new_vol
        elif vol_type == 'ellipsoids_random':
            n_ellipsoids = np.random.randint(1, 5)
            volume = BirefringentVolume(backend=backend,
                                        optical_info=optical_info,
                                        volume_creation_args={'init_mode' : 'zeros'})
            for _ in range(n_ellipsoids):
                ellipsoid_args = {
                    'radius' : np.random.uniform(.5, 3.5, [3]) * 10,
                    'center' : [np.random.uniform(0.35, 0.65),] + list(np.random.uniform(0.3, 0.70, [2])),
                    'delta_n' : np.random.uniform(-0.01, -0.001),
                    'border_thickness' : 1 * 3
                    }
                new_vol = BirefringentVolume(
                    backend=backend,
                    optical_info=optical_info,
                    volume_creation_args={'init_mode' : 'ellipsoid', 'init_args' : ellipsoid_args}
                    )
                volume += new_vol
        elif vol_type == 'sphere':
            sphere_args = {
                'radius' : [np.random.uniform(3, 6)] * 3,
                'center' : [np.random.uniform(0.35, 0.65),] + list(np.random.uniform(0.3, 0.70, [2])),
                'delta_n' : -0.01,
                'border_thickness' : 1
                }
            volume = BirefringentVolume(
                backend=backend,
                optical_info=optical_info,
                volume_creation_args={'init_mode' : 'ellipsoid', 'init_args' : sphere_args}
                )
        elif vol_type == 'small_sphere':
            sphere_args = {
                'radius' : [3] * 3,
                'center' : [0.5] * 3,
                'delta_n' : -0.01,
                'border_thickness' : 1
                }
            volume = BirefringentVolume(
                backend=backend,
                optical_info=optical_info,
                volume_creation_args={'init_mode' : 'ellipsoid', 'init_args' : sphere_args}
                )
        elif vol_type == 'small_sphere_pos':
            min_x = 0.5 - 0.125
            max_x = 0.5 + 0.124
            sphere_args = {
                'radius' : [np.random.uniform(1, 2)] * 3,
                'center' : [np.random.uniform(min_x, max_x),] + list(np.random.uniform(0.42, 0.55, [2])),
                'delta_n' : 0.01,
                'border_thickness' : 1
                }
            volume = BirefringentVolume(
                backend=backend,
                optical_info=optical_info,
                volume_creation_args={'init_mode' : 'ellipsoid', 'init_args' : sphere_args}
                ) 
        elif vol_type == 'small_sphere_rand_bir':
            min_x = 0.5 - 0.125
            max_x = 0.5 + 0.124
            sphere_args = {
                'radius' : [np.random.uniform(1, 2)] * 3,
                'center' : [np.random.uniform(min_x, max_x),] + list(np.random.uniform(0.42, 0.55, [2])),
                'delta_n' : np.random.uniform(0.005, 0.015),
                'border_thickness' : 1
                }
            volume = BirefringentVolume(
                backend=backend,
                optical_info=optical_info,
                volume_creation_args={'init_mode' : 'ellipsoid', 'init_args' : sphere_args}
                )   
        # elif 'my_volume:' # Feel free to add new volumes here
        else:
            raise NotImplementedError
        return volume


############ Implementations
class BirefringentRaytraceLFM(RayTraceLFM, BirefringentElement):
    '''This class extends RayTraceLFM, and implements the forward function,
    where voxels contribute to ray's Jones-matrices with a retardance and axis
    in a non-commutative matter'''
    def __init__(self, backend : BackEnds = BackEnds.NUMPY,
                 torch_args={}, optical_info={}):
        '''Initialize the class with attibriutes, including those from RayTraceLFM.'''
        super(BirefringentRaytraceLFM, self).__init__(
            backend=backend, torch_args=torch_args, optical_info=optical_info
        )

        self.vox_indices_ml_shifted = {}
        self.vox_indices_ml_shifted_all = []
        self.ray_valid_indices_all = None
        self.MLA_volume_geometry_ready = False
        self.verbose = True
        self.mla_execution_times = {}
        self.vox_indices_ml_shifted = {}
        self.vox_indices_by_mla_idx = {}

    def __str__(self):
        info = (
            f"BirefringentRaytraceLFM(backend={self.backend}, "
            # f"torch_args={self.torch_args},
            f"optical_info={self.optical_info})"
            f"\nvox_indices_ml_shifted={self.vox_indices_ml_shifted}"
            f"\nvox_indices_ml_shifted_all={self.vox_indices_ml_shifted_all}"
            f"\nMLA_volume_geometry_ready={self.MLA_volume_geometry_ready}"
            f"\nvox_ctr_idx={self.vox_ctr_idx}"
            f"\nvoxel_span_per_ml={self.voxel_span_per_ml}"
            f"\nray_valid_indices={self.ray_valid_indices}"
            f"\nray_valid_indices_all={self.ray_valid_indices_all}"
            f"\nray_direction_basis[0][0]={self.ray_direction_basis[0][0]}"
            f"\nray_vol_colli_indices[0]={self.ray_vol_colli_indices[0]}"
            f"\nray_vol_colli_lengths[0]={self.ray_vol_colli_lengths[0]}"
            f"\nnonzero_pixels_dict[(0, 0)].shape={self.nonzero_pixels_dict[(0, 0)].shape}"
            f"\nuse_lenslet_based_filtering={self.use_lenslet_based_filtering}"
        )
        return info

    def get_volume_reachable_region(self):
        ''' Returns a binary mask where the MLA's can reach into the volume'''

        n_micro_lenses = self.optical_info['n_micro_lenses']
        n_voxels_per_ml = self.optical_info['n_voxels_per_ml']
        n_ml_half = floor(n_micro_lenses * n_voxels_per_ml / 2.0)
        mask = torch.zeros(self.optical_info['volume_shape'])
        include_ray_angle_reach = True
        if include_ray_angle_reach:
            vox_span_half = int(self.voxel_span_per_ml + (n_micro_lenses * n_voxels_per_ml) / 2)
            mask[:,
                self.vox_ctr_idx[1]-vox_span_half+1 : self.vox_ctr_idx[1]+vox_span_half,
                self.vox_ctr_idx[2]-vox_span_half+1 : self.vox_ctr_idx[2]+vox_span_half] = 1.0
        else:
            mask[:,
                self.vox_ctr_idx[1]-n_ml_half+1 : self.vox_ctr_idx[1]+n_ml_half,
                self.vox_ctr_idx[2]-n_ml_half+1 : self.vox_ctr_idx[2]+n_ml_half] = 1.0
        # mask_volume = BirefringentVolume(backend=self.backend,
        #                   optical_info=self.optical_info, Delta_n=0.01, optic_axis=[0.5,0.5,0])
        # [r,a] = self.ray_trace_through_volume(mask_volume)
        # # Check gradients to see what is affected
        # L = r.mean() + a.mean()
        # L.backward()
        # with torch.no_grad():
        #     mask = mask_volume.Delta_n
        #     mask[mask.grad==0] = 0
        return mask.detach()

    def precompute_MLA_volume_geometry(self):
        """ Expand the ray-voxel interactions from a single microlens to an nxn MLA"""
        if self.MLA_volume_geometry_ready:
            return
        # volume_shape defines the size of the workspace
        # the number of micro lenses defines the valid volume inside the workspace
        volume_shape = self.optical_info['volume_shape']
        n_micro_lenses = self.optical_info['n_micro_lenses']
        n_voxels_per_ml = self.optical_info['n_voxels_per_ml']
        n_pixels_per_ml = self.optical_info['pixels_per_ml']
        n_ml_half = floor(n_micro_lenses / 2.0)
        n_voxels_per_ml_half = floor(self.optical_info['n_voxels_per_ml'] * n_micro_lenses / 2.0)

        # Check if the volume_size can fit these micro_lenses.
        # # considering that some rays go beyond the volume in front of the microlens
        # border_size_around_mla = np.ceil((volume_shape[1]-(n_micro_lenses*n_voxels_per_ml)) / 2)
        min_needed_volume_size = int(self.voxel_span_per_ml + (n_micro_lenses*n_voxels_per_ml))
        assert min_needed_volume_size <= volume_shape[1] and min_needed_volume_size <= volume_shape[2], "The volume in front of the microlenses" + \
             f"({n_micro_lenses},{n_micro_lenses}) is too large for a volume_shape: {self.optical_info['volume_shape'][1:]}. " + \
                f"Increase the volume_shape to at least [{min_needed_volume_size+1},{min_needed_volume_size+1}]"        

        odd_mla_shift = np.mod(n_micro_lenses, 2)
        # Iterate microlenses in y direction
        for iix,ml_ii in tqdm(enumerate(range(-n_ml_half, n_ml_half+odd_mla_shift)),
                              f'Computing rows of microlens ret+azim {self.backend}'):
            # Iterate microlenses in x direction
            for jjx,ml_jj in enumerate(range(-n_ml_half, n_ml_half+odd_mla_shift)):
                # Compute offset to top corner of the volume in front of the microlens (ii,jj)
                current_offset = (
                    np.array([n_voxels_per_ml * ml_ii, n_voxels_per_ml * ml_jj])
                    + np.array(self.vox_ctr_idx[1:]) - n_voxels_per_ml_half
                    )
                self.vox_indices_ml_shifted_all += [
                    [
                        RayTraceLFM.ravel_index(
                        (vox[ix][0], vox[ix][1]+current_offset[0], vox[ix][2]+current_offset[1]),
                        self.optical_info['volume_shape']) for ix in range(len(vox))
                    ]
                    for vox in self.ray_vol_colli_indices
                    ]
                # Shift ray-pixel indices
                if self.ray_valid_indices_all is None:
                    self.ray_valid_indices_all = self.ray_valid_indices.clone()
                else:
                    self.ray_valid_indices_all = torch.cat(
                        (self.ray_valid_indices_all,
                        self.ray_valid_indices + torch.tensor([jjx * n_pixels_per_ml,
                                                               iix * n_pixels_per_ml]).unsqueeze(1)),
                        1)
        # Replicate ray info for all the microlenses
        self.ray_vol_colli_lengths = nn.Parameter(
            self.ray_vol_colli_lengths.repeat(n_micro_lenses ** 2, 1)
            )
        self.ray_direction_basis = nn.Parameter(
            self.ray_direction_basis.repeat(1, n_micro_lenses ** 2, 1)
            )

        self.MLA_volume_geometry_ready = True
        return

    def store_shifted_vox_indices(self):
        """Store the shifted voxel indices for each microlens in a
        dictionary. The shape of the volume is taken from the
        optical_info, which should be the same as the volume used in the
        ray tracing process.

        Args:
            None

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
        n_micro_lenses = self.optical_info['n_micro_lenses']
        n_voxels_per_ml = self.optical_info['n_voxels_per_ml']
        n_ml_half = floor(n_micro_lenses / 2.0)
        collision_indices = self.ray_vol_colli_indices
        if self.verbose:
            print(f"Storing shifted voxel indices for each microlens")
            row_iterable = tqdm(
                range(n_micro_lenses),
                desc=f'Computing rows of microlenses'
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
        
        return self.vox_indices_by_mla_idx

    def ray_trace_through_volume(self, volume_in : BirefringentVolume = None,
                                 all_rays_at_once=True, intensity=False):
        """ This function forward projects a whole volume, by iterating through
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
        # volume_shape defines the size of the workspace
        # the number of micro lenses defines the valid volume inside the workspace
        volume_shape = volume_in.optical_info['volume_shape']
        n_micro_lenses = self.optical_info['n_micro_lenses']
        n_voxels_per_ml = self.optical_info['n_voxels_per_ml']
        n_ml_half = floor(n_micro_lenses / 2.0)

        # Check if the volume_size can fit these microlenses.
        # # considering that some rays go beyond the volume in front of the microlenses
        # border_size_around_mla = np.ceil((volume_shape[1]-(n_micro_lenses*n_voxels_per_ml)) / 2)
        min_required_volume_size = self._calculate_min_volume_size(n_micro_lenses, n_voxels_per_ml)
        self._validate_volume_size(min_required_volume_size, volume_shape)
        # The following assert statement is redundant, but it is kept for clarity
        err_message = "The volume in front of the microlenses" + \
            f"({n_micro_lenses},{n_micro_lenses}) is too large for a volume_shape: {self.optical_info['volume_shape'][1:]}. " + \
            f"Increase the volume_shape to at least [{min_required_volume_size+1},{min_required_volume_size+1}]"
        raise_error = False # DEBUG: set to False to avoid raising error
        if raise_error:
            assert min_required_volume_size <= volume_shape[1] and min_required_volume_size <= volume_shape[2], err_message

        # Traverse volume for every ray, and generate intensity images or retardance and azimuth images

        # Initialize a list to store the final concatenated images
        full_img_list = [None] * 5

        # Calculate shift for odd number of microlenses
        odd_mla_shift = np.mod(n_micro_lenses, 2)

        ml_ii_idx = 0
        # Iterate over each row of microlenses (y direction)
        if self.verbose:
            row_iterable = tqdm(
                range(-n_ml_half, n_ml_half + odd_mla_shift),
                desc=f'Computing rows of microlenses {self.backend}'
            )
        else:
            row_iterable = range(-n_ml_half, n_ml_half + odd_mla_shift)

        for ml_ii in row_iterable:
            # Initialize a list for storing concatenated images of the current row
            full_img_row_list = [None] * 5
            ml_jj_idx = 0
            # Iterate over each column of microlenses in the current row (x direction)
            for ml_jj in range(-n_ml_half, n_ml_half+odd_mla_shift):

                # Calculate the offset to the top corner of the volume in front of
                #   the current microlens (ml_ii, ml_jj)
                current_offset = self._calculate_current_offset(
                    ml_ii, ml_jj, n_voxels_per_ml, n_micro_lenses
                    )

                # Generate (intensity or ret/azim) images for the current microlens,
                #   by passing an offset to this function
                #   depending on the microlens and the super resolution
                current_mla_index = (ml_jj_idx, ml_ii_idx)
                start_time = time.time()
                img_list = self.generate_images(volume_in, current_offset,
                                intensity, mla_index=current_mla_index)
                execution_time = time.time() - start_time
                mla_index = (ml_jj_idx, ml_ii_idx)
                if mla_index not in self.mla_execution_times:
                    self.mla_execution_times[mla_index] = 0
                self.mla_execution_times[mla_index] += execution_time

                # Concatenate the generated images with the images of the current row
                if full_img_row_list[0] is None:
                    full_img_row_list = img_list
                else:
                    full_img_row_list = self._concatenate_images(
                        full_img_row_list, img_list, axis=0
                        )

                ml_jj_idx += 1

            # Concatenate the row images with the full image list
            if full_img_list[0] is None:
                full_img_list = full_img_row_list
            else:
                full_img_list = self._concatenate_images(full_img_list, full_img_row_list, axis=1)

            ml_ii_idx += 1

        return full_img_list

    def _calculate_min_volume_size(self, num_microlenses, num_voxels_per_ml):
        return int(self.voxel_span_per_ml + (num_microlenses * num_voxels_per_ml))

    def _validate_volume_size(self, min_required_volume_size, volume_shape):
        if min_required_volume_size > volume_shape[1] or min_required_volume_size > volume_shape[2]:
            print(f"WARNING: The required volume size ({min_required_volume_size}) exceeds the provided volume shape {volume_shape[1:]}.")
            raise_error = False # DEBUG: set to False to avoid raising error
            if raise_error:
                raise ValueError(f"The required volume size ({min_required_volume_size}) exceeds the provided volume shape {volume_shape[1:]}.")

    def _calculate_current_offset(self, row_index, col_index,
                                  num_voxels_per_ml, num_microlenses):
        """Maps the position of a microlens in its array to the corresponding
        position in the volumetric data, identified by its row and column
        indices. This function calculates the offset to the top corner of the
        volume in front of the current microlens.

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
        scaled_indices = np.array([num_voxels_per_ml * row_index,
                                   num_voxels_per_ml * col_index])

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
        to the ray tracing process."""
        if intensity:
            return self.intensity_images(volume, microlens_offset=offset)
        else:
            return self.ret_and_azim_images(volume, microlens_offset=offset,
                                            mla_index=mla_index)

    def _concatenate_images(self, img_list1, img_list2, axis):
        if self.backend == BackEnds.NUMPY:
            return [np.concatenate((img1, img2), axis) for img1, img2 in zip(img_list1, img_list2)]
        elif self.backend == BackEnds.PYTORCH:
            return [torch.concatenate((img1, img2), axis) for img1, img2 in zip(img_list1, img_list2)]

    def retardance(self, JM):
        '''Phase delay introduced between the fast and slow axis in a Jones Matrix'''
        if self.backend == BackEnds.NUMPY:
            e1, e2 = np.linalg.eigvals(JM)
            phase_diff = np.angle(e1) - np.angle(e2)
            retardance = np.abs(phase_diff)
        elif self.backend == BackEnds.PYTORCH:
            x = torch.linalg.eigvals(JM)
            if JM.ndim == 2:
                # JM is a single 2x2 matrix
                retardance = (torch.angle(x[1]) - torch.angle(x[0])).abs()
            elif JM.ndim == 3:
                # JM is a batch of 2x2 matrices
                retardance = (torch.angle(x[:, 1]) - torch.angle(x[:, 0])).abs()
            else:
                raise ValueError("JM must be either a 2x2 matrix or a batch of 2x2 matrices.")
            # retardance = (torch.angle(x[:,1]) - torch.angle(x[:,0])).abs()
        else:
            raise NotImplementedError
        return retardance

    def azimuth(self, JM):
        '''Rotation angle of the fast axis (neg phase)'''
        if self.backend == BackEnds.NUMPY:
            diag_sum = JM[0, 0] + JM[1, 1]
            diag_diff = JM[1, 1] - JM[0, 0]
            off_diag_sum = JM[0, 1] + JM[1, 0]
            a = np.imag(diag_diff / diag_sum)
            b = np.imag(off_diag_sum / diag_sum)
            # if np.isclose(np.abs(a), 0.0):
            #     a = 0.0
            # if np.isclose(np.abs(b), 0.0):
            #     b = 0.0
            azimuth = np.arctan2(-b, -a) / 2 + np.pi / 2
            # if np.isclose(azimuth,np.pi):
            #     azimuth = 0.0
        elif self.backend == BackEnds.PYTORCH:
            if JM.ndim == 2:
                # JM is a single 2x2 matrix
                diag_sum = JM[0, 0] + JM[1, 1]
                diag_diff = JM[1, 1] - JM[0, 0]
                off_diag_sum = JM[0, 1] + JM[1, 0]     
            elif JM.ndim == 3:
                # JM is a batch of 2x2 matrices           
                diag_sum = (JM[:, 0, 0] + JM[:, 1, 1])
                diag_diff = (JM[:, 1, 1] - JM[: ,0, 0])
                off_diag_sum = JM[:, 0, 1] + JM[:, 1, 0]

            a = (diag_diff / diag_sum).imag
            b = (off_diag_sum / diag_sum).imag

            # atan2 with zero entries causes nan in backward, so let's filter them out

            # Intermediate variables for zero tensor
            zero_a = torch.tensor(0.0, dtype=a.dtype, device=a.device)
            zero_b = torch.tensor(0.0, dtype=b.dtype, device=b.device)
            zero_for_a = torch.zeros([1], dtype=a.dtype, device=a.device)
            zero_for_b = torch.zeros([1], dtype=b.dtype, device=b.device)

            # Check if a and b are scalar values (zero-dimensional)
            if a.ndim == 0 and b.ndim == 0:
                # Handle the scalar case
                azimuth = zero_a
                if not torch.isclose(a, zero_a) or not torch.isclose(b, zero_b):
                    azimuth = torch.arctan2(-b, -a) / 2.0 + torch.pi / 2.0
            else:
                # Handle the non-scalar case
                azimuth = torch.zeros_like(a)
                close_to_zero_a = torch.isclose(a, zero_for_a)
                close_to_zero_b = torch.isclose(b, zero_for_b)
                zero_a_b = close_to_zero_a.bitwise_and(close_to_zero_b)
                azimuth[~zero_a_b] = torch.arctan2(-b[~zero_a_b], -a[~zero_a_b]) / 2.0 + torch.pi / 2.0

            # TODO: if output azimuth is pi, make it 0 and vice-versa (arctan2 bug)
            # zero_index = torch.isclose(azimuth, torch.zeros([1]), atol=1e-5)
            # pi_index = torch.isclose(azimuth, torch.tensor(torch.pi), atol=1e-5)
            # azimuth[zero_index] = torch.pi
            # azimuth[pi_index] = 0
        return azimuth

    def calc_cummulative_JM_of_ray(self, volume_in : BirefringentVolume,
                                   microlens_offset=[0,0], mla_index=(0, 0)):
        if self.backend==BackEnds.NUMPY:
            # TODO: check if the function calling is appropriate
            return self.calc_cummulative_JM_of_ray_numpy(volume_in, microlens_offset)
        elif self.backend==BackEnds.PYTORCH:
            return self.calc_cummulative_JM_of_ray_torch(volume_in,
                                        microlens_offset, mla_index=mla_index)

    def calc_cummulative_JM_of_ray_numpy(self, i, j,
                                         volume_in : BirefringentVolume, microlens_offset=[0,0]):
        '''For the (i,j) pixel behind a single microlens'''
        # Fetch precomputed Siddon parameters
        voxels_of_segs, ell_in_voxels = self.ray_vol_colli_indices, self.ray_vol_colli_lengths
        # rays are stored in a 1D array, let's look for index i,j
        n_ray = j + i * self.optical_info['pixels_per_ml']
        rayDir = self.ray_direction_basis[n_ray][:]

        JM_list = []
        try:
            for m in range(len(voxels_of_segs[n_ray])):
                ell = ell_in_voxels[n_ray][m]
                vox = voxels_of_segs[n_ray][m]

                # Check if indices are within bounds
                y_index = vox[1] + microlens_offset[0]
                z_index = vox[2] + microlens_offset[1]
                y_in_bounds = 0 <= y_index < volume_in.Delta_n.shape[1]
                z_in_bounds = 0 <= z_index < volume_in.Delta_n.shape[2]
                if not (y_in_bounds and z_in_bounds):
                    err_msg = ("Cumulative Jones Matrix computation failed. " +
                        "Index out of bounds: Attempted to access Delta_n at index "
                        f"[{vox[0]}, {y_index}, {z_index}], but this is outside "
                        f"the valid range of Delta_n's shape {volume_in.Delta_n.shape}.")
                    raise IndexError(err_msg)

                Delta_n = volume_in.Delta_n[vox[0],
                                            vox[1]+microlens_offset[0],
                                            vox[2]+microlens_offset[1]]
                opticAxis = volume_in.optic_axis[:,
                                                vox[0],
                                                vox[1]+microlens_offset[0],
                                                vox[2]+microlens_offset[1]]
                JM = self.voxRayJM(Delta_n, opticAxis, rayDir, ell, self.optical_info['wavelength'])
                JM_list.append(JM)
        except IndexError as e:
            raise IndexError(err_msg)
        except:
            raise Exception("Cumulative Jones Matrix computation failed. " +
                "Error accessing the volume, try increasing the volume size in Y-Z")
        material_JM = BirefringentRaytraceLFM.rayJM_numpy(JM_list)
        return material_JM

    def calc_cummulative_JM_of_ray_torch(self, volume_in : BirefringentVolume,
            microlens_offset=[0,0], all_rays_at_once=False, mla_index=(0, 0)):
        """ Computes the cumulative Jones Matrices (JM) for all rays defined in
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
        # DEBUG
        # assert not all(element == voxels_of_segs[0] for element in voxels_of_segs)
        # Note: if all elements of voxels_of_segs are equal, then all of
        #   self.ray_vol_colli_indices may be equal

        # DEBUG
        # print("DEBUG: making the optical info of volume and self the same")
        # print("vol in: ", volume_in.optical_info)
        # print("self in: ", self.optical_info)
        # print({self.optical_info[k] - volume_in.optical_info[k] for k in self.optical_info.items()})
        # volume_in.optical_info = self.optical_info
        # try:
        #     errors.compare_dicts(self.optical_info, volume_in.optical_info)
        # except ValueError as e:
        #     print('Optical info between ray-tracer and volume mismatch. ' + \
        #     'This might cause issues on the border microlenses.')

        # Initialize material Jones Matrix
        # Note: This could allow the try statement to be removed, but it is kept for clarity
        material_JM = None

        ###### Mask out the rays that lead to nonzero pixels
        if self.use_lenslet_based_filtering:
            # DEBUG
            # err_message = f"mla_index {mla_index} is not in nonzero_pixels_dict"
            # assert mla_index in self.nonzero_pixels_dict, err_message
            ell_in_voxels, ray_dir_basis, collision_indices = self._filter_ray_data(mla_index)
            if all_rays_at_once:
                err_message = ("all_rays_at_once not implemented " +
                               "for lenslet-based filtering of rays")
                raise NotImplementedError(err_message)
            voxels_of_segs = self._update_vox_indices_shifted(
                    microlens_offset, collision_indices
                    )
            err_message = "The list of voxels of segments should be the same " + \
                "length as the list of filtered ray volume collision indices."
            assert len(voxels_of_segs) == len(collision_indices), err_message

        else:
            ell_in_voxels = self.ray_vol_colli_lengths
            ray_dir_basis = self.ray_direction_basis
            # Determine voxel indices based on the processing mode. The voxel
            #    indices correspond to the voxels that each ray segment traverses.
            if all_rays_at_once:
                voxels_of_segs = self.vox_indices_ml_shifted_all
            else:
                if mla_index not in self.vox_indices_by_mla_idx.keys():
                    vox_list = self._gather_voxels_of_rays_pytorch(
                        microlens_offset, self.ray_vol_colli_indices
                        )
                    self.vox_indices_by_mla_idx[mla_index] = vox_list
                voxels_of_segs = self.vox_indices_by_mla_idx[mla_index]

        # Process interactions of all rays with each voxel
        # Iterate the interactions of all rays with the m-th voxel
        # Some rays interact with less voxels,
        #   so we mask the rays valid with rays_with_voxels.
        material_JM = self._get_default_JM()
        for m in range(ell_in_voxels.shape[1]):
            # Determine which rays have remaining voxels to traverse
            rays_with_voxels = [len(vx) > m for vx in voxels_of_segs]
            # n_rays_with_voxels = sum(rays_with_voxels)
            # print(f"The number of rays with voxels to transverse at this step is {n_rays_with_voxels}")

            # Get the length that each ray travels through the m-th voxel
            ell = ell_in_voxels[rays_with_voxels, m]

            # Get the voxel coordinates each ray interacts with
            vox = [vx[m] for ix, vx in enumerate(voxels_of_segs) if rays_with_voxels[ix]]

            try:
                # Extract the birefringence and optic axis information from the volume
                Delta_n = (volume_in.Delta_n_combined[vox] if OPTIMIZING_MODE
                           else volume_in.Delta_n[vox])
                opticAxis = volume_in.optic_axis[:, vox].permute(1,0)

                # Subset of precomputed ray directions that interact with voxels in this step
                filtered_ray_directions = ray_dir_basis[:, rays_with_voxels, :]

                # Compute the interaction from the rays with their corresponding voxels
                JM = self.voxRayJM(Delta_n=Delta_n, opticAxis=opticAxis,
                                   rayDir=filtered_ray_directions, ell=ell,
                                   wavelength=self.optical_info['wavelength'])

                # Combine the current Jones Matrix with the cumulative one
                if m == 0:
                    material_JM = JM
                else:
                    material_JM[rays_with_voxels, ...] = \
                        material_JM[rays_with_voxels, ...] @ JM
            except IndexError as e:
                err_msg = ("Cumulative Jones Matrix computation failed. " +
                    "At least one voxel index is out of bounds for the Delta_n " +
                    f"which is of shape {volume_in.Delta_n.shape}. " +
                    f"The max of the list of voxel indices (length {len(vox)}) is {max(vox)}.")
                raise IndexError(err_msg)
            except:
                raise Exception("Cumulative Jones Matrix computation failed. " +
                    "Error accessing the volume, try increasing the volume size in Y-Z")

        return material_JM

    def _get_default_JM(self):
        """Returns the default Jones Matrix for a ray that does not
        interact with any voxels. This is the identity matrix.
        """
        if self.backend == BackEnds.NUMPY:
            return np.array([[1, 0], [0, 1]])
        elif self.backend == BackEnds.PYTORCH:
            # .repeat(ell_in_voxels.shape[0], 1, 1)
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
                [RayTraceLFM.ravel_index((vox[ix][0],
                        vox[ix][1] + microlens_offset[0],
                        vox[ix][2] + microlens_offset[1]),
                self.optical_info['volume_shape']) for ix in range(len(vox))]
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
        list_of_voxel_lists = [
        [RayTraceLFM.ravel_index((vox[ix][0],
                vox[ix][1] + microlens_offset[0],
                vox[ix][2] + microlens_offset[1]),
        self.optical_info['volume_shape']) for ix in range(len(vox))]
        for vox in collision_indices
        ]

        return list_of_voxel_lists

    def _count_vox_raytrace_occurrences(self, zero_retardance_voxels=False):
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
        assert self.vox_indices_by_mla_idx, "Voxel indices data must be populated first."
        n_micro_lenses = self.optical_info['n_micro_lenses']
        count = Counter()
        for ml_ii_idx in range(n_micro_lenses):
            for ml_jj_idx in range(n_micro_lenses):
                mla_index = (ml_jj_idx, ml_ii_idx)
                vox_indices = self.vox_indices_by_mla_idx[mla_index]

                if zero_retardance_voxels:
                    # Get the boolean mask for the pixels that are not zero
                    nonzero_mask = self._form_mask_from_nonzero_pixels_dict(mla_index)
                    # Find the voxels that lead to a zero pixel in the retardance image
                    zero_ret_vox_indices = [vox_indices[i] for i, nonzero_bool in enumerate(nonzero_mask) if not nonzero_bool]
                    vox_indices = zero_ret_vox_indices

                flat_list = [item for sublist in vox_indices for item in sublist]
                count.update(flat_list)
                # print(f"mla_index: {mla_index}, counter: {count}")
        return count

    def identify_voxels_repeated_zero_ret(self):
        counts = self._count_vox_raytrace_occurrences(zero_retardance_voxels=True)
        vox_list = filter_keys_by_count(counts, 2)
        return vox_list

    def _filter_ray_data(self, mla_index):
        """
        Extract the ray tracing variables that contribute to the image.
        This is done by applying a mask to the ray tracing varibles. No class
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

        assert self.ray_vol_colli_lengths is not None, "Ray data must be populated first."
        assert self.ray_direction_basis is not None, "Ray data must be populated first."
        assert self.ray_vol_colli_indices is not None, "Ray data must be populated first."

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

        return ell_in_voxels_filtered, ray_dir_basis_filtered, ray_vol_colli_indices_filtered

    def ret_and_azim_images(self, volume_in : BirefringentVolume,
                            microlens_offset=[0,0], mla_index=(0, 0)):
        '''Calculate retardance and azimuth values for a ray with a Jones Matrix.'''
        if self.backend==BackEnds.NUMPY:
            # TODO: pass mla_index argument into the numpy function
            return self.ret_and_azim_images_numpy(volume_in, microlens_offset)
        elif self.backend==BackEnds.PYTORCH:
            return self.ret_and_azim_images_torch(volume_in, microlens_offset,
                                                  mla_index=mla_index)

    def ret_and_azim_images_numpy(self, volume_in : BirefringentVolume,
                                  microlens_offset=[0,0]):
        '''Calculate retardance and azimuth values for a ray with a Jones Matrix.'''
        pixels_per_ml = self.optical_info['pixels_per_ml']
        ret_image = np.zeros((pixels_per_ml, pixels_per_ml))
        azim_image = np.zeros((pixels_per_ml, pixels_per_ml))
        for i in range(pixels_per_ml):
            for j in range(pixels_per_ml):
                if np.isnan(self.ray_entry[0, i, j]):
                    ret_image[i, j] = 0
                    azim_image[i, j] = 0
                else:
                    effective_JM = self.calc_cummulative_JM_of_ray_numpy(
                                            i, j, volume_in, microlens_offset
                                            )
                    ret_image[i, j] = self.retardance(effective_JM)
                    if np.isclose(ret_image[i, j], 0.0):
                        azim_image[i, j] = 0
                    else:
                        azim_image[i, j] = self.azimuth(effective_JM)
        return [ret_image, azim_image]

    def ret_and_azim_images_mla_torch(self, volume_in : BirefringentVolume):
        '''This function computes the retardance and azimuth images
        of the precomputed rays going through a volume for all rays at once.'''

        # Calculate the number of pixels in the microlens array
        pix_per_lenslet = self.optical_info['pixels_per_ml']
        num_micro_lenses = self.optical_info['n_micro_lenses']
        pixels_per_mla = pix_per_lenslet * num_micro_lenses

        # Calculate Jones Matrices for all rays
        effective_JM = self.calc_cummulative_JM_of_ray_torch(
            volume_in, all_rays_at_once=True
            )
        # Calculate retardance and azimuth
        retardance = self.retardance(effective_JM)
        azimuth = self.azimuth(effective_JM)

        # Create output images
        ret_image = torch.zeros((pixels_per_mla, pixels_per_mla), dtype=torch.float32,
                                requires_grad=True, device=self.get_device())
        azim_image = torch.zeros((pixels_per_mla, pixels_per_mla), dtype=torch.float32,
                                 requires_grad=True, device=self.get_device())
        ret_image.requires_grad = False
        azim_image.requires_grad = False

        # Fill the values in the images
        ray_indices_all = self.ray_valid_indices_all
        ret_image[ray_indices_all[0, :], ray_indices_all[1, :]] = retardance
        azim_image[ray_indices_all[0, :], ray_indices_all[1, :]] = azimuth
        # Alternative version
        # ret_image = torch.sparse_coo_tensor(indices = self.ray_valid_indices,
        #                       values = retardance, size=(pixels_per_ml, pixels_per_ml)).to_dense()
        # azim_image = torch.sparse_coo_tensor(indices = self.ray_valid_indices,
        #                       values = azimuth, size=(pixels_per_ml, pixels_per_ml)).to_dense()
        return [ret_image, azim_image]

    def ret_and_azim_images_torch(self, volume_in : BirefringentVolume,
                                  microlens_offset=[0,0], mla_index=(0, 0)):
        """
        Computes the retardance and azimuth images for a given volume and
        microlens offset using PyTorch.

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
        # Fetch the number of pixels per microlens array from the optic configuration
        pixels_per_ml = self.optic_config.mla_config.n_pixels_per_mla

        # Calculate Jones Matrices for all rays given the volume and microlens offset
        effective_JM = self.calc_cummulative_JM_of_ray(volume_in,
                                        microlens_offset, mla_index=mla_index)

        # Calculate retardance and azimuth from the effective Jones Matrices
        retardance = self.retardance(effective_JM)
        azimuth = self.azimuth(effective_JM)

        # Initialize output images for retardance and azimuth on the appropriate device
        ret_image = torch.zeros(
            (pixels_per_ml, pixels_per_ml), dtype=torch.float32,
             requires_grad=True, device=self.get_device()
             )
        azim_image = torch.zeros(
            (pixels_per_ml, pixels_per_ml), dtype=torch.float32,
             requires_grad=True, device=self.get_device()
             )
        ret_image.requires_grad = False
        azim_image.requires_grad = False

        # Retrieve the ray indices specific to the lenslet
        current_lenslet_indices = self._retrieve_lenslet_indices(mla_index)
        # Fill the calculated values into the images at the lenslet indices
        ret_image[current_lenslet_indices[0,:], current_lenslet_indices[1,:]] = retardance
        azim_image[current_lenslet_indices[0,:], current_lenslet_indices[1,:]] = azimuth

        # Alternative implementation using sparse tensors (commented out)
        # ret_image = torch.sparse_coo_tensor(indices = self.ray_valid_indices,
        #                       values = retardance, size=(pixels_per_ml, pixels_per_ml)).to_dense()
        # azim_image = torch.sparse_coo_tensor(indices = self.ray_valid_indices,
        #                       values = azimuth, size=(pixels_per_ml, pixels_per_ml)).to_dense()

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

    def intensity_images(self, volume_in : BirefringentVolume, microlens_offset=[0,0]):
        '''Calculate intensity images using Jones Calculus. The polarizer and
        analyzer are applied to the cummulated Jones matrices.'''
        analyzer = self.optical_info['analyzer']
        swing = self.optical_info['polarizer_swing']
        pixels_per_ml = self.optical_info['pixels_per_ml']
        lenslet_JM = self.calc_cummulative_JM_lenslet(volume_in, microlens_offset)
        intensity_image_list = [np.zeros((pixels_per_ml, pixels_per_ml))] * 5

        # if not self.MLA_volume_geometry_ready:
        #     self.precompute_MLA_volume_geometry()

        for setting in range(5):
            polarizer = JonesMatrixGenerators.universal_compensator_modes(setting=setting,
                                                                          swing=swing)
            pol_hor = polarizer @ JonesVectorGenerators.horizonal()
            if self.backend == BackEnds.NUMPY:
                E_out = analyzer @ lenslet_JM @ pol_hor
                intensity = np.linalg.norm(E_out, axis=2) ** 2
                intensity_image_list[setting] = intensity
            else:
                intensity_image_list[setting] = torch.zeros(
                    (pixels_per_ml, pixels_per_ml), dtype=torch.float32, device=self.get_device()
                    )
                pol_torch = torch.from_numpy(pol_hor).type(torch.complex64)
                ana_torch =  torch.from_numpy(analyzer).type(torch.complex64)
                E_out = ana_torch @ lenslet_JM @ pol_torch
                intensity = torch.linalg.norm(E_out, axis=1) ** 2
                intensity_image_list[setting][self.ray_valid_indices[0,:],
                                              self.ray_valid_indices[1,:]
                                             ] = intensity

        return intensity_image_list

    def calc_cummulative_JM_lenslet(self, volume_in : BirefringentVolume,
                                          microlens_offset=[0,0]):
        '''Calculate the Jones matrix associated with each pixel behind a lenslet.'''
        pixels_per_ml = self.optical_info['pixels_per_ml']
        lenslet = np.zeros((pixels_per_ml, pixels_per_ml, 2, 2), dtype=np.complex128)
        if self.backend == BackEnds.PYTORCH:
            lenslet = torch.from_numpy(lenslet).to(volume_in.Delta_n.device)
            is_nan = torch.isnan
            lenslet = self.calc_cummulative_JM_of_ray_torch(volume_in, microlens_offset)
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

    # TODO: once validated merge this with numpy function
    # TODO: these are re-implemented in abstract_classes in OpticalElement
    def voxRayJM(self, Delta_n, opticAxis, rayDir, ell, wavelength):
        '''Compute Jones matrix associated with a particular ray and voxel combination'''
        # nromAxis = np.norm(optixAxis)
        # small_value = 1e-10
        # normAxis[normAxis == 0] = small_value

        # result = opticAxis / normAxis
        if self.backend == BackEnds.NUMPY:
            # Azimuth is the angle of the slow axis of retardance.
            azim = np.arctan2(np.dot(opticAxis, rayDir[1]), np.dot(opticAxis, rayDir[2]))
            if Delta_n == 0:
                azim = 0
            elif Delta_n < 0:
                azim = azim + np.pi / 2
            # print(f"Azimuth angle of index ellipsoid is
            #   {np.around(np.rad2deg(azim), decimals=0)} degrees.")
            normAxis = np.linalg.norm(opticAxis)
            proj_along_ray = np.dot(opticAxis, rayDir[0])
            # np.divide(my_arr, my_arr1, out=np.ones_like(my_arr, dtype=np.float32), where=my_arr1 != 0)
            ret = abs(Delta_n) * (1 - np.dot(opticAxis, rayDir[0]) ** 2) * 2 * np.pi * ell / wavelength
            # print(f"Accumulated retardance from index ellipsoid is
            #   {np.around(np.rad2deg(ret), decimals=0)} ~ {int(np.rad2deg(ret)) % 360} degrees.")

            # TODO: compare speed
            # The following series of operations is an equivalent method as
            # offdiag = 1j * np.sin(2 * azim) * np.sin(ret / 2)
            # diag1 = np.cos(ret / 2) + 1j * np.cos(2 * azim) * np.sin(ret / 2)
            # diag2 = np.conj(diag1)
            # JM = np.array([[diag1, offdiag], [offdiag, diag2]])
            JM = JonesMatrixGenerators.linear_retarder(ret, azim)
        elif self.backend == BackEnds.PYTORCH:
            number_of_voxels = opticAxis.shape[0]
            if not torch.is_tensor(opticAxis):
                opticAxis = torch.from_numpy(opticAxis).to(Delta_n.device)
            # Dot product of optical axis and 3 ray-direction vectors
            OA_dot_rayDir = torch.linalg.vecdot(opticAxis, rayDir)
            if NORM_PROJ:
                normAxis = torch.linalg.norm(opticAxis, axis=1)
                proj_along_ray = torch.full_like(OA_dot_rayDir[0,:], fill_value=1)
                proj_along_ray[normAxis != 0] = OA_dot_rayDir[0,:][normAxis != 0] / normAxis[normAxis != 0]
                # OA_dot_rayDir[0,:][normAxis == 0] = 1
                # Azimuth is the angle of the slow axis of retardance.
                azim_unadj = 2 * torch.arctan2(OA_dot_rayDir[1,:], OA_dot_rayDir[2,:])
                ret = abs(Delta_n) * (1 - proj_along_ray ** 2) * torch.pi * ell / wavelength
            else:
                azim_unadj = 2 * torch.arctan2(OA_dot_rayDir[1,:], OA_dot_rayDir[2,:])
                ret = abs(Delta_n) * (1 - (OA_dot_rayDir[0,:]) ** 2) * torch.pi * ell / wavelength

            # TODO: check how the gradients are affected--might be a discontinuity
            adjust_azim = True
            if adjust_azim is True:
                azim = azim_unadj.clone()
                azim[Delta_n < 0] += torch.tensor(np.pi)
            else:
                azim = azim_unadj

            # The following series of operations is an equivalent, but more
            # efficient method than:
            #   JM = JonesMatrixGenerators.linear_retarder(ret, azim, self.backend)
            offdiag = 1j * torch.sin(azim) * torch.sin(ret)
            diag1 = torch.cos(ret) + 1j * torch.cos(azim) * torch.sin(ret)
            diag2 = torch.conj(diag1)
            # Construct Jones Matrix
            JM = torch.zeros([Delta_n.shape[0], 2, 2],
                             dtype=torch.complex64,
                             device=Delta_n.device)
            JM[:,0,0] = diag1
            JM[:,0,1] = offdiag
            JM[:,1,0] = offdiag
            JM[:,1,1] = diag2
            try:
                from VolumeRaytraceLFM.utils import error_handling
                error_handling.check_for_inf_or_nan(JM)
            except ValueError as e:
                print(f"Error: {e}")
            assert not torch.isnan(JM).any(), "A Jones matrix contains NaN values."
        return JM

    def vox_ray_ret_azim(self, Delta_n, opticAxis, rayDir, ell, wavelength):
        '''Calculate the effective retardance and azimuth of a ray passing through a voxel'''
        if self.backend == BackEnds.NUMPY:
            # Azimuth is the angle of the slow axis of retardance.
            azim = np.arctan2(np.dot(opticAxis, rayDir[1]), np.dot(opticAxis, rayDir[2]))
            if Delta_n == 0:
                azim = 0
            elif Delta_n < 0:
                azim = azim + np.pi / 2
            # print(f"Azimuth angle of index ellipsoid is
            #   {np.around(np.rad2deg(azim), decimals=0)} degrees.")
            normAxis = np.linalg.norm(opticAxis)
            proj_along_ray = np.dot(opticAxis, rayDir[0])
            # np.divide(my_arr, my_arr1, out=np.ones_like(my_arr, dtype=np.float32), where=my_arr1 != 0)
            ret = abs(Delta_n) * (1 - np.dot(opticAxis, rayDir[0]) ** 2) * 2 * np.pi * ell / wavelength
        else:
            raise NotImplementedError("Not implemented for pytorch yet.")
        return ret, azim

    def vox_ray_matrix(self, ret, azim):
        '''Calculate the Jones matrix associated with a particular ray and voxel combination'''
        if self.backend == BackEnds.NUMPY:
            JM = JonesMatrixGenerators.linear_retarder(ret, azim)
            pass
        else:
            raise NotImplementedError("Not implemented for pytorch yet.")
            offdiag = 1j * torch.sin(azim) * torch.sin(ret)
            diag1 = torch.cos(ret) + 1j * torch.cos(azim) * torch.sin(ret)
            diag2 = torch.conj(diag1)
            # Construct Jones Matrix
            JM = torch.zeros([Delta_n.shape[0], 2, 2], dtype=torch.complex64, device=Delta_n.device)
            JM[:,0,0] = diag1
            JM[:,0,1] = offdiag
            JM[:,1,0] = offdiag
            JM[:,1,1] = diag2
        return JM

    def clone(self):
        # Code to create a copy of this instance
        new_instance = BirefringentVolume(...)
        return new_instance

    @staticmethod
    def rayJM_numpy(JMlist):
        '''Computes product of Jones matrix sequence
        Equivalent method: np.linalg.multi_dot([JM1, JM2])
        '''
        product = np.identity(2)
        for JM in JMlist:
            product = product @ JM
        return product

    @staticmethod
    def rayJM_torch(JMlist, voxels_of_segs):
        '''Computes product of Jones matrix sequence
        Equivalent method: torch.linalg.multi_dot([JM1, JM2])
        '''
        n_rays = len(JMlist[0])
        product = torch.tensor([[1.0,0],[0,1.0]],
                               dtype=torch.complex64,
                               device=JMlist[0].device).unsqueeze(0).repeat(n_rays,1,1)
        for ix, JM in enumerate(JMlist):
            rays_with_voxels = [len(vx)>ix for vx in voxels_of_segs]
            product[rays_with_voxels,...] = product[rays_with_voxels,...] @ JM
        return product

    def apply_polarizers(self, material_JM):
        '''Apply the polarizer and analyzer to a product of Jones matrices representing the
        material. material_JM can be a 2x2 array or probably a list/array of 2x2 array.'''
        if self.backend==BackEnds.PYTORCH:
            # Possibly need to attach .to(Delta_n.device)
            polarizer = torch.from_numpy(self.optical_info['polarizer']).type(torch.complex64)
            analyzer = torch.from_numpy(self.optical_info['analyzer']).type(torch.complex64)
        elif self.backend==BackEnds.NUMPY:
            polarizer = self.optical_info['polarizer']
            analyzer = self.optical_info['analyzer']
        effective_JM = analyzer @ material_JM @ polarizer
        return effective_JM

    @staticmethod
    def ret_and_azim_from_intensity(image_list, swing):
        '''Note: this function is still in development.'''
        if len(image_list) != 5:
            raise ValueError(f"Expected 5 images, got {len(imgs)}.")
        # The order of the images matters!
        imgs = [image_list[0], image_list[2], image_list[3], image_list[1], image_list[4]]
        # using arctan vs arctan2 does not seem to make a difference
        epsilon = np.finfo(float).eps
        # A = (imgs[1] - imgs[2]) / (imgs[1] + imgs[2] - 2 * imgs[0] + epsilon) * np.arctan(swing / 2)
        # B = (imgs[4] - imgs[3]) / (imgs[4] + imgs[3] - 2 * imgs[0] + epsilon) * np.arctan(swing / 2)
        # Ensure that the denominator is not zero by adding epsilon
        denominator_A = (imgs[1] + imgs[2] - 2 * imgs[0] + epsilon)
        denominator_B = (imgs[4] + imgs[3] - 2 * imgs[0] + epsilon)
        # Check where the denominator is zero and set those values to epsilon
        denominator_A[denominator_A == 0] = epsilon
        denominator_B[denominator_B == 0] = epsilon
        # Calculate A and B with the safe denominator
        A = (imgs[1] - imgs[2]) / denominator_A * np.arctan(swing / 2)
        B = (imgs[4] - imgs[3]) / denominator_B * np.arctan(swing / 2)
        # ret = np.arctan(np.sqrt(A ** 2, B ** 2))
        ret = np.arctan(np.sqrt(A ** 2 + B ** 2))
        test_value = imgs[1] + imgs[2] - 2 * imgs[0]
        indices = np.where(test_value < 0)
        ret[indices] = 2 * np.pi - ret[indices]
        # azim = 0.5 * np.arctan2(A, B) + np.pi / 2
        # azim = 0.5 * np.arctan2(A, B) + np.pi / 2
        azim = 0.5 * np.arctan2(B, A) + np.pi / 2
        return [ret, azim]
