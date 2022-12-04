# Third party libraries imports
import torch
import torch.nn as nn
import torch.nn.functional as f
from enum import Enum
import pickle
from os.path import exists

# Waveblocks imports
from waveblocks.blocks.optic_block import OpticBlock
from waveblocks.blocks.optic_config import *
# todo: move this to a place inside the library, like utils
from ray_optics import rays_through_vol
from my_siddon import *

class SimulType(Enum):
    """ Defines which types of volumes we can have, as each type has a different ray-voxel interaction"""
    FLUOR_INTENS    = 1     # Voxels add intensity as the ray goes trough the volume: commutative
    BIREFRINGENT    = 2     # Voxels modify polarization as the ray goes trough: non commutative
    FLUOR_POLAR     = 3     # todo
    DIPOLES         = 4     # Voxels add light depending with their angle with respect to the dipole
    # etc. attenuators and di-attenuators (polarization dependent)

class VolumeLFM(OpticBlock):
    """ This is class storing voxel information depending on a type of volume"""
    def __init__(
        self, optic_config : OpticConfig, members_to_learn : list =[], simul_type : SimulType = SimulType.BIREFRINGENT
    ):
        self.simul_type = simul_type
        self.config = optic_config.volume_config

        # List of voxels with parameters assigned to each voxel. The parameters will depend on the simul_type.
        # array of parameters, followed by x, y, z, coordinates
        self.voxel_parameters = None
    
    def forward(self, coords : list):
        """ This function samples the volume on a given coordinate or list of coordinates"""
        return self.voxel_parameters[coords,...]
    
    def plot_volume_plotly(self, voxels=None, opacity=0.5):
        
        if voxels is None:
            voxels = self.voxel_parameters[0,...].clone()

        import plotly.graph_objects as go
        import numpy as np
        volume_shape = self.config.volume_shape
        volume_size_um = self.config.volume_size_um
        [dz, dxy, dxy] = self.config.voxel_size_um
        # Define grid 
        z_coords,y_coords,x_coords = np.indices(np.array(voxels.shape) + 1).astype(float)
        
        x_coords += 0.5
        y_coords += 0.5
        z_coords += 0.5
        x_coords *= dxy
        y_coords *= dxy
        z_coords *= dz
        fig = go.Figure(data=go.Volume(
            x=z_coords[:-1,:-1,:-1].flatten(),
            y=y_coords[:-1,:-1,:-1].flatten(),
            z=x_coords[:-1,:-1,:-1].flatten(),
            value=voxels.flatten(),
            # isomin=-0.1,
            # isomax=0.1,
            opacity=opacity, # needs to be small to see through all surfaces
            surface_count=20, # needs to be a large number for good volume rendering
            colorscale='inferno'
            ))
        fig.data = fig.data[::-1]
        # Draw the whole volume span
        fig.add_mesh3d(
                # 8 vertices of a cube
                x=[0, 0, volume_size_um[0], volume_size_um[0], 0, 0, volume_size_um[0], volume_size_um[0]],
                y=[0, volume_size_um[1], volume_size_um[1], 0, 0, volume_size_um[1], volume_size_um[1], 0],
                z=[0, 0, 0, 0, volume_size_um[2], volume_size_um[2], volume_size_um[2], volume_size_um[2]],
                colorbar_title='z',
                colorscale='inferno',
                opacity=0.1,
                # Intensity of each vertex, which will be interpolated and color-coded
                intensity = np.linspace(0, 1, 8, endpoint=True),
                # i, j and k give the vertices of triangles
                i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            )
        fig.show()
        return

    def get_vox_params(self, vox_index):
        '''vox_index is a tuple'''
        return self.voxel_parameters[:, vox_index]


class RayTraceLFM(OpticBlock):
    """This is a base class that takes a volume geometry and LFM geometry and calculates which arrive to each of the pixels behind each micro-lense, and discards the rest.
       This class also pre-computes how each rays traverses the volume with the Siddon algorithm.
       The interaction between the voxels and the rays is defined by each specialization of this class."""

    def __init__(
        self, optic_config : OpticConfig, members_to_learn : list =[], simul_type : SimulType = SimulType.BIREFRINGENT
    ):
        # optic_config contains mla_config and volume_config
        super(RayTraceLFM, self).__init__(
            optic_config=optic_config, members_to_learn=members_to_learn
        )
        self.simul_type = simul_type

        # Pre-compute rays and paths through the volume
        # This are defined in compute_rays_geometry
        self.ray_valid_indexes = None
        self.ray_vol_colli_indexes = None
        self.ray_vol_colli_lengths = None

    def forward(self, volume_in : VolumeLFM=None):
        # Check if type of volume is the same as input volume, if one is provided
        if volume_in is not None:
            assert volume_in.simul_type == self.simul_type, f"Error: wrong type of volume provided, this \
            ray-tracer works for {self.simul_type} and a volume {volume_in.simul_type} was provided"
        return self.ray_trace_through_volume(volume_in)
    
    def compute_rays_geometry(self, filename=None):
        if filename is not None and exists(filename):
            data = self.unpickle(filename)
            print(f'Loaded RayTraceLFM object from {filename}')
            return data


        # The valid workspace is defined by the number of micro-lenses
        valid_vol_shape = self.optic_config.mla_config.n_micro_lenses


        # Fetch needed variables
        pixels_per_ml = self.optic_config.mla_config.n_pixels_per_mla
        naObj = self.optic_config.PSF_config.NA
        nMedium = self.optic_config.PSF_config.ni
        vol_shape = [self.optic_config.volume_config.volume_shape[0],] + 2*[valid_vol_shape]
        axial_pitch,xy_pitch,xy_pitch = self.optic_config.volume_config.voxel_size_um
        vox_ctr_idx = np.array([vol_shape[0] / 2, vol_shape[1] / 2, vol_shape[2] / 2]) # in index units
        self.vox_ctr_idx = vox_ctr_idx.astype(int)
        self.voxCtr = np.array([vol_shape[0] / 2, vol_shape[1] / 2, vol_shape[2] / 2]) # in index units
        self.volCtr = [self.voxCtr[0] * axial_pitch, self.voxCtr[1] * xy_pitch, self.voxCtr[2] * xy_pitch]   # in vol units (um)
        
        # Call Geneva's function
        ray_enter, ray_exit, ray_diff = rays_through_vol(pixels_per_ml, naObj, nMedium, self.volCtr)

        # Store locally
        # 2D to 1D index
        self.ray_entry = torch.from_numpy(ray_enter).float()
        self.ray_exit = torch.from_numpy(ray_exit).float()
        self.ray_direction = torch.from_numpy(ray_diff).float()
        self.voxel_span_per_ml = 0

        i_range,j_range = self.ray_entry.shape[1:]
        # Compute Siddon's algorithm for each ray
        ray_valid_indexes = []
        ray_valid_direction = []
        ray_vol_colli_indexes = []
        ray_vol_colli_lengths = []
        for ii in range(i_range):
            for jj in range(j_range):
                start = ray_enter[:,ii,jj]
                stop = ray_exit[:,ii,jj]
                if np.any(np.isnan(start)) or np.any(np.isnan(stop)):
                    continue
                siddon_list = siddon_params(start, stop, self.optic_config.volume_config.voxel_size_um, vol_shape)
                seg_mids = siddon_midpoints(start, stop, siddon_list)
                voxels_of_segs = vox_indices(seg_mids, self.optic_config.volume_config.voxel_size_um)
                voxel_intersection_lengths = siddon_lengths(start, stop, siddon_list)

                # Store in a temporary list
                ray_valid_indexes.append((ii,jj))
                ray_vol_colli_indexes.append(voxels_of_segs)
                ray_vol_colli_lengths.append(voxel_intersection_lengths)
                ray_valid_direction.append(self.ray_direction[:,ii,jj])

                # What is the maximum span of the rays of a micro lens?
                self.voxel_span_per_ml = max([self.voxel_span_per_ml,] + [vx[1] for vx in ray_vol_colli_indexes[0]])

        
        # Maximum number of interactions, to define 
        max_ray_voxels_collision = np.max([len(D) for D in ray_vol_colli_indexes])
        n_valid_rays = len(ray_valid_indexes)

        # Create the information to store
        self.ray_valid_indexes = ray_valid_indexes
        # Store as tuples for now
        self.ray_vol_colli_indexes = ray_vol_colli_indexes #torch.zeros(n_valid_rays, max_ray_voxels_collision)
        self.ray_vol_colli_lengths = torch.zeros(n_valid_rays, max_ray_voxels_collision)
        self.ray_valid_direction = torch.zeros(n_valid_rays, 3)


        # Fill these tensors
        # todo: indexes is indices 
        for valid_ray in range(n_valid_rays):
            # Fetch the ray-voxel collision indexes for this ray
            # val_indexes = ray_vol_colli_indexes[valid_ray]
            # self.ray_vol_colli_indexes[valid_ray, :len(val_indexes)] = val_indexes
            # Fetch the ray-voxel intersection length for this ray
            val_lengths = ray_vol_colli_lengths[valid_ray]
            self.ray_vol_colli_lengths[valid_ray, :len(val_lengths)] = torch.tensor(val_lengths)
            self.ray_valid_direction[valid_ray, :] = ray_valid_direction[valid_ray]
        
        # Update volume shape information, to account for the whole workspace
        vol_shape = self.optic_config.volume_config.volume_shape
        axial_pitch,xy_pitch,xy_pitch = self.optic_config.volume_config.voxel_size_um
        vox_ctr_idx = np.array([vol_shape[0] / 2, vol_shape[1] / 2, vol_shape[2] / 2]) # in index units
        self.vox_ctr_idx = vox_ctr_idx.astype(int)
        self.voxCtr = np.array([vol_shape[0] / 2, vol_shape[1] / 2, vol_shape[2] / 2]) # in index units
        self.volCtr = [self.voxCtr[0] * axial_pitch, self.voxCtr[1] * xy_pitch, self.voxCtr[2] * xy_pitch]   # in vol units (um)

        if filename is not None:
            self.pickle(filename)
            print(f'Saved RayTraceLFM object from {filename}')
        return self

    def pickle(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    
    @staticmethod
    def unpickle(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)





    ######## Not implemented: These functions need an implementation in derived objects
    def ray_trace_through_volume(self, volume_in : VolumeLFM=None):
        """ We have a separate function as we have some basic functionality that is shared"""
        raise NotImplementedError
    
    def init_volume(self, volume_in : VolumeLFM=None):
        """ This function assigns a volume the correct internal structure for a given simul_type
        For example: a single value per voxel for fluorescence, or two values for birefringence"""
        raise NotImplementedError


    
        