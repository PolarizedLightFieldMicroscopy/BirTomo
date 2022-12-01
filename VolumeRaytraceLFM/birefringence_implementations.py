
import torch
import torch.nn as nn
import torch.nn.functional as f

# Waveblocks imports
from waveblocks.blocks.optic_block import OpticBlock
from waveblocks.blocks.optic_config import *

from VolumeRaytraceLFM.abstract_classes import *
from jones_torch import *

############ Implementations

class BirefringentRaytraceLFM(RayTraceLFM):
    """This class extends RayTraceLFM, and implements the forward function, where voxels contribute to ray's Jones-matrices with a retardance and axis in a non-commutative matter"""
    def __init__(
        self, optic_config : OpticConfig, members_to_learn : list =[]
    ):
        # optic_config contains mla_config and volume_config
        super(BirefringentRaytraceLFM, self).__init__(
            optic_config=optic_config, members_to_learn=members_to_learn, simul_type=SimulType.BIREFRINGENT
        )
        # Create tensors to store the Jones-matrices per ray

        # We either define a volume here or use one provided by the user

    def ray_trace_through_volume(self, volume_in : VolumeLFM = None):
        
        return 0
    
    def calc_cummulative_JM_of_ray(self, ray_ix, voxel_parameters):
        '''For the (i,j) pixel behind a single microlens'''
        i,j = self.ray_valid_indexes[ray_ix]
        voxels_of_segs, ell_in_voxels = self.ray_vol_colli_indexes[ray_ix], self.ray_vol_colli_lengths[ray_ix]
        ray = self.ray_direction[:,i,j]
        rayDir = calc_rayDir(ray)
        JM_list = []
        for m in range(len(voxels_of_segs)):
            ell = ell_in_voxels[m]
            vox = voxels_of_segs[m]
            my_params = voxel_parameters[:, vox[0], vox[1], vox[2]]
            Delta_n = my_params[0]
            opticAxis = my_params[1:]
            # Only compute if there's an opticAxis, if not, return identity
            if torch.all(opticAxis==0):
                JM = torch.tensor([[1.0,0],[0,1.0]], dtype=torch.complex64)
            else:
                JM = voxRayJM(Delta_n, opticAxis, rayDir, ell)
            JM_list.append(JM)
        effective_JM = rayJM(JM_list)
        return effective_JM

    def ret_and_azim_images(self, volume_in : VolumeLFM = None):
        pixels_per_ml = self.optic_config.mla_config.n_pixels_per_mla
        ret_image = np.zeros((pixels_per_ml, pixels_per_ml))
        azim_image = np.zeros((pixels_per_ml, pixels_per_ml))
        for ray_ix, (i,j) in enumerate(self.ray_valid_indexes):
            effective_JM = self.calc_cummulative_JM_of_ray(ray_ix, volume_in.voxel_parameters)
            ret_image[i, j] = calc_retardance(effective_JM)
            azim_image[i, j] = calc_azimuth(effective_JM)
        return ret_image, azim_image


########### Generate different birefringent volumes 
    def init_volume(self, volume_ref : VolumeLFM = None, init_mode='zeros'):
        # IF the user doesn't provide a volume, let's create one and return it
        if volume_ref is None:
            volume_ref = VolumeLFM(self.optic_config, [], self.simul_type)
        
        if init_mode=='zeros':
            volume_ref.voxel_parameters = torch.zeros([4,] + volume_ref.config.volume_shape)
        elif init_mode=='random':
            volume_ref.voxel_parameters = self.generate_random_volume(volume_ref.config.volume_shape)
        elif 'planes' in init_mode:
            n_planes = int(init_mode[0])
            volume_ref.voxel_parameters = self.generate_planes_volume(volume_ref.config.volume_shape, n_planes) # Perpendicular optic axes each with constant birefringence and orientation 
        elif init_mode=='psf':
            volume_ref.voxel_parameters = torch.zeros([4,] + volume_ref.config.volume_shape)
            volume_ref.voxel_parameters[:, ]
        
        return volume_ref

    
    @staticmethod
    def generate_random_volume(volume_shape):
        Delta_n = torch.FloatTensor(*volume_shape).uniform_(-5, 5)
        # Random axis
        a_0 = torch.FloatTensor(*volume_shape).uniform_(-5, 5)
        a_1 = torch.FloatTensor(*volume_shape).uniform_(-5, 5)
        a_2 = torch.FloatTensor(*volume_shape).uniform_(-5, 5)
        norm_A = (a_0**2+a_1**2+a_2**2).sqrt()
        return torch.cat((Delta_n.unsqueeze(0), (a_0/norm_A).unsqueeze(0), (a_1/norm_A).unsqueeze(0), (a_2/norm_A).unsqueeze(0)),0)
    
    @staticmethod
    def generate_planes_volume(volume_shape, n_planes=1):
        vol = torch.zeros([4,] + volume_shape)
        z_size = volume_shape[0]
        z_ranges = np.linspace(0, z_size-1, n_planes*2).astype(int)

        if n_planes==1:
            vol[0, z_size//2, :, :] = 0.1
            vol[1, z_size//2, :, :] = 1
            # vol[2, z_size//2, :, :] = 1
            return vol
        random_data = BirefringentRaytraceLFM.generate_random_volume([n_planes])
        for z_ix in range(0,n_planes):
            vol[:,z_ranges[z_ix*2] : z_ranges[z_ix*2+1]] = random_data[:,z_ix].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,1,volume_shape[1],volume_shape[2])
        
        return vol

