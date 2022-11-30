
import torch
import torch.nn as nn
import torch.nn.functional as f

# Waveblocks imports
from waveblocks.blocks.optic_block import OpticBlock
from waveblocks.blocks.optic_config import *

from VolumeRaytraceLFM.abstract_classes import *

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

    def ray_trace(self, volume_in : VolumeLFM = None):
        #todo
        return 0
    
    def init_volume(self, volume_ref : VolumeLFM = None, init_mode='zeros'):
        # IF the user doesn't provide a volume, let's create one and return it
        if volume_ref is None:
            volume_ref = VolumeLFM(self.optic_config, [], self.simul_type)
        
        if init_mode=='zeros':
            volume_ref.voxel_parameters = torch.zeros([4,] + volume_ref.config.volume_shape)
        elif init_mode=='random':
            volume_ref.voxel_parameters = self.generate_random_volume(volume_ref.config.volume_shape)
        elif init_mode=='3planes':
            volume_ref.voxel_parameters = self.generate_planes_volume(volume_ref.config.volume_shape) # Perpendicular optic axes each with constant birefringence and orientation 
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
    def generate_planes_volume(volume_shape, n_planes=3):
        vol = torch.zeros([4,] + volume_shape)
        z_size = volume_shape[0]
        z_ranges = np.linspace(0, z_size-1, n_planes*2).astype(int)

        random_data = BirefringentRaytraceLFM.generate_random_volume([n_planes])
        for z_ix in range(0,n_planes):
            vol[:,z_ranges[z_ix*2] : z_ranges[z_ix*2+1]] = random_data[:,z_ix].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,1,volume_shape[1],volume_shape[2])
        
        return vol

