
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
    
    def calc_cummulative_JM_of_ray(self, voxel_parameters):
        '''This function computes the Jones Matrices of all rays defined in this object.
            It uses pytorch's batch dimension to store each ray, and process them in parallel'''

        # Fetch the voxels traversed per ray and the lengths that each ray travels through every voxel
        voxels_of_segs, ell_in_voxels = self.ray_vol_colli_indexes, self.ray_vol_colli_lengths
        # Fetch the ray's directions
        rays = self.ray_valid_direction
        # Calculate the ray's direction with the two normalized perpendicular directions
        # Returns a list size 3, where each element is a torch tensor shaped [n_rays, 3]
        rayDir = calc_rayDir(rays)
        # Init an array to store the Jones matrices.
        JM_list = []

        # Iterate the interactions of all rays with the m-th voxel
        # Some rays interact with less voxels, so we mask the rays valid
        # for this step with rays_with_voxels
        for m in range(self.ray_vol_colli_lengths.shape[1]):
            # Check which rays still have voxels to traverse
            rays_with_voxels = [len(vx)>m for vx in voxels_of_segs]
            # How many rays at this step
            n_rays_with_voxels = sum(rays_with_voxels)
            # The lengths these rays traveled through the current voxels
            ell = ell_in_voxels[rays_with_voxels,m]
            # The voxel coordinates each ray collides with
            vox = [vx[m] for ix,vx in enumerate(voxels_of_segs) if rays_with_voxels[ix]]

            # Extract the information from the volume
            my_params = voxel_parameters[:, [v[0] for v in vox], [v[1] for v in vox], [v[2] for v in vox]]
            # Retardance
            Delta_n = my_params[0]
            # And axis
            opticAxis = my_params[1:].permute(1,0)

            # Grab the subset of precomputed ray directions that have voxels in this step
            filtered_rayDir = [rayDir[0][rays_with_voxels,:], rayDir[1][rays_with_voxels,:], rayDir[2][rays_with_voxels,:]]

            # Initiallize identity Jones Matrices, shape [n_rays_with_voxels, 2, 2]
            JM = torch.tensor([[1.0,0],[0,1.0]], dtype=torch.complex64).unsqueeze(0).repeat(n_rays_with_voxels,1,1)
            # Only compute if there's an Delta_n
            # Create a mask of the valid voxels
            valid_voxel = Delta_n!=0
            if valid_voxel.sum() > 0:
                # Compute the interaction from the rays with their corresponding voxels
                JM[valid_voxel, :, :] = voxRayJM(Delta_n = Delta_n[valid_voxel], 
                                                opticAxis = opticAxis[valid_voxel, :], 
                                                rayDir = [filtered_rayDir[0][valid_voxel], filtered_rayDir[1][valid_voxel], filtered_rayDir[2][valid_voxel]], 
                                                ell = ell[valid_voxel])
            # Store current interaction step
            JM_list.append(JM)
        # JM_list contains m steps of rays interacting with voxels
        # Each JM_list[m] is shaped [n_rays, 2, 2]
        # We pass voxels_of_segs to compute which rays have a voxel in each step
        effective_JM = rayJM(JM_list, voxels_of_segs)
        return effective_JM

    def ret_and_azim_images(self, volume_in : VolumeLFM):
        '''This function computes the retardance and azimuth images of the precomputed rays going through a volume'''
        # Fetch needed variables
        pixels_per_ml = self.optic_config.mla_config.n_pixels_per_mla
        # Create output images
        ret_image = torch.zeros((pixels_per_ml, pixels_per_ml), requires_grad=True)
        azim_image = torch.zeros((pixels_per_ml, pixels_per_ml), requires_grad=True)
        
        # Calculate Jones Matrices for all rays
        effective_JM = self.calc_cummulative_JM_of_ray(volume_in.voxel_parameters)
        # Calculate retardance and azimuth
        retardance = calc_retardance(effective_JM)
        azimuth = calc_azimuth(effective_JM)
        ret_image.requires_grad = False
        azim_image.requires_grad = False
        # Assign the computed ray values to the image pixels
        for ray_ix, (i,j) in enumerate(self.ray_valid_indexes):
            ret_image[i, j] = retardance[ray_ix]
            azim_image[i, j] = azimuth[ray_ix]
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
        
        # Enable gradients for auto-differentiation 
        volume_ref.voxel_parameters.requires_grad = True
        return volume_ref

    
    @staticmethod
    def generate_random_volume(volume_shape):
        Delta_n = torch.FloatTensor(*volume_shape).uniform_(0, .01)
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
            # vol[1, z_size//2, :, :] = 0.5
            vol[2, z_size//2, :, :] = 1
            return vol
        random_data = BirefringentRaytraceLFM.generate_random_volume([n_planes])
        for z_ix in range(0,n_planes):
            vol[:,z_ranges[z_ix*2] : z_ranges[z_ix*2+1]] = random_data[:,z_ix].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,1,volume_shape[1],volume_shape[2])
        
        return vol

