# Third party libraries imports
from enum import Enum
import pickle
from os.path import exists
from my_siddon import *


# Optional imports: as the classes here depend on Waveblocks Opticblock.
# We create a dummy class for the case where either Waveblocks is not installed
# Or the user just don't wan to use it
try:
    import torch
    import torch.nn as nn
    from waveblocks.blocks.optic_config import *
    from waveblocks.blocks.optic_block import OpticBlock
except:
    class OpticBlock:  # Dummy function for numpy
        def __init__(
            self, optic_config=None, members_to_learn=None,
        ):  
            pass

class SimulType(Enum):
    ''' Defines which types of volumes we can have, as each type has a different ray-voxel interaction'''
    NOT_SPECIFIED   = 0
    FLUOR_INTENS    = 1     # Voxels add intensity as the ray goes trough the volume: commutative
    BIREFRINGENT    = 2     # Voxels modify polarization as the ray goes trough: non commutative
    # FLUOR_POLAR     = 3     # 
    # DIPOLES         = 4     # Voxels add light depending with their angle with respect to the dipole
    # etc. attenuators and di-attenuators (polarization dependent)


class BackEnds(Enum):
    ''' Defines type of backend (numpy,pytorch,etc)'''
    NUMPY       = 1     # Use numpy back-end
    PYTORCH     = 2     # Use Pytorch, with auto-differentiation and GPU support.


class OpticalElement(OpticBlock):
    ''' Abstract class defining a elements, with a back-end ans some optical information'''
    default_optical_info = {'volume_shape' : 3*[1], 'voxel_size_um' : 3*[1.0], 'pixels_per_ml' : 17, 'na_obj' : 1.2, 
                'n_medium' : 1.52, 'wavelength' : 0.550, 'n_micro_lenses' : 1}
    def __init__(self, back_end : BackEnds = BackEnds.NUMPY, torch_args={},#{'optic_config' : None, 'members_to_learn' : []},
                optical_info={}):
        # Optical info is needed
        assert len(optical_info) > 0, f'Optical info (optical_info) dictionary needed: use OpticalElement.default_optical_info as reference {OpticalElement.default_optical_info}'
        # Check if back-end is torch and overwrite self with an optic block, for Waveblocks compatibility.
        if back_end==BackEnds.PYTORCH:
            # If no optic_config is provided, create one
            if 'optic_config' not in torch_args.keys():
                torch_args['optic_config'] = OpticConfig()
                torch_args['optic_config'].volume_config.volume_shape = optical_info['volume_shape']
                torch_args['optic_config'].volume_config.voxel_size_um = optical_info['voxel_size_um']
                torch_args['optic_config'].mla_config.n_pixels_per_mla = optical_info['pixels_per_ml']
                torch_args['optic_config'].mla_config.n_micro_lenses = optical_info['n_micro_lenses']
                torch_args['optic_config'].PSF_config.NA = optical_info['na_obj']
                torch_args['optic_config'].PSF_config.ni = optical_info['n_medium']
                torch_args['optic_config'].PSF_config.wvl = optical_info['wavelength']

            super(OpticalElement, self).__init__(optic_config=torch_args['optic_config'], 
                    members_to_learn=torch_args['members_to_learn'] if 'members_to_learn' in torch_args.keys() else [])

        # Store variables
        self.back_end = back_end
        self.simul_type = SimulType.NOT_SPECIFIED
        self.optical_info = optical_info

        # if we are using pytorch and waveblocks, grab system info from optic_config
        # if self.back_end == BackEnds.PYTORCH:
        #     self.optical_info = \
        #             {'volume_shape' : self.optic_config.volume_config.volume_shape, 
        #             'voxel_size_um' : self.optic_config.volume_config.voxel_size_um, 
        #             'pixels_per_ml' : self.optic_config.mla_config.n_pixels_per_mla, 
        #             'n_micro_lenses' : self.optic_config.mla_config.n_micro_lenses, 
        #             'na_obj' : self.optic_config.PSF_config.NA, 
        #             'n_medium' : self.optic_config.PSF_config.ni,
        #             'wavelength' : self.optic_config.PSF_config.wvl}


###########################################################################################
class RayTraceLFM(OpticalElement):
    '''This is a base class that takes a volume geometry and LFM geometry and calculates which arrive to each of the pixels behind each micro-lense, and discards the rest.
       This class also pre-computes how each rays traverses the volume with the Siddon algorithm.
       The interaction between the voxels and the rays is defined by each specialization of this class.'''

    def __init__(
        self, back_end : BackEnds = BackEnds.NUMPY, torch_args={},#{'optic_config' : None, 'members_to_learn' : []},
            optical_info={'volume_shape' : [11,11,11], 'voxel_size_um' : 3*[1.0], 'pixels_per_ml' : 17, 'na_obj' : 1.2, 'n_medium' : 1.52, 'wavelength' : 0.550, 'n_micro_lenses' : 1, 'n_voxels_per_ml' : 1}):
        super(RayTraceLFM, self).__init__(back_end=back_end, torch_args=torch_args, optical_info=optical_info)
        
        
        # Create dummy variables for pre-computed rays and paths through the volume
        # This are defined in compute_rays_geometry
        self.ray_valid_indices = None
        self.ray_vol_colli_indices = None
        self.ray_vol_colli_lengths = None
        self.ray_direction_basis = None

    def forward(self, volume_in ):
        # Check if type of volume is the same as input volume, if one is provided
        # if volume_in is not None:
        #     assert volume_in.simul_type == self.simul_type, f"Error: wrong type of volume provided, this \
        #     ray-tracer works for {self.simul_type} and a volume {volume_in.simul_type} was provided"
        return self.ray_trace_through_volume(volume_in)
    

###########################################################################################
    # Helper functions 

    @staticmethod
    def rotation_matrix(axis, angle):
        '''Generates the rotation matrix that will rotate a 3D vector
        around "axis" by "angle" counterclockwise.'''
        ax, ay, az = axis[0], axis[1], axis[2]
        s = np.sin(angle)
        c = np.cos(angle)
        u = 1 - c
        R_tuple = ( ( ax*ax*u + c,    ax*ay*u - az*s, ax*az*u + ay*s ),
            ( ay*ax*u + az*s, ay*ay*u + c,    ay*az*u - ax*s ),
            ( az*ax*u - ay*s, az*ay*u + ax*s, az*az*u + c    ) )
        R = np.asarray(R_tuple)
        return R

    @staticmethod
    def find_orthogonal_vec(v1, v2):
        '''v1 and v2 are numpy arrays (3d vectors)
        This function accommodates for a divide by zero error.'''
        value = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # Check if vectors are parallel or anti-parallel
        if np.linalg.norm(value) == 1:
            if v1[1] == 0:
                normal_vec = np.array([0, 1, 0])
            elif v1[2] == 0:
                normal_vec = np.array([0, 0, 1])
            elif v1[0] == 0:
                normal_vec = np.array([1, 0, 0])
            else:
                non_par_vec = np.array([1, 0, 0])
                normal_vec = np.cross(v1, non_par_vec) / np.linalg.norm(np.cross(v1, non_par_vec))
        else:
            normal_vec = np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2))
        return normal_vec

    @staticmethod
    def rotation_matrix_torch(axis, angle):
        '''Generates the rotation matrix that will rotate a 3D vector
        around "axis" by "angle" counterclockwise.'''
        ax, ay, az = axis[:,0], axis[:,1], axis[:,2]
        s = torch.sin(angle)
        c = torch.cos(angle)
        u = 1 - c
        R = torch.zeros([angle.shape[0],3,3], device=axis.device)
        R[:,0,0] = ax*ax*u + c
        R[:,0,1] = ax*ay*u - az*s
        R[:,0,2] = ax*az*u + ay*s
        R[:,1,0] = ay*ax*u + az*s
        R[:,1,1] = ay*ay*u + c
        R[:,1,2] = ay*az*u - ax*s
        R[:,2,0] = az*ax*u - ay*s
        R[:,2,1] = az*ay*u + ax*s
        R[:,2,2] = az*az*u + c
        return R

    @staticmethod
    def find_orthogonal_vec_torch(v1, v2):
        '''v1 and v2 are numpy arrays (3d vectors)
        This function accomodates for a divide by zero error.'''
        x = torch.linalg.multi_dot((v1, v2)) / (torch.linalg.norm(v1.unsqueeze(2),dim=1) * torch.linalg.norm(v2))[0]
        # Check if vectors are parallel or anti-parallel
        normal_vec = torch.zeros_like(v1)

        # Search for invalid indices 
        invalid_indices = torch.isclose(x.abs(),torch.ones([1], device=x.device))
        valid_indices = ~invalid_indices
        # Compute the invalid normal_vectors
        if invalid_indices.sum():
            for n_axis in range(3):
                normal_vec[invalid_indices,n_axis] = (v1[invalid_indices,n_axis]==0) * 1.0
                # Turn off fixed indices
                invalid_indices[v1[:,n_axis]==0] = False
            if invalid_indices.sum(): # treat remaning ones
                non_par_vec = torch.tensor([1.0, 0, 0], device=x.device).unsqueeze(0).repeat(v1.shape[0],1)
                C = torch.cross(v1[invalid_indices,:], non_par_vec[invalid_indices,:])
                normal_vec[invalid_indices,:] = C / torch.linalg.norm(C,dim=1)

        # Compute the valid normal_vectors
        normal_vec[valid_indices] = torch.cross(v1[valid_indices], v2.unsqueeze(0).repeat(v1.shape[0],1)[valid_indices]) / torch.linalg.norm(torch.linalg.cross(v1[valid_indices], v2.unsqueeze(0).repeat(v1.shape[0],1)[valid_indices]).unsqueeze(2),dim=1)
        return normal_vec


    @staticmethod
    def calc_rayDir(ray):
        '''
        Allows to the calculations to be done in ray-space coordinates
        as oppossed to laboratory coordinates
        Parameters:
            ray (np.array): normalized 3D vector giving the direction 
                            of the light ray
        Returns:
            ray (np.array): same as input
            ray_perp1 (np.array): normalized 3D vector
            ray_perp2 (np.array): normalized 3D vector
        '''
        # ray = ray / np.linalg.norm(ray)    # in case ray is not a unit vector <- does not need to be normalized
        theta = np.arccos(np.dot(ray, np.array([1,0,0])))
        # Unit vectors that give the laboratory axes, can be changed
        scope_axis = np.array([1,0,0])
        scope_perp1 = np.array([0,1,0])
        scope_perp2 = np.array([0,0,1])
        theta = np.arccos(np.dot(ray, scope_axis))
        # print(f"Rotating by {np.around(np.rad2deg(theta), decimals=0)} degrees")
        normal_vec = RayTraceLFM.find_orthogonal_vec(ray, scope_axis)
        Rinv = RayTraceLFM.rotation_matrix(normal_vec, -theta)
        # Extracting basis vectors that are orthogonal to the ray and will be parallel
        # to the laboratory axes that are not the optic axis after a rotation.
        # Note: If scope_perp1 if the y-axis, then ray_perp1 if the 2nd column of Rinv.
        ray_perp1 = np.dot(Rinv, scope_perp1)
        ray_perp2 = np.dot(Rinv, scope_perp2)
        return [ray, ray_perp1, ray_perp2]

    @staticmethod
    def calc_rayDir_torch(ray_in):
        '''
        Allows to the calculations to be done in ray-space coordinates
        as oppossed to laboratory coordinates
        Parameters:
            ray_in [n_rays,3] (torch.array): normalized 3D vector giving the direction 
                            of the light ray
        Returns:
            ray (torch.array): same as input
            ray_perp1 (torch.array): normalized 3D vector
            ray_perp2 (torch.array): normalized 3D vector
        '''
        if not torch.is_tensor(ray_in):
            ray = torch.from_numpy(ray_in)
        else:
            ray = ray_in
        theta = torch.arccos(torch.linalg.multi_dot((ray, torch.tensor([1.0,0,0] ,dtype=ray.dtype, device=ray_in.device))))
        # Unit vectors that give the laboratory axes, can be changed
        scope_axis = torch.tensor([1.0,0,0],dtype=ray.dtype, device=ray_in.device)
        scope_perp1 = torch.tensor([0,1.0,0],dtype=ray.dtype, device=ray_in.device)
        scope_perp2 = torch.tensor([0,0,1.0],dtype=ray.dtype, device=ray_in.device)
        # print(f"Rotating by {np.around(torch.rad2deg(theta).numpy(), decimals=0)} degrees")
        normal_vec = RayTraceLFM.find_orthogonal_vec_torch(ray, scope_axis)
        Rinv = RayTraceLFM.rotation_matrix_torch(normal_vec, -theta)
        # Extracting basis vectors that are orthogonal to the ray and will be parallel
        # to the laboratory axes that are not the optic axis after a rotation.
        # Note: If scope_perp1 if the y-axis, then ray_perp1 if the 2nd column of Rinv.
        if scope_perp1[0]==0 and scope_perp1[1]==1 and scope_perp1[2]==0:
            ray_perp1 = Rinv[:,:,1] # dot product needed
        else: 
            # todo: we need to put a for loop to do this operation
            # ray_perp1 = torch.linalg.multi_dot((Rinv, scope_perp1))
            raise NotImplementedError
        if scope_perp2[0]==0 and scope_perp2[1]==0 and scope_perp2[2]==1:
            ray_perp2 = Rinv[:,:,2]
        else: 
            # todo: we need to put a for loop to do this operation
            # ray_perp2 = torch.linalg.multi_dot((Rinv, scope_perp2))
            raise NotImplementedError
        
        # Returns a list size 3, where each element is a torch tensor shaped [n_rays, 3]
        return torch.cat([ray.unsqueeze(0), ray_perp1.unsqueeze(0), ray_perp2.unsqueeze(0)], 0)
    

###########################################################################################
    # Ray-tracing functions

    @staticmethod
    def rays_through_vol(pixels_per_ml, naObj, nMedium, volume_ctr_um):
        '''Identifies the rays that pass through the volume and the central lenslet
        Parameters:
            pixels_per_ml (int): number of pixels per microlens in one direction,
                                    preferrable to be odd integer so there is a central
                                    lenslet
            naObj (float): numerical aperture of the objective lens
            nMedium (float): refractive index of the volume
            volume_ctr_um (np.array): 3D vector containing the coordinates of the center of the
                                volume in volume space units (um)
        Returns:
            ray_enter (np.array): (3, X, X) array where (3, i, j) gives the coordinates 
                                    within the volume ray entrance plane for which the 
                                    ray that is incident on the (i, j) pixel with intersect
            ray_exit (np.array): (3, X, X) array where (3, i, j) gives the coordinates 
                                    within the volume ray exit plane for which the 
                                    ray that is incident on the (i, j) pixel with intersect
            ray_diff (np.array): (3, X, X) array giving the direction of the rays through 
                                    the volume
        
        '''
        # Units are in pixel indicies, referring to the pixel that is centered up 0.5 units
        #   Ex: if ml_ctr = [8, 8], then the spatial center pixel is at [8.5, 8.5]
        ml_ctr = [(pixels_per_ml - 1)/ 2, (pixels_per_ml - 1)/ 2]
        ml_radius = 7.5 # pixels_per_ml / 2
        i = np.linspace(0, pixels_per_ml - 1, pixels_per_ml)
        j = np.linspace(0, pixels_per_ml - 1, pixels_per_ml)
        jv, iv = np.meshgrid(i, j)
        dist_from_ctr = np.sqrt((iv - ml_ctr[0]) ** 2 + (jv - ml_ctr[1]) ** 2)

        # Angles that reach the pixels
        cam_pixels_azim = np.arctan2(jv - ml_ctr[1], iv - ml_ctr[0])
        cam_pixels_azim[dist_from_ctr > ml_radius] = np.NaN
        dist_from_ctr[dist_from_ctr > ml_radius] = np.NaN #
        cam_pixels_tilt = np.arcsin(dist_from_ctr / ml_radius * naObj / nMedium)

        # Positions of the ray in volume coordinates
        # assuming rays pass through the center voxel
        ray_enter_x = np.zeros([pixels_per_ml, pixels_per_ml])
        ray_enter_y = volume_ctr_um[0] * np.tan(cam_pixels_tilt) * np.sin(cam_pixels_azim) + volume_ctr_um[1]
        ray_enter_z = volume_ctr_um[0] * np.tan(cam_pixels_tilt) * np.cos(cam_pixels_azim) + volume_ctr_um[2]
        ray_enter_x[np.isnan(ray_enter_y)] = np.NaN
        ray_enter = np.array([ray_enter_x, ray_enter_y, ray_enter_z])
        vol_ctr_grid_tmp = np.array([np.full((pixels_per_ml, pixels_per_ml), volume_ctr_um[i]) for i in range(3)])
        ray_exit = ray_enter + 2 * (vol_ctr_grid_tmp - ray_enter)

        # Direction of the rays at the exit plane
        ray_diff = ray_exit - ray_enter
        ray_diff = ray_diff / np.linalg.norm(ray_diff, axis=0)
        return ray_enter, ray_exit, ray_diff



    def compute_rays_geometry(self, filename=None):
        '''Computes the ray-voxel collision based on the Siddon algorithm. 
        Requires: 
            calling self.rays_through_volumes to compute ray entry, exit and directions.
        Parameters:
            filename (str) optional: Saves the geometry to a pickle file, and loads the geometry
                                    from a file if the file exists.
        Returns:
            None
        Computes:
            self.vox_ctr_idx ([3]):     3D index of the central voxel.
            self.volume_ctr_um ([3]):   3D coordinate in um of the central voxel
            self.ray_valid_indices (list of tuples n_rays*[(i,j),]):
                                        Store the 2D ray index of a valid ray (without nan in entry/exit)
            self.ray_vol_colli_indices (list of list of tuples n_valid_rays*[(z,y,x),(z,y,x)]):
                                        Stores the coordinates of the voxels that the ray n collides with.
            self.ray_vol_colli_lengths (list of list of floats n_valid_rays*[ell1,ell2]):
                                        Stores the length of traversal of ray n through the voxels inside ray_vol_colli_indices.
            self.ray_valid_direction  (list [n_valid_rays, 3]):
                                        Stores the direction of ray n.        
        '''

        # If a filename is provided, check if it exists and load the whole ray tracer class from it.
        if filename is not None and exists(filename):
            data = self.unpickle(filename)
            print(f'Loaded RayTraceLFM object from {filename}')
            return data

        # todo: We treat differently numpy and torch rays, as some rays go outside the volume of interest.
        # We need to revisit this when we start computing images with more than one micro-lens in numpy
        # if False:#self.back_end == BackEnds.NUMPY:
        #     # The valid workspace is defined by the number of micro-lenses
        #     valid_vol_shape = self.optical_info['volume_shape'][1]
        # elif self.back_end == BackEnds.PYTORCH:
        valid_vol_shape = self.optical_info['n_micro_lenses']
        


        # Fetch needed variables
        pixels_per_ml = self.optical_info['pixels_per_ml']
        naObj = self.optical_info['na_obj']
        nMedium = self.optical_info['n_medium']
        vol_shape = [self.optical_info['volume_shape'][0],] + 2*[valid_vol_shape]
        voxel_size_um = self.optical_info['voxel_size_um']
        vox_ctr_idx = np.array([vol_shape[0] / 2, vol_shape[1] / 2, vol_shape[2] / 2]) # in index units
        self.vox_ctr_idx = vox_ctr_idx.astype(int)
        self.volume_ctr_um = vox_ctr_idx * voxel_size_um # in vol units (um)
        
        # Calculate the ray geometry 
        ray_enter, ray_exit, ray_diff = RayTraceLFM.rays_through_vol(pixels_per_ml, naObj, nMedium, self.volume_ctr_um)

        # Store locally
        self.ray_entry = torch.from_numpy(ray_enter).float()        if self.back_end == BackEnds.PYTORCH else ray_enter
        self.ray_exit = torch.from_numpy(ray_exit).float()          if self.back_end == BackEnds.PYTORCH else ray_exit
        self.ray_direction = torch.from_numpy(ray_diff).float()     if self.back_end == BackEnds.PYTORCH else ray_diff
        self.voxel_span_per_ml = 0


        # Pre-comute things for torch and store in tensors
        i_range,j_range = self.ray_entry.shape[1:]

        # Compute Siddon's algorithm for each ray
        ray_valid_indices = []
        ray_valid_direction = []
        ray_vol_colli_indices = []
        ray_vol_colli_lengths = []

        # Iterate rays
        for ii in range(i_range):
            for jj in range(j_range):
                # Fetch ray information
                start = ray_enter[:,ii,jj]
                stop = ray_exit[:,ii,jj]

                # We only store the valid rays
                if np.any(np.isnan(start)) or np.any(np.isnan(stop)):
                    if self.back_end == BackEnds.PYTORCH:
                        continue
                    siddon_list = []
                    voxels_of_segs = []
                    seg_mids = []
                    voxel_intersection_lengths = []

                else:
                    siddon_list = siddon_params(start, stop, voxel_size_um, vol_shape)
                    seg_mids = siddon_midpoints(start, stop, siddon_list)
                    voxels_of_segs = vox_indices(seg_mids, voxel_size_um)
                    voxel_intersection_lengths = siddon_lengths(start, stop, siddon_list)

                # Store in a temporary list
                ray_valid_indices.append((ii,jj))
                ray_vol_colli_indices.append(voxels_of_segs)
                ray_vol_colli_lengths.append(voxel_intersection_lengths)
                ray_valid_direction.append(self.ray_direction[:,ii,jj])

                # What is the maximum span of the rays of a micro lens?
                self.voxel_span_per_ml = max([self.voxel_span_per_ml,] + [vx[1] for vx in ray_vol_colli_indices[0]])

            
        # Maximum number of ray-voxel interactions, to define 
        max_ray_voxels_collision = np.max([len(D) for D in ray_vol_colli_indices])
        n_valid_rays = len(ray_valid_indices)

        # Create the information to store
        self.ray_valid_indices = ray_valid_indices
        # Store as tuples for now
        self.ray_vol_colli_indices = ray_vol_colli_indices
        if self.back_end == BackEnds.NUMPY:
            self.ray_vol_colli_lengths = np.zeros([n_valid_rays, max_ray_voxels_collision])
            self.ray_valid_direction = np.zeros([n_valid_rays, 3])
        elif self.back_end == BackEnds.PYTORCH:
            # Save as nn.Parameters so Pytorch can handle them correctly, for things like moving this whole class to GPU
            self.ray_vol_colli_lengths = nn.Parameter(torch.zeros(n_valid_rays, max_ray_voxels_collision)) 
            self.ray_vol_colli_lengths.requires_grad = False
            self.ray_valid_direction = nn.Parameter(torch.zeros(n_valid_rays, 3))
            self.ray_valid_direction.requires_grad = False


        # Filter out rays that aren't valid (contain nan)
        # todo: indices is indices 
        for valid_ray in range(n_valid_rays):
            # Fetch the ray-voxel intersection length for this ray
            val_lengths = ray_vol_colli_lengths[valid_ray]
            self.ray_vol_colli_lengths[valid_ray, :len(val_lengths)] = torch.tensor(val_lengths)    if self.back_end == BackEnds.PYTORCH else val_lengths
            self.ray_valid_direction[valid_ray, :] = ray_valid_direction[valid_ray]
        
        # Update volume shape information, to account for the whole workspace
        # todo: mainly for pytorch multi-lenslet computation
        vol_shape = self.optical_info['volume_shape']
        vox_ctr_idx = np.array([vol_shape[0] / 2, vol_shape[1] / 2, vol_shape[2] / 2]) # in index units
        self.vox_ctr_idx = vox_ctr_idx.astype(int)
        self.volume_ctr_um = vox_ctr_idx * voxel_size_um

        if filename is not None:
            self.pickle(filename)
            print(f'Saved RayTraceLFM object from {filename}')
    
        # Calculate the ray's direction with the two normalized perpendicular directions
        # Returns a list size 3, where each element is a torch tensor shaped [n_rays, 3]
        if self.back_end == BackEnds.NUMPY:
            self.ray_direction_basis = []
            for n_ray,ray in enumerate(self.ray_valid_direction):
                self.ray_direction_basis.append(RayTraceLFM.calc_rayDir(ray))
        elif self.back_end == BackEnds.PYTORCH:
            self.ray_direction_basis = RayTraceLFM.calc_rayDir_torch(self.ray_valid_direction)

        return self


    # Helper functions to load/save the whole class to disk
    def pickle(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    
    @staticmethod
    def unpickle(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


    ######## Not implemented: These functions need an implementation in derived objects
    def ray_trace_through_volume(self, volume_in):
        ''' We have a separate function as we have some basic functionality that is shared'''
        raise NotImplementedError
    
    def init_volume(self, volume_in):
        ''' This function assigns a volume the correct internal structure for a given simul_type
        For example: a single value per voxel for fluorescence, or two values for birefringence'''
        raise NotImplementedError


    
        