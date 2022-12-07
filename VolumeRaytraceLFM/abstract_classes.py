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
    FLUOR_INTENS    = 1     # Voxels add intensity as the ray goes trough the volume: commutative
    BIREFRINGENT    = 2     # Voxels modify polarization as the ray goes trough: non commutative
    # FLUOR_POLAR     = 3     # 
    # DIPOLES         = 4     # Voxels add light depending with their angle with respect to the dipole
    # etc. attenuators and di-attenuators (polarization dependent)


class BackEnds(Enum):
    ''' Defines type of backend (numpy,pytorch,etc)'''
    NUMPY       = 1     # Use numpy back-end
    PYTORCH     = 2     # Use Pytorch, with auto-differentiation and GPU support.



class AnisotropicOpticalElement(OpticBlock):
    ''' Abstract class defining a birefringent object'''
    def __init__(self, back_end : BackEnds = BackEnds.NUMPY, torch_args={'optic_config' : None, 'members_to_learn' : []},
                system_info={'volume_shape' : 3*[1], 'voxel_size_um' : 3*[1.0], 'pixels_per_ml' : 17, 'na_obj' : 1.2, 
                'n_medium' : 1.52, 'wavelength' : 0.550}):
        # Check if back-end is torch and overwrite self with an optic block, for Waveblocks compatibility.
        if back_end==BackEnds.PYTORCH:
            super(AnisotropicOpticalElement, self).__init__(optic_config=torch_args['optic_config'], 
                    members_to_learn=torch_args['members_to_learn'] if 'members_to_learn' in torch_args.keys() else [])
        self.back_end = back_end
        self.simul_type = SimulType.BIREFRINGENT
        self.system_info = system_info

        # if we are using pytorch and waveblocks, grab system info from optic_config
        if self.back_end == BackEnds.PYTORCH:
            self.system_info = \
                    {'volume_shape' : self.optic_config.volume_config.volume_shape, 
                    'voxel_size_um' : self.optic_config.volume_config.voxel_size_um, 
                    'pixels_per_ml' : self.optic_config.mla_config.n_pixels_per_mla, 
                    'na_obj' : self.optic_config.PSF_config.NA, 
                    'n_medium' : self.optic_config.PSF_config.ni,
                    'wavelength' : self.optic_config.PSF_config.wvl}




###########################################################################################
    # Constructors for different types of elements
    # This methods are constructors only, they don't support torch optimization of internal variables
    # todo: rename such that it is clear that these are presets for different birefringent objects

    @staticmethod
    def rotator(angle, back_end=BackEnds.NUMPY):
        '''2D rotation matrix
        Args:
            angle: angle to rotate by counterclockwise [radians]
        Return: Jones matrix'''
        if back_end == BackEnds.NUMPY:
            s = np.sin(angle)
            c = np.cos(angle)
            R = np.array([[c, -s], [s, c]])
        elif back_end == BackEnds.PYTORCH:
            s = torch.sin(angle)
            c = torch.cos(angle)
            R = torch.tensor([[c, -s], [s, c]])
        return R

    @staticmethod
    def LR(ret, azim, back_end=BackEnds.NUMPY):
        '''Linear retarder
        Args:
            ret (float): retardance [radians]
            azim (float): azimuth angle of fast axis [radians]
        Return: Jones matrix    
        '''
        retardor_azim0 = AnisotropicOpticalElement.LR_azim0(ret)
        R = AnisotropicOpticalElement.rotator(azim)
        Rinv = AnisotropicOpticalElement.rotator(-azim)
        return R @ retardor_azim0 @ Rinv

    @staticmethod
    def LR_azim0(ret, back_end=BackEnds.NUMPY):
        '''todo'''
        if back_end == BackEnds.NUMPY:
            return np.array([[np.exp(1j * ret / 2), 0], [0, np.exp(-1j * ret / 2)]])
        else:
            return torch.tensor([torch.exp(1j * ret / 2), 0], [0, torch.exp(-1j * ret / 2)])

    @staticmethod
    def LR_azim90(ret, back_end=BackEnds.NUMPY):
        '''todo'''
        if back_end == BackEnds.NUMPY:
            return np.array([[np.exp(-1j * ret / 2), 0], [0, np.exp(1j * ret / 2)]])
        else:
            return torch.tensor([torch.exp(-1j * ret / 2), 0], [0, torch.exp(1j * ret / 2)])

    @staticmethod
    def QWP(azim):
        '''Quarter Waveplate
        Linear retarder with lambda/4 or equiv pi/2 radians
        Commonly used to convert linear polarized light to circularly polarized light'''
        ret = np.pi / 2
        return AnisotropicOpticalElement.LR(ret, azim)

    @staticmethod
    def HWP(azim):
        '''Half Waveplate
        Linear retarder with lambda/2 or equiv pi radians
        Commonly used to rotate the plane of linear polarization'''
        # Faster method
        s = np.sin(2 * azim)
        c = np.cos(2 * azim)
        # # Alternative method
        # ret = np.pi
        # JM = self.LR(ret, azim)
        return np.array([[c, s], [s, -c]])

    @staticmethod
    def LP(theta):
        '''Linear Polarizer
        Args:
            theta: angle that light can pass through
        Returns: Jones matrix
        '''
        c = np.cos(theta)
        s = np.sin(theta)
        J00 = c ** 2
        J11 = s ** 2
        J01 = s * c
        J10 = J01
        return np.array([[J00, J01], [J10, J11]])
    
    @staticmethod
    def RCP():
        '''Right Circular Polarizer'''
        return 1 / 2 * np.array([[1, -1j], [1j, 1]])

    @staticmethod
    def LCP():
        '''Left Circular Polarizer'''
        return 1 / 2 * np.array([[1, 1j], [-1j, 1]])
    @staticmethod
    def RCR(ret):
        '''Right Circular Retarder'''
        return AnisotropicOpticalElement.rotator(-ret / 2)
    @staticmethod
    def LCR(ret):
        '''Left Circular Retarder'''
        return AnisotropicOpticalElement.rotator(ret / 2)





###########################################################################################
# Implementations of AnisotropicOpticalElement

class AnisotropicVoxel(AnisotropicOpticalElement):
    '''This class stores a 3D array of voxels with birefringence properties, either with a numpy or pytorch back-end.'''
    def __init__(self, back_end=BackEnds.NUMPY, torch_args={'optic_config' : None, 'members_to_learn' : []}, 
        system_info={'volume_shape' : [11,11,11], 'voxel_size_um' : 3*[1.0], 'pixels_per_ml' : 17, 'na_obj' : 1.2, 'n_medium' : 1.52, 'wavelength' : 0.550},
        Delta_n=0, optic_axis=[1, 0, 0]):
        '''AnisotropicVoxel
        Args:
            back_end (BackEnd):     A computation BackEnd (Numpy vs Pytorch). If Pytorch is used, torch_args are required
                                    to initialize the head class OpticBlock from Waveblocks.
            torch_args (dic):       Required for PYTORCH back-end. Contains optic_config object and members_to_learn
            system_info (dic):
                                    volume_shape ([3]:[sz,sy,sz]):
                                                            Shape of the volume in voxel numbers per dimension.
                                    voxel_size_um ([3]):    Size of a voxel in micrometers.
                                    pixels_per_ml (int):    Number of pixels covered by a micro-lens in a light-field system
                                    na_obj (float):         Numerical aperture of the objective.
                                    n_medium (float):       Refractive index of immersion medium.
                                    wavelength (float):     Wavelength of light used.
            Delta_n (float or [sz,sy,sz] array):        
                                    Defines the birefringence magnitude of the voxels.
                                    If a float is passed, all the voxels will have the same Delta_n.
            optic_axis ([3] or [3,sz,sy,sz]:             
                                    Defines the optic axis per voxel.
                                    If a single 3D vector is passed all the voxels will share this optic axis.
            '''
        super(AnisotropicVoxel, self).__init__(back_end=back_end, torch_args=torch_args, system_info=system_info)
       

        if self.back_end == BackEnds.NUMPY:
            self.volume_shape = self.system_info['volume_shape']
            # In the case when an optic axis per voxel of a 3D volume is provided
            # e.g. [3,nz,ny,nx]
            if isinstance(optic_axis, np.ndarray) and len(optic_axis.shape) == 4:
                self.volume_shape = optic_axis.shape[1:]
                # flatten all the voxels in order to normalize them
                optic_axis = optic_axis.reshape(3, self.volume_shape[0]* self.volume_shape[1]* self.volume_shape[2])
                for n_voxel in range(len(optic_axis[0,...])):
                    optic_axis[:,n_voxel] /= np.linalg.norm(optic_axis[:,n_voxel]) 
                # Set 4D shape again 
                self.optic_axis = optic_axis.reshape(3, *self.volume_shape)

                self.Delta_n = Delta_n
                assert len(self.Delta_n.shape) == 3, '3D Delta_n expected, as the optic_axis was provided as a 3D array'

            # Single optical axis, we replicate it for all the voxels
            elif isinstance(optic_axis, list) or isinstance(optic_axis, np.ndarray):
                # Same optic axis for all voxels
                optic_axis = np.array(optic_axis)
                norm = np.linalg.norm(optic_axis)
                if norm != 0:
                    optic_axis /= norm
                self.optic_axis = np.expand_dims(optic_axis,[1,2,3]).repeat(self.volume_shape[0],1).repeat(self.volume_shape[1],2).repeat(self.volume_shape[2],3)
            
                # Create Delta_n 3D volume
                self.Delta_n = Delta_n * np.ones(self.volume_shape)

                
        elif self.back_end == BackEnds.PYTORCH:
            # Update volume shape from optic config 
            self.volume_shape = torch_args['optic_config'].volume_config.volume_shape
            # Normalization of optical axis, depending on input
            if torch.is_tensor(optic_axis) and len(optic_axis.shape) == 4:
                norm_A = (optic_axis[0,...]**2+optic_axis[1,...]**2+optic_axis[2,...]**2).sqrt()
                self.optic_axis = nn.Parameter(optic_axis / norm_A.repeat(3,1,1,1))
                assert len(Delta_n.shape) == 3, '3D Delta_n expected, as the optic_axis was provided as a 3D torch tensor'
                self.Delta_n = Delta_n
            else:
                # Same optic axis for all voxels
                optic_axis = np.array(optic_axis).astype(np.float32)
                norm = np.linalg.norm(optic_axis)
                if norm != 0:
                    optic_axis /= norm
                self.optic_axis = torch.from_numpy(optic_axis).unsqueeze(1).unsqueeze(1).unsqueeze(1) \
                                    .repeat(1, self.volume_shape[0], self.volume_shape[1], self.volume_shape[2])
                
                # Store the data as pytorch parameters
                self.optic_axis = nn.Parameter(self.optic_axis)
                self.Delta_n = nn.Parameter(Delta_n * torch.ones(self.volume_shape))


        

    ###########################################################################################
    # Methods necessary for determining the Jones matrix of a birefringent material
    # maybe this section should be a subclass of JonesMatrix
    def calc_retardance(self, ray_dir, thickness):
        if self.back_end==BackEnds.NUMPY:
            ret = abs(self.Delta_n) * (1 - np.dot(self.optic_axis, ray_dir) ** 2) * 2 * np.pi * thickness / self.system_info['wavelength']
        elif self.back_end==BackEnds.PYTORCH:
            ret = abs(self.Delta_n) * (1 - torch.linalg.vecdot(self.optic_axis, ray_dir) ** 2) * 2 * torch.pi * thickness / self.system_info['wavelength']
        else:
            raise NotImplementedError
        # print(f"Accumulated retardance from index ellipsoid is {np.around(np.rad2deg(ret), decimals=0)} ~ {int(np.rad2deg(ret)) % 360} degrees.")
        return ret


    def calc_azimuth(self, ray_dir_basis=[]):
        if self.back_end==BackEnds.NUMPY:
            azim = np.arctan2(np.dot(self.optic_axis, ray_dir_basis[1]), np.dot(self.optic_axis, ray_dir_basis[2]))
            if self.Delta_n == 0:
                azim = 0
            elif self.Delta_n < 0:
                azim = azim + np.pi / 2
        elif self.back_end==BackEnds.PYTORCH:
            azim = torch.arctan2(torch.linalg.vecdot(self.optic_axis , ray_dir_basis[1]), torch.linalg.vecdot(self.optic_axis , ray_dir_basis[2])) 
            azim[self.Delta_n==0] = 0
            azim[self.Delta_n<0] += torch.pi / 2
        else:
            raise NotImplementedError
        
        # print(f"Azimuth angle of index ellipsoid is {np.around(np.rad2deg(azim), decimals=0)} degrees.")
        return azim
    
    def LR_material(self):
        ret = self.calc_retardance()
        azim = self.calc_azimuth()
        return self.LR(ret, azim)


    def plot_volume_plotly(self, voxels=None, opacity=0.5):
        
        if voxels is None:
            voxels = self.voxel_parameters[0,...].clone().cpu().detach()

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

###########################################################################################
class RayTraceLFM(AnisotropicOpticalElement):
    '''This is a base class that takes a volume geometry and LFM geometry and calculates which arrive to each of the pixels behind each micro-lense, and discards the rest.
       This class also pre-computes how each rays traverses the volume with the Siddon algorithm.
       The interaction between the voxels and the rays is defined by each specialization of this class.'''

    def __init__(
        self, back_end : BackEnds = BackEnds.NUMPY, torch_args={'optic_config' : None, 'members_to_learn' : []}, simul_type : SimulType = SimulType.BIREFRINGENT,
            system_info={'volume_shape' : [11,11,11], 'voxel_size_um' : 3*[1.0], 'pixels_per_ml' : 17, 'na_obj' : 1.2, 'n_medium' : 1.52, 'wavelength' : 0.550}):
        super(RayTraceLFM, self).__init__(back_end=back_end, torch_args=torch_args, system_info=system_info)
        
        # Store system information
        self.simul_type = simul_type
        
        # Create dummy variables for pre-computed rays and paths through the volume
        # This are defined in compute_rays_geometry
        self.ray_valid_indices = None
        self.ray_vol_colli_indices = None
        self.ray_vol_colli_lengths = None
        self.ray_direction_basis = None

    def forward(self, volume_in : AnisotropicVoxel=None):
        # Check if type of volume is the same as input volume, if one is provided
        if volume_in is not None:
            assert volume_in.simul_type == self.simul_type, f"Error: wrong type of volume provided, this \
            ray-tracer works for {self.simul_type} and a volume {volume_in.simul_type} was provided"
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
        if self.back_end == BackEnds.NUMPY:
            # The valid workspace is defined by the number of micro-lenses
            valid_vol_shape = self.system_info['volume_shape'][1]
        elif self.back_end == BackEnds.PYTORCH:
            valid_vol_shape = self.optic_config.mla_config.n_micro_lenses
        


        # Fetch needed variables
        pixels_per_ml = self.system_info['pixels_per_ml']
        naObj = self.system_info['na_obj']
        nMedium = self.system_info['n_medium']
        vol_shape = [self.system_info['volume_shape'][0],] + 2*[valid_vol_shape]
        voxel_size_um = self.system_info['voxel_size_um']
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
        vol_shape = self.system_info['volume_shape']
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
    def ray_trace_through_volume(self, volume_in : AnisotropicVoxel=None):
        ''' We have a separate function as we have some basic functionality that is shared'''
        raise NotImplementedError
    
    def init_volume(self, volume_in : AnisotropicVoxel=None):
        ''' This function assigns a volume the correct internal structure for a given simul_type
        For example: a single value per voxel for fluorescence, or two values for birefringence'''
        raise NotImplementedError


    
        