from VolumeRaytraceLFM.abstract_classes import *
from tqdm import tqdm

class BirefringentElement(OpticalElement):
    ''' Birefringent element, such as voxel, raytracer, etc, extending optical element, so it has a back-end and optical information'''
    def __init__(self, back_end : BackEnds = BackEnds.NUMPY, torch_args={},#{'optic_config' : None, 'members_to_learn' : []},
                optical_info={'volume_shape' : 3*[1], 'voxel_size_um' : 3*[1.0], 'pixels_per_ml' : 17, 'na_obj' : 1.2, 
                'n_medium' : 1.52, 'wavelength' : 0.550, 'n_micro_lenses' : 1, 'n_voxels_per_ml' : 1}):
        super(BirefringentElement, self).__init__(back_end=back_end, torch_args=torch_args, optical_info=optical_info)

        self.simul_type = SimulType.BIREFRINGENT


###########################################################################################
# Implementations of OpticalElement
# todo: rename to BirefringentVolume inherits 
class BirefringentVolume(BirefringentElement):
    '''This class stores a 3D array of voxels with birefringence properties, either with a numpy or pytorch back-end.'''
    def __init__(self, back_end=BackEnds.NUMPY, torch_args={},#{'optic_config' : None, 'members_to_learn' : []}, 
        optical_info={},#{'volume_shape' : [11,11,11], 'voxel_size_um' : 3*[1.0], 'pixels_per_ml' : 17, 'na_obj' : 1.2, 'n_medium' : 1.52, 'wavelength' : 0.550, 'n_micro_lenses' : 1},
        Delta_n=0, optic_axis=[1, 0, 0]):
        '''BirefringentVolume
        Args:
            back_end (BackEnd):     A computation BackEnd (Numpy vs Pytorch). If Pytorch is used, torch_args are required
                                    to initialize the head class OpticBlock from Waveblocks.
            torch_args (dic):       Required for PYTORCH back-end. Contains optic_config object and members_to_learn
            optical_info (dic):
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
        super(BirefringentVolume, self).__init__(back_end=back_end, torch_args=torch_args, optical_info=optical_info)
       

        if self.back_end == BackEnds.NUMPY:
            self.volume_shape = self.optical_info['volume_shape']
            # In the case when an optic axis per voxel of a 3D volume is provided
            # e.g. [3,nz,ny,nx]
            if isinstance(optic_axis, np.ndarray) and len(optic_axis.shape) == 4:
                self.volume_shape = optic_axis.shape[1:]
                # flatten all the voxels in order to normalize them
                optic_axis = optic_axis.reshape(3, self.volume_shape[0]* self.volume_shape[1]* self.volume_shape[2])
                for n_voxel in range(len(optic_axis[0,...])):
                    oa_norm = np.linalg.norm(optic_axis[:,n_voxel]) 
                    if oa_norm > 0:
                        optic_axis[:,n_voxel] /= oa_norm
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

            self.Delta_n[np.isnan(self.Delta_n)] = 0
            self.optic_axis[np.isnan(self.optic_axis)] = 0
                
        elif self.back_end == BackEnds.PYTORCH:
            # Update volume shape from optic config 
            self.volume_shape = self.optical_info['volume_shape']
            # Normalization of optical axis, depending on input
            if not isinstance(optic_axis, list) and optic_axis.ndim==4:
                if isinstance(optic_axis, np.ndarray):
                    optic_axis = torch.from_numpy(optic_axis).type(torch.get_default_dtype())
                    
                norm_A = (optic_axis[0,...]**2+optic_axis[1,...]**2+optic_axis[2,...]**2).sqrt()
                self.optic_axis = optic_axis / norm_A.repeat(3,1,1,1)
                assert len(Delta_n.shape) == 3, '3D Delta_n expected, as the optic_axis was provided as a 3D torch tensor'
                self.Delta_n = Delta_n
                if not torch.is_tensor(Delta_n):
                    self.Delta_n = torch.from_numpy(Delta_n).type(torch.get_default_dtype())
                
            else:
                # Same optic axis for all voxels
                optic_axis = np.array(optic_axis).astype(np.float32)
                norm = np.linalg.norm(optic_axis)
                if norm != 0:
                    optic_axis /= norm
                self.optic_axis = torch.from_numpy(optic_axis).unsqueeze(1).unsqueeze(1).unsqueeze(1) \
                                    .repeat(1, self.volume_shape[0], self.volume_shape[1], self.volume_shape[2])

                self.Delta_n = Delta_n * torch.ones(self.volume_shape)

            # Check for not a number, for when the voxel optic_axis is all zeros
            self.Delta_n[torch.isnan(self.Delta_n)] = 0
            self.optic_axis[torch.isnan(self.optic_axis)] = 0
            # Store the data as pytorch parameters
            self.optic_axis = nn.Parameter(self.optic_axis).type(torch.get_default_dtype())
            self.Delta_n = nn.Parameter(self.Delta_n ).type(torch.get_default_dtype())

    @staticmethod
    def plot_volume_plotly(optical_info, voxels_in=None, opacity=0.5, colormap='gray'):
        
        voxels = voxels_in - voxels_in.min()
        import plotly.graph_objects as go
        import numpy as np
        volume_shape = optical_info['volume_shape']
        volume_size_um = [optical_info['voxel_size_um'][i] * optical_info['volume_shape'][i] for i in range(3)]
        [dz, dxy, dxy] = optical_info['voxel_size_um']
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
            value=voxels.flatten() / voxels.max(),
            isomin=0,
            isomax=0.1,
            opacity=opacity, # needs to be small to see through all surfaces
            surface_count=20, # needs to be a large number for good volume rendering
            # colorscale=colormap
            ))
        # fig.data = fig.data[::-1]
        # Draw the whole volume span
        # fig.add_mesh3d(
        #         # 8 vertices of a cube
        #         x=[0, 0, volume_size_um[0], volume_size_um[0], 0, 0, volume_size_um[0], volume_size_um[0]],
        #         y=[0, volume_size_um[1], volume_size_um[1], 0, 0, volume_size_um[1], volume_size_um[1], 0],
        #         z=[0, 0, 0, 0, volume_size_um[2], volume_size_um[2], volume_size_um[2], volume_size_um[2]],
        #         colorbar_title='z',
        #         colorscale='inferno',
        #         opacity=0.001,
        #         # Intensity of each vertex, which will be interpolated and color-coded
        #         intensity = np.linspace(0, 1, 8, endpoint=True),
        #         # i, j and k give the vertices of triangles
        #         i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        #         j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        #         k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        #     )
        fig.update_layout(
            scene = dict(
                        xaxis = dict(nticks=volume_shape[0], range=[0, volume_size_um[0]]),
                        yaxis = dict(nticks=volume_shape[1], range=[0, volume_size_um[1]]),
                        zaxis = dict(nticks=volume_shape[2], range=[0, volume_size_um[2]]),
                        xaxis_title='Axial dimension',
                        aspectratio = dict( x=volume_size_um[0], y=volume_size_um[1], z=volume_size_um[2] ), aspectmode = 'manual'),
            # width=700,
            margin=dict(r=0, l=0, b=0, t=0),
            autosize=True
            )
        fig.show()
        return

    def get_vox_params(self, vox_index):
        '''vox_index is a tuple'''
        return self.Delta_n[vox_index], self.optic_axis[vox_index]


############ Implementations
class BirefringentRaytraceLFM(RayTraceLFM, BirefringentElement):
    """This class extends RayTraceLFM, and implements the forward function, where voxels contribute to ray's Jones-matrices with a retardance and axis in a non-commutative matter"""
    def __init__(
            self, back_end : BackEnds = BackEnds.NUMPY, torch_args={},#{'optic_config' : None, 'members_to_learn' : []},
            optical_info={}):#{'volume_shape' : [11,11,11], 'voxel_size_um' : 3*[1.0], 'pixels_per_ml' : 17, 'na_obj' : 1.2, 'n_medium' : 1.52, 'wavelength' : 0.550, 'n_micro_lenses' : 1}):
        # optic_config contains mla_config and volume_config
        super(BirefringentRaytraceLFM, self).__init__(
            back_end=back_end, torch_args=torch_args, optical_info=optical_info
        )
        
        
    def ray_trace_through_volume(self, volume_in : BirefringentVolume = None):
        """ This function forward projects a whole volume, by iterating through the volume in front of each micro-lens in the system.
            By computing an offset (current_offset) that shifts the volume indices reached by each ray.
            Then we accumulate the images generated by each micro-lens, and concatenate in a final image"""

        # volume_shape defines the size of the workspace
        # the number of micro lenses defines the valid volume inside the workspace
        volume_shape = volume_in.optical_info['volume_shape']
        n_micro_lenses = self.optical_info['n_micro_lenses']
        n_voxels_per_ml = self.optical_info['n_voxels_per_ml']
        n_ml_half = floor(n_micro_lenses / 2.0)

        # Check if the volume_size can fit these micro_lenses.
        # considering that some rays go beyond the volume in front of the micro-lens
        voxel_span_per_ml = self.voxel_span_per_ml + (n_micro_lenses*n_voxels_per_ml) + 1
        assert voxel_span_per_ml < volume_shape[1] and voxel_span_per_ml < volume_shape[2], f"The volume in front of the microlenses ({n_micro_lenses},{n_micro_lenses}) is to large for a volume_shape: {self.optical_info['volume_shape'][1:]}. Increase the volume_shape to at least [{voxel_span_per_ml},{voxel_span_per_ml}]"
        # assert volume_shape[1] - 2*self.voxel_span_per_ml > 0 and volume_shape[2] - 2*self.voxel_span_per_ml > 0, f"The volume in front of the microlenses ({n_micro_lenses},{n_micro_lenses}) is to large. \
        

        # Traverse volume for every ray, and generate retardance and azimuth images
        full_img_r = None
        full_img_a = None
        # Iterate micro-lenses in y direction
        for ml_ii in tqdm(range(-n_ml_half, n_ml_half+1), f'Computing rows of micro-lens ret+azim {self.back_end}'):
            full_img_row_r = None
            full_img_row_a = None
            # Iterate micro-lenses in x direction
            for ml_jj in range(-n_ml_half, n_ml_half+1):
                current_offset = [n_voxels_per_ml * ml_ii, n_voxels_per_ml*ml_jj]
                # Compute images for current microlens, by passing an offset to this function depending on the micro lens and the super resolution
                ret_image_torch, azim_image_torch = self.ret_and_azim_images(volume_in, micro_lens_offset=current_offset)
                # If this is the first image, create
                if full_img_row_r is None:
                    full_img_row_r = ret_image_torch
                    full_img_row_a = azim_image_torch
                else: # Concatenate to existing image otherwise
                    if self.back_end == BackEnds.NUMPY:
                        full_img_row_r = np.concatenate((full_img_row_r, ret_image_torch), 0)
                        full_img_row_a = np.concatenate((full_img_row_a, azim_image_torch), 0)
                    elif self.back_end == BackEnds.PYTORCH:
                        full_img_row_r = torch.cat((full_img_row_r, ret_image_torch), 0)
                        full_img_row_a = torch.cat((full_img_row_a, azim_image_torch), 0)
            if full_img_r is None:
                full_img_r = full_img_row_r
                full_img_a = full_img_row_a
            else:
                if self.back_end == BackEnds.NUMPY:
                    full_img_r = np.concatenate((full_img_r, full_img_row_r), 1)
                    full_img_a = np.concatenate((full_img_a, full_img_row_a), 1)
                elif self.back_end == BackEnds.PYTORCH:
                    full_img_r = torch.cat((full_img_r, full_img_row_r), 1)
                    full_img_a = torch.cat((full_img_a, full_img_row_a), 1)
        return full_img_r, full_img_a
    
    def retardance(self, JM):
        '''Phase delay introduced between the fast and slow axis in a Jones Matrix'''
        if self.back_end == BackEnds.NUMPY:
            e1, e2 = np.linalg.eigvals(JM)
            phase_diff = np.angle(e1) - np.angle(e2)
            retardance = np.abs(phase_diff)
        elif self.back_end == BackEnds.PYTORCH:
            x = torch.linalg.eigvals(JM)
            retardance = (torch.angle(x[:,1]) - torch.angle(x[:,0])).abs()
        else:
            raise NotImplementedError
        return retardance

    def azimuth(self, JM):
        '''Rotation angle of the fast axis (neg phase)'''
        if self.back_end == BackEnds.NUMPY:
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
        elif self.back_end == BackEnds.PYTORCH: 
            diag_sum = (JM[:, 0, 0] + JM[:, 1, 1])
            diag_diff = (JM[:, 1, 1] - JM[: ,0, 0])
            off_diag_sum = JM[:, 0, 1] + JM[:, 1, 0]
            a = (diag_diff / diag_sum).imag
            b = (off_diag_sum / diag_sum).imag
            azimuth = torch.arctan2(-b, -a) / 2.0 + torch.pi / 2.0

            # todo: if output azimuth is pi, make it 0 and vice-versa (arctan2 bug)
            # zero_index = torch.isclose(azimuth, torch.zeros([1]), atol=1e-5)
            # pi_index = torch.isclose(azimuth, torch.tensor(torch.pi), atol=1e-5)
            # azimuth[zero_index] = torch.pi
            # azimuth[pi_index] = 0
        else:
            raise NotImplementedError
        return azimuth
    
    def calc_cummulative_JM_of_ray(self, volume_in : BirefringentVolume, micro_lens_offset=[0,0]):
        if self.back_end==BackEnds.NUMPY:
            return self.calc_cummulative_JM_of_ray_numpy(volume_in, micro_lens_offset)
        elif self.back_end==BackEnds.PYTORCH:
            return self.calc_cummulative_JM_of_ray_torch(volume_in, micro_lens_offset)

    def calc_cummulative_JM_of_ray_numpy(self, i, j, volume_in : BirefringentVolume, micro_lens_offset=[0,0]):
        '''For the (i,j) pixel behind a single microlens'''
        # Fetch precomputed Siddon parameters
        voxels_of_segs, ell_in_voxels = self.ray_vol_colli_indices, self.ray_vol_colli_lengths
        # rays are stored in a 1D array, let's look for index i,j
        n_ray = j + i *  self.optical_info['pixels_per_ml']
        rayDir = self.ray_direction_basis[n_ray][:]

        polarizer = self.optical_info['polarizer']
        analyzer = self.optical_info['analyzer']

        JM_list = []
        JM_list.append(polarizer)
        for m in range(len(voxels_of_segs[n_ray])):
            ell = ell_in_voxels[n_ray][m]
            vox = voxels_of_segs[n_ray][m]
            Delta_n = volume_in.Delta_n[vox[0], vox[1]+micro_lens_offset[0], vox[2]+micro_lens_offset[1]]
            opticAxis = volume_in.optic_axis[:, vox[0], vox[1]+micro_lens_offset[0], vox[2]+micro_lens_offset[1]]
            JM = self.voxRayJM(Delta_n, opticAxis, rayDir, ell, self.optical_info['wavelength'])
            JM_list.append(JM)
        JM_list.append(analyzer)
        effective_JM = BirefringentRaytraceLFM.rayJM_numpy(JM_list)
        return effective_JM

    def calc_cummulative_JM_of_ray_torch(self, volume_in : BirefringentVolume, micro_lens_offset=[0,0]):
        '''This function computes the Jones Matrices of all rays defined in this object.
            It uses pytorch's batch dimension to store each ray, and process them in parallel'''

        # Fetch the voxels traversed per ray and the lengths that each ray travels through every voxel
        voxels_of_segs, ell_in_voxels = self.ray_vol_colli_indices, self.ray_vol_colli_lengths
            
        # Init an array to store the Jones matrices.
        JM_list = []

        assert self.optical_info == volume_in.optical_info, 'Optical info between ray-tracer and volume mismatch. This might cause issues on the border micro-lenses.'
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
            # Birefringence 
            Delta_n = volume_in.Delta_n[[v[0] for v in vox], [v[1]+micro_lens_offset[0] for v in vox], [v[2]+micro_lens_offset[1] for v in vox]]

            # Initiallize identity Jones Matrices, shape [n_rays_with_voxels, 2, 2]
            JM = torch.tensor([[1.0,0],[0,1.0]], dtype=torch.complex64, device=self.get_device()).unsqueeze(0).repeat(n_rays_with_voxels,1,1)

            if not torch.all(Delta_n==0):
                # And axis
                opticAxis = volume_in.optic_axis[:, [v[0] for v in vox], [v[1]+micro_lens_offset[0] for v in vox], [v[2]+micro_lens_offset[1] for v in vox]]
                # If a single voxel, this would collapse
                opticAxis = opticAxis.permute(1,0)
                # Grab the subset of precomputed ray directions that have voxels in this step
                filtered_rayDir = self.ray_direction_basis[:,rays_with_voxels,:]

                # Only compute if there's an Delta_n
                # Create a mask of the valid voxels
                valid_voxel = Delta_n!=0
                if valid_voxel.sum() > 0:
                    # Compute the interaction from the rays with their corresponding voxels
                    JM[valid_voxel, :, :] = self.voxRayJM(   Delta_n = Delta_n[valid_voxel], 
                                                                                opticAxis = opticAxis[valid_voxel, :], 
                                                                                rayDir = [filtered_rayDir[0][valid_voxel], filtered_rayDir[1][valid_voxel], filtered_rayDir[2][valid_voxel]], 
                                                                                ell = ell[valid_voxel],
                                                                                wavelength=self.optical_info['wavelength'])
            else:
                pass
            # Store current interaction step
            JM_list.append(JM)
        # JM_list contains m steps of rays interacting with voxels
        # Each JM_list[m] is shaped [n_rays, 2, 2]
        # We pass voxels_of_segs to compute which rays have a voxel in each step
        material_JM = BirefringentRaytraceLFM.rayJM_torch(JM_list, voxels_of_segs)
        polarizer = torch.from_numpy(self.optical_info['polarizer']).type(torch.complex64).to(Delta_n.device)
        analyzer = torch.from_numpy(self.optical_info['analyzer']).type(torch.complex64).to(Delta_n.device)
        effective_JM = analyzer @ material_JM @ polarizer
        return effective_JM

    def ret_and_azim_images(self, volume_in : BirefringentVolume, micro_lens_offset=[0,0]):
        '''Calculate retardance and azimuth values for a ray with a Jones Matrix'''
        if self.back_end==BackEnds.NUMPY:
            return self.ret_and_azim_images_numpy(volume_in, micro_lens_offset)
        elif self.back_end==BackEnds.PYTORCH:
            return self.ret_and_azim_images_torch(volume_in, micro_lens_offset)

    def ret_and_azim_images_numpy(self, volume_in : BirefringentVolume, micro_lens_offset=[0,0]):
        '''Calculate retardance and azimuth values for a ray with a Jones Matrix'''
        n_micro_lenses = self.optical_info['n_micro_lenses']
        n_ml_half = floor(n_micro_lenses / 2.0)
        micro_lens_offset = np.array(micro_lens_offset) + np.array(self.vox_ctr_idx[1:]) - n_ml_half

        pixels_per_ml = self.optical_info['pixels_per_ml']
        ret_image = np.zeros((pixels_per_ml, pixels_per_ml))
        azim_image = np.zeros((pixels_per_ml, pixels_per_ml))
        for i in range(pixels_per_ml):
            for j in range(pixels_per_ml):
                if np.isnan(self.ray_entry[0, i, j]):
                    ret_image[i, j] = 0
                    azim_image[i, j] = 0
                else:
                    effective_JM = self.calc_cummulative_JM_of_ray_numpy(i, j, volume_in, micro_lens_offset)
                    ret_image[i, j] = self.retardance(effective_JM)
                    if np.isclose(ret_image[i, j], 0.0):
                        azim_image[i, j] = 0
                    else:
                        azim_image[i, j] = self.azimuth(effective_JM)
        return ret_image, azim_image


    def ret_and_azim_images_torch(self, volume_in : BirefringentVolume, micro_lens_offset=[0,0]):
        '''This function computes the retardance and azimuth images of the precomputed rays going through a volume'''
        # Include offset to move to the center of the volume, as the ray collisions are computed only for a single micro-lens
        # todo: n_micro_lenses and n_micro_lenses missmatch
        # n_micro_lenses = self.optic_config.mla_config.n_micro_lenses
        n_micro_lenses = self.optical_info['n_micro_lenses']
        n_ml_half = floor(n_micro_lenses / 2.0)
        micro_lens_offset = np.array(micro_lens_offset) + np.array(self.vox_ctr_idx[1:]) - n_ml_half
        # Fetch needed variables
        pixels_per_ml = self.optic_config.mla_config.n_pixels_per_mla
        # Create output images
        ret_image = torch.zeros((pixels_per_ml, pixels_per_ml), requires_grad=True)
        azim_image = torch.zeros((pixels_per_ml, pixels_per_ml), requires_grad=True)
        
        # Calculate Jones Matrices for all rays
        effective_JM = self.calc_cummulative_JM_of_ray(volume_in, micro_lens_offset)
        # Calculate retardance and azimuth
        retardance = self.retardance(effective_JM)
        azimuth = self.azimuth(effective_JM)
        ret_image.requires_grad = False
        azim_image.requires_grad = False
        zero_ret_idx = torch.isclose(retardance, torch.tensor([0.0], dtype=retardance.dtype))
        azimuth[zero_ret_idx] = 0
        # Assign the computed ray values to the image pixels
        for ray_ix, (i,j) in enumerate(self.ray_valid_indices):
            ret_image[i, j] = retardance[ray_ix]
            azim_image[i, j] = azimuth[ray_ix]
        return ret_image, azim_image


    # todo: once validated merge this with numpy function
    # todo: these are re-implemented in abstract_classes in OpticalElement
    def voxRayJM(self, Delta_n, opticAxis, rayDir, ell, wavelength):
        '''Compute Jones matrix associated with a particular ray and voxel combination'''
        if self.back_end == BackEnds.NUMPY:
            # Azimuth is the angle of the slow axis of retardance.
            azim = np.arctan2(np.dot(opticAxis, rayDir[1]), np.dot(opticAxis, rayDir[2]))
            if Delta_n == 0:
                azim = 0
            elif Delta_n < 0:
                azim = azim + np.pi / 2
            # print(f"Azimuth angle of index ellipsoid is {np.around(np.rad2deg(azim), decimals=0)} degrees.")
            ret = abs(Delta_n) * (1 - np.dot(opticAxis, rayDir[0]) ** 2) * 2 * np.pi * ell / wavelength
            # print(f"Accumulated retardance from index ellipsoid is {np.around(np.rad2deg(ret), decimals=0)} ~ {int(np.rad2deg(ret)) % 360} degrees.")
            
            # todo: compare speed
            # old method
            if False:
                offdiag = 1j * np.sin(2 * azim) * np.sin(ret / 2)
                diag1 = np.cos(ret / 2) + 1j * np.cos(2 * azim) * np.sin(ret / 2)
                diag2 = np.conj(diag1)

                JM = np.array([[diag1, offdiag], [offdiag, diag2]])
            else:
                JM = BirefringentJMgenerators.LR(ret,azim)

        elif self.back_end == BackEnds.PYTORCH:
            n_voxels = opticAxis.shape[0]
            if not torch.is_tensor(opticAxis):
                opticAxis = torch.from_numpy(opticAxis).to(Delta_n.device)
            # Azimuth is the angle of the sloq axis of retardance.
            azim = torch.arctan2(torch.linalg.vecdot(opticAxis , rayDir[1]), torch.linalg.vecdot(opticAxis , rayDir[2])) # todo: pvjosue dangerous, vecdot similar to dot?
            azim[Delta_n==0] = 0
            azim[Delta_n<0] += torch.pi / 2
            # print(f"Azimuth angle of index ellipsoid is {np.around(torch.rad2deg(azim).numpy(), decimals=0)} degrees.")
            ret = abs(Delta_n) * (1 - torch.linalg.vecdot(opticAxis, rayDir[0]) ** 2) * 2 * torch.pi * ell[:n_voxels] / wavelength
            # print(f"Accumulated retardance from index ellipsoid is {np.around(torch.rad2deg(ret).numpy(), decimals=0)} ~ {int(torch.rad2deg(ret).numpy()) % 360} degrees.")
            if True: # old method
                offdiag = 1j * torch.sin(2 * azim) * torch.sin(ret / 2)
                diag1 = torch.cos(ret / 2) + 1j * torch.cos(2 * azim) * torch.sin(ret / 2)
                diag2 = torch.conj(diag1)
                # Construct Jones Matrix
                JM = torch.zeros([Delta_n.shape[0], 2, 2], dtype=torch.complex64, device=Delta_n.device)
                JM[:,0,0] = diag1
                JM[:,0,1] = offdiag
                JM[:,1,0] = offdiag
                JM[:,1,1] = diag2
            else: # Much more operations in this method
                JM = BirefringentJMgenerators.LR(ret, azim, self.back_end)
        return JM

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
        product = torch.tensor([[1.0,0],[0,1.0]], dtype=torch.complex64, device=JMlist[0].device).unsqueeze(0).repeat(n_rays,1,1)
        for ix,JM in enumerate(JMlist):
            rays_with_voxels = [len(vx)>ix for vx in voxels_of_segs]
            product[rays_with_voxels,...] = product[rays_with_voxels,...] @ JM
        return product
        




########### Generate different birefringent volumes 
    def init_volume(self, volume_shape, init_mode='zeros', init_args={}):
        
        if init_mode=='zeros':
            if self.back_end == BackEnds.NUMPY:
                voxel_parameters = np.zeros([4,] + volume_shape)
            if self.back_end == BackEnds.PYTORCH:
                voxel_parameters = torch.zeros([4,] + volume_shape)
        elif init_mode=='random':
            voxel_parameters = self.generate_random_volume(volume_shape)
        elif 'planes' in init_mode:
            n_planes = int(init_mode[0])
            voxel_parameters = self.generate_planes_volume(volume_shape, n_planes) # Perpendicular optic axes each with constant birefringence and orientation 
        elif init_mode=='ellipsoid':
            # Look for variables in init_args, else init with something
            radius = init_args['radius'] if 'radius' in init_args.keys() else [5.5,5.5,3.5]
            center = init_args['center'] if 'center' in init_args.keys() else [0,0,0]
            delta_n = init_args['delta_n'] if 'delta_n' in init_args.keys() else 0.1
            alpha = init_args['border_thickness'] if 'border_thickness' in init_args.keys() else 0.1
            
            voxel_parameters = self.generate_ellipsoid_volume(volume_shape, center=center, radius=radius, alpha=alpha, delta_n=delta_n)
        
        volume_ref = BirefringentVolume(back_end=self.back_end, optical_info=self.optical_info,
                                        Delta_n=voxel_parameters[0,...], optic_axis=voxel_parameters[1:,...])
        # Enable gradients for auto-differentiation 
        # if self.back_end == BackEnds.PYTORCH:
        #     volume_ref.Delta_n.requires_grad = False
        #     volume_ref.optic_axis.requires_grad = False
        #     volume_ref.Delta_n = nn.Parameter(volume_ref.Delta_n.to(self.get_device()))
        #     volume_ref.optic_axis = nn.Parameter(volume_ref.optic_axis.detach())
        #     volume_ref.Delta_n.requires_grad = True
        #     volume_ref.optic_axis.requires_grad = True

        return volume_ref

    
    @staticmethod
    def generate_random_volume(volume_shape, init_args={'Delta_n_range' : [0,0.01], 'axes_range' : [-1,1]}):
        Delta_n = np.random.uniform(init_args['Delta_n_range'][0], init_args['Delta_n_range'][1], volume_shape)
        # Random axis
        a_0 = np.random.uniform(init_args['axes_range'][0], init_args['axes_range'][1], volume_shape)
        a_1 = np.random.uniform(init_args['axes_range'][0], init_args['axes_range'][1], volume_shape)
        a_2 = np.random.uniform(init_args['axes_range'][0], init_args['axes_range'][1], volume_shape)
        norm_A = np.sqrt(a_0**2+a_1**2+a_2**2)
        return np.concatenate((np.expand_dims(Delta_n, axis=0), np.expand_dims(a_0/norm_A, axis=0), np.expand_dims(a_1/norm_A, axis=0), np.expand_dims(a_2/norm_A, axis=0)),0)
    
    @staticmethod
    def generate_planes_volume(volume_shape, n_planes=1, z_offset=0):
        vol = np.zeros([4,] + volume_shape)
        z_size = volume_shape[0]
        z_ranges = np.linspace(0, z_size-1, n_planes*2).astype(int)

        if n_planes==1:
            # Birefringence
            vol[0, z_size//2+z_offset, :, :] = 0.1
            # Axis
            # vol[1, z_size//2, :, :] = 0.5
            vol[1, z_size//2+z_offset, :, :] = 1
            return vol
        random_data = BirefringentRaytraceLFM.generate_random_volume([n_planes])
        for z_ix in range(0,n_planes):
            vol[:,z_ranges[z_ix*2] : z_ranges[z_ix*2+1]] = np.expand_dims(random_data[:,z_ix],[1,2,3]).repeat(1,1).repeat(volume_shape[1],2).repeat(volume_shape[2],3)
        
        return vol
    
    @staticmethod
    def generate_ellipsoid_volume(volume_shape, center=[0.5,0.5,0.5], radius=[10,10,10], alpha=0.1, delta_n=0.1):
        # Grabbed from https://math.stackexchange.com/questions/2931909/normal-of-a-point-on-the-surface-of-an-ellipsoid
        vol = np.zeros([4,] + volume_shape)
        
        kk,jj,ii = np.meshgrid(np.arange(volume_shape[0]), np.arange(volume_shape[1]), np.arange(volume_shape[2]), indexing='ij')
        # shift to center
        kk = floor(center[0]*volume_shape[0]) - kk.astype(float)
        jj = floor(center[1]*volume_shape[1]) - jj.astype(float)
        ii = floor(center[2]*volume_shape[2]) - ii.astype(float)

        ellipsoid_border = (kk**2) / (radius[0]**2) + (jj**2) / (radius[1]**2) + (ii**2) / (radius[2]**2)
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
        return vol





###########################################################################################
    # Constructors for different types of elements
    # This methods are constructors only, they don't support torch optimization of internal variables
    # todo: rename such that it is clear that these are presets for different birefringent objects

class BirefringentJMgenerators(BirefringentElement):
    def __init__(self, back_end : BackEnds = BackEnds.NUMPY):
        super(BirefringentElement, self).__init__(back_end=back_end, torch_args={}, optical_info={})

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
        retarder_azim0 = BirefringentJMgenerators.LR_azim0(ret, back_end=back_end)
        R = BirefringentJMgenerators.rotator(azim, back_end=back_end)
        Rinv = BirefringentJMgenerators.rotator(-azim, back_end=back_end)
        return R @ retarder_azim0 @ Rinv

    @staticmethod
    def LR_azim0(ret, back_end=BackEnds.NUMPY):
        '''todo'''
        if back_end == BackEnds.NUMPY:
            return np.array([[np.exp(1j * ret / 2), 0], [0, np.exp(-1j * ret / 2)]])
        else:
            return torch.cat(
                (torch.cat((torch.exp(1j * ret / 2).unsqueeze(1), torch.zeros(len(ret),1)),1).unsqueeze(2),
                torch.cat((torch.zeros(len(ret),1), torch.exp(-1j * ret / 2).unsqueeze(1)),1).unsqueeze(2)),
                2
            )
            return torch.tensor([[torch.exp(1j * ret / 2), 0], [0, torch.exp(-1j * ret / 2)]])

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
        return BirefringentJMgenerators.LR(ret, azim)

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
        return BirefringentJMgenerators.rotator(-ret / 2)
    @staticmethod
    def LCR(ret):
        '''Left Circular Retarder'''
        return BirefringentJMgenerators.rotator(ret / 2)

    @staticmethod
    def universal_compensator(retA, retB):
        '''Universal Polarizer
        Used as the polarizer for the LC-PolScope'''
        return BirefringentJMgenerators.LR_azim0(retB) @ BirefringentJMgenerators.LR(retA, np.pi / 4) @ BirefringentJMgenerators.LP(0)


