from VolumeRaytraceLFM.abstract_classes import *
import h5py
from tqdm import tqdm

class BirefringentElement(OpticalElement):
    ''' Birefringent element, such as voxel, raytracer, etc, extending optical element, so it has a back-end and optical information'''
    def __init__(self, backend : BackEnds = BackEnds.NUMPY, torch_args={},
                optical_info=None):
        super(BirefringentElement, self).__init__(backend=backend, torch_args=torch_args, optical_info=optical_info)

        self.simul_type = SimulType.BIREFRINGENT


###########################################################################################
# Implementations of OpticalElement
# todo: rename to BirefringentVolume inherits 
class BirefringentVolume(BirefringentElement):
    '''This class stores a 3D array of voxels with birefringence properties, either with a numpy or pytorch back-end.'''
    def __init__(self, backend=BackEnds.NUMPY, torch_args={},#{'optic_config' : None, 'members_to_learn' : []}, 
        optical_info={},#{'volume_shape' : [11,11,11], 'voxel_size_um' : 3*[1.0], 'pixels_per_ml' : 17, 'na_obj' : 1.2, 'n_medium' : 1.52, 'wavelength' : 0.550, 'n_micro_lenses' : 1},
        Delta_n=0.0, optic_axis=[1.0, 0.0, 0.0],
        volume_creation_args=None):
        '''BirefringentVolume
        Args:
            backend (BackEnd):     A computation BackEnd (Numpy vs Pytorch). If Pytorch is used, torch_args are required
                                    to initialize the head class OpticBlock from Waveblocks.
            torch_args (dict):       Required for PYTORCH back-end. Contains optic_config object and members_to_learn
            optical_info (dict):
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
            volume_creation_args (dict): 
                                    Containing information on how to init a volume, such as:
                                        init_type (str): zeros, nplanes, where n is a number, ellipsoid...
                                        init_args (dic): see self.init_volume function for specific arguments
                                        per init_type.

            '''
        super(BirefringentVolume, self).__init__(backend=backend, torch_args=torch_args, optical_info=optical_info)

        if self.backend == BackEnds.NUMPY:
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
                
        elif self.backend == BackEnds.PYTORCH:
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

            self.optic_axis = nn.Parameter(self.optic_axis.reshape(3,-1)).type(torch.get_default_dtype())
            self.Delta_n = nn.Parameter(self.Delta_n.flatten() ).type(torch.get_default_dtype())


        # Check if a volume creation was requested
        if volume_creation_args is not None:
            self.init_volume(volume_creation_args['init_mode'], volume_creation_args['init_args'] if 'init_args' in volume_creation_args.keys() else {})

    def get_delta_n(self):
        if self.backend == BackEnds.PYTORCH:
            return self.Delta_n.view(self.optical_info['volume_shape'])
        else:
            return self.Delta_n

    def get_optic_axis(self):
        if self.backend == BackEnds.PYTORCH:
            return self.optic_axis.view(3, self.optical_info['volume_shape'][0],
                                            self.optical_info['volume_shape'][1],
                                            self.optical_info['volume_shape'][2])
        else:
            return self.optic_axis

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
            torch.set_grad_enabled(has_grads)
            self.Delta_n.requires_grad = True
            self.optic_axis.requires_grad = True
        return self


    def plot_lines_plotly(self, opacity=0.5, mode='lines', colormap='Bluered_r', size_scaler=5, fig=None, draw_spheres=True):
        
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

        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        volume_shape = optical_info['volume_shape']
        volume_size_um = [optical_info['voxel_size_um'][i] * optical_info['volume_shape'][i] for i in range(3)]
        [dz, dxy, dxy] = optical_info['voxel_size_um']
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

        all_color[all_color!=0] -= all_color[all_color!=0].min()
        all_color += 0.5
        all_color /= all_color.max()

        if fig is None:
            fig = go.Figure()
        
        fig.add_scatter3d(z=all_x, y=all_y, x=all_z,
            marker=dict(color=all_color, colorscale=colormap, size=4), 
            line=dict(color=all_color, colorscale=colormap, width=size_scaler), 
            connectgaps=False, 
            mode='lines'
            )
        
        if draw_spheres:
            fig.add_scatter3d(z=x_base.flatten(), y=y_base.flatten(), x=z_base.flatten(),
                marker=dict(color=all_color[::3]-0.5, colorscale=colormap, size=size_scaler*5*all_color[::3]),
                line=dict(color=all_color[::3]-0.5, colorscale=colormap, width=5), 
                mode = 'markers')
        

        fig.update_layout(
            scene = dict(
                        xaxis = dict(nticks=volume_shape[0], range=[0, volume_size_um[0]]),
                        yaxis = dict(nticks=volume_shape[1], range=[0, volume_size_um[1]]),
                        zaxis = dict(nticks=volume_shape[2], range=[0, volume_size_um[2]]),
                        xaxis_title='Axial dimension',
                        aspectratio = dict( x=volume_size_um[0], y=volume_size_um[1], z=volume_size_um[2] ), aspectmode = 'manual'),
            # width=700,
            margin=dict(r=0, l=0, b=0, t=0),
            # autosize=True
            )
        # fig.data = fig.data[::-1]
        # fig.show()
        return fig

    @staticmethod
    def plot_volume_plotly(optical_info, voxels_in=None, opacity=0.5, colormap='gray', fig=None):
        
        voxels = voxels_in * 1.0
        
        # Check if this is a torch tensor
        if not isinstance(voxels_in, np.ndarray):
            try:
                voxels = voxels.detach()
                voxels = voxels.cpu().abs().numpy()
            except:
                pass
        voxels = np.abs(voxels)
                
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
            # colorscale=colormap
            )

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
        # fig.data = fig.data[::-1]
        # fig.show()
        return fig

    def get_vox_params(self, vox_index):
        '''vox_index is a tuple'''
        return self.Delta_n[vox_index], self.optic_axis[vox_index]



########### Generate different birefringent volumes 
    def save_as_file(self, h5_file_path):
        '''Store this volume into an h5 file'''
        print(f'Saving volume to h5 file: {h5_file_path}')

        # Create file
        with h5py.File(h5_file_path, "w") as f:
            # Save optical_info
            oc_grp = f.create_group('optical_info')
            for k,v in self.optical_info.items():
                try:
                    oc_grp.create_dataset(k, np.array(v).shape if isinstance(v,list) else [1], data=v)
                    # print(f'Added optical_info/{k} to {h5_file_path}')
                except:
                    pass

            # Save data (birefringence and optic_axis)
            delta_n = self.get_delta_n()
            optic_axis = self.get_optic_axis()

            if self.backend == BackEnds.PYTORCH:
                delta_n = delta_n.detach().cpu().numpy()
                optic_axis = optic_axis.detach().cpu().numpy()
            
            data_grp = f.create_group('data')
            data_grp.create_dataset("delta_n", delta_n.shape, data=delta_n.astype(np.float32))
            data_grp.create_dataset("optic_axis", optic_axis.shape, data=optic_axis.astype(np.float32))

    @staticmethod
    def init_from_file(h5_file_path, backend=BackEnds.NUMPY, optical_info=None):
        ''' Loads a birefringent volume from an h5 file and places it in the center of the volume
            It requires to have:
                optical_info/volume_shape [3]: shape of the volume in voxels [nz,ny,nx]
                data/delta_n [nz,ny,nx]: Birefringence volumetric information.
                data/optic_axis [3,nz,ny,nx]: Optical axis per voxel.'''

        # Load volume
        volume_file = h5py.File(h5_file_path, "r")

        # Fetch birefringence
        delta_n = np.array(volume_file['data/delta_n'])
        # Fetch optic_axis
        optic_axis = np.array(volume_file['data/optic_axis'])

        # Compute padding to match optica_info['volume_shape]
        z_,y_, x_ = delta_n.shape
        z, y, x = optical_info['volume_shape']
        assert z_<=z and y_<=y and x_<=x, f"Input volume is to large ({delta_n.shape}) for optical_info defined volume_shape {optical_info['volume_shape']}"

        z_pad = abs(z_-z)
        y_pad = abs(y_-y)
        x_pad = abs(x_-x)

        # Pad
        delta_n = np.pad(delta_n,(
                        (z_pad//2, z_pad//2 + z_pad%2),
                        (y_pad//2, y_pad//2 + y_pad%2), 
                        (x_pad//2, x_pad//2 + x_pad%2)),
                    mode = 'constant').astype(np.float64)
        
        optic_axis = np.pad(optic_axis,((0,0),
                        (z_pad//2, z_pad//2 + z_pad%2),
                        (y_pad//2, y_pad//2 + y_pad%2), 
                        (x_pad//2, x_pad//2 + x_pad%2)),
                    mode = 'constant').astype(np.float64)

        # Create volume
        volume_out = BirefringentVolume(backend=backend, optical_info=optical_info, Delta_n=delta_n, optic_axis=optic_axis)
        # return
        return volume_out

    def init_volume(self, init_mode='zeros', init_args={}):
        ''' This function creates predefined volumes and shapes, such as planes, ellipsoids, random, etc
            TODO: use init_args for random and planes'''

        volume_shape = self.optical_info['volume_shape']

        if init_mode=='zeros':
            if self.backend == BackEnds.NUMPY:
                voxel_parameters = np.zeros([4,] + volume_shape)
            if self.backend == BackEnds.PYTORCH:
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
        
        volume_ref = BirefringentVolume(backend=self.backend, optical_info=self.optical_info,
                                        Delta_n=voxel_parameters[0,...], optic_axis=voxel_parameters[1:,...])
        
        self.Delta_n = volume_ref.Delta_n
        self.optic_axis = volume_ref.optic_axis

    @staticmethod
    def generate_random_volume(volume_shape, init_args={'Delta_n_range' : [-0.00005,0.00005], 'axes_range' : [-0.05,0.05]}):
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
        random_data = BirefringentVolume.generate_random_volume([n_planes])
        for z_ix in range(0,n_planes):
            vol[:,z_ranges[z_ix*2] : z_ranges[z_ix*2+1]] = np.expand_dims(random_data[:,z_ix],[1,2,3]).repeat(1,1).repeat(volume_shape[1],2).repeat(volume_shape[2],3)
        
        return vol
    
    @staticmethod
    def generate_ellipsoid_volume(volume_shape, center=[0.5,0.5,0.5], radius=[10,10,10], alpha=0.1, delta_n=0.1):
        ''' generate_ellipsoid_volume: Creates an ellipsoid with optical axis normal to the ellipsoid surface.
            Args:
                Center [3]: [cz,cy,cx] from 0 to 1 where 0.5 is the center of the volume_shape.
                radius [3]: in voxels, the radius in z,y,x for this ellipsoid.
                alpha float: Border thickness.
                delta_n float: Delta_n value of birefringence in the volume
            '''
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

    @staticmethod
    def create_dummy_volume(backend=BackEnds.NUMPY, optical_info=None, vol_type="shell", volume_axial_offset=0):
        '''Create different volumes, some of them randomized... Feel free to add your volumes here'''
        
        # What's the center of the volume?
        vox_ctr_idx = np.array([optical_info['volume_shape'][0] / 2, optical_info['volume_shape'][1] / 2, optical_info['volume_shape'][2] / 2]).astype(int)
        if vol_type == "single_voxel":
            voxel_delta_n = 0.01
            # TODO: make numpy version of birefringence axis
            voxel_birefringence_axis = torch.tensor([1,0.0,0])
            voxel_birefringence_axis /= voxel_birefringence_axis.norm()

            # Create empty volume
            volume = BirefringentVolume(backend=backend, optical_info=optical_info, volume_creation_args={'init_mode' : 'zeros'})
            # Set delta_n
            volume.get_delta_n()[volume_axial_offset,
                                            vox_ctr_idx[1],
                                            vox_ctr_idx[2]] = voxel_delta_n
            # set optical_axis
            volume.get_optic_axis()[:, volume_axial_offset,
                                    vox_ctr_idx[1],
                                    vox_ctr_idx[2]] \
                                    = voxel_birefringence_axis

        elif vol_type in ["ellipsoid", "shell"]:    # whole plane
            ellipsoid_args = {  'radius' : [5.5, 9.5, 5.5],
                        'center' : [volume_axial_offset / optical_info['volume_shape'][0], \
                                        0.50, 0.5],  # from 0 to 1
                        'delta_n' : -0.01,
                        'border_thickness' : 0.3}
            volume = BirefringentVolume(backend=backend, optical_info=optical_info, volume_creation_args={'init_mode' : 'ellipsoid', 'init_args' : ellipsoid_args})

            # Do we want a shell? let's remove some of the volume
            if vol_type == 'shell':
                volume.get_delta_n()[:optical_info['volume_shape'][0] // 2 + 2,...] = 0


        elif 'ellipsoids' in vol_type:
            n_ellipsoids = int(vol_type[0])
            volume = BirefringentVolume(backend=backend, optical_info=optical_info, volume_creation_args={'init_mode' : 'zeros'})

            for n_ell in range(n_ellipsoids):
                ellipsoid_args = {  'radius' : np.random.uniform(.5, 3.5, [3]),
                                    'center' : [np.random.uniform(0.35, 0.65),] + list(np.random.uniform(0.3, 0.70, [2])),
                                    'delta_n' : np.random.uniform(-0.01, -0.001),
                                    'border_thickness' : 0.1}
                new_vol = BirefringentVolume(backend=backend, optical_info=optical_info, volume_creation_args={'init_mode' : 'ellipsoid', 'init_args' : ellipsoid_args})
                
                volume += new_vol

        # elif 'my_volume:' # Feel free to add new volumes here
        else:
            raise NotImplementedError
                
        return volume


############ Implementations
class BirefringentRaytraceLFM(RayTraceLFM, BirefringentElement):
    """This class extends RayTraceLFM, and implements the forward function, where voxels contribute to ray's Jones-matrices with a retardance and axis in a non-commutative matter"""
    def __init__(
            self, backend : BackEnds = BackEnds.NUMPY, torch_args={},#{'optic_config' : None, 'members_to_learn' : []},
            optical_info={}):#{'volume_shape' : [11,11,11], 'voxel_size_um' : 3*[1.0], 'pixels_per_ml' : 17, 'na_obj' : 1.2, 'n_medium' : 1.52, 'wavelength' : 0.550, 'n_micro_lenses' : 1}):
        # optic_config contains mla_config and volume_config
        super(BirefringentRaytraceLFM, self).__init__(
            backend=backend, torch_args=torch_args, optical_info=optical_info
        )

        # Ray-voxel colisions for different micro-lenses, this dictionary gets filled in: calc_cummulative_JM_of_ray_torch
        self.vox_indices_ml_shifted = {}
        
        
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

        n_voxels_per_ml_half = floor(self.optical_info['n_voxels_per_ml'] * n_micro_lenses / 2.0)

        # Check if the volume_size can fit these micro_lenses.
        # considering that some rays go beyond the volume in front of the micro-lens
        voxel_span_per_ml = self.voxel_span_per_ml + (n_micro_lenses*n_voxels_per_ml) + 1
        assert voxel_span_per_ml < volume_shape[1] and voxel_span_per_ml < volume_shape[2], f"The volume in front of the microlenses ({n_micro_lenses},{n_micro_lenses}) is to large for a volume_shape: {self.optical_info['volume_shape'][1:]}. Increase the volume_shape to at least [{voxel_span_per_ml},{voxel_span_per_ml}]"        

        # Traverse volume for every ray, and generate retardance and azimuth images
        full_img_r = None
        full_img_a = None
        # Iterate micro-lenses in y direction
        for ml_ii in tqdm(range(-n_ml_half, n_ml_half+1), f'Computing rows of micro-lens ret+azim {self.backend}'):
            full_img_row_r = None
            full_img_row_a = None
            # Iterate micro-lenses in x direction
            for ml_jj in range(-n_ml_half, n_ml_half+1):
                # Compute offset to top corner of the volume in front of the micro-lens (ii,jj)
                current_offset = np.array([n_voxels_per_ml * ml_ii, n_voxels_per_ml*ml_jj]) + np.array(self.vox_ctr_idx[1:]) - n_voxels_per_ml_half

                # Compute images for current microlens, by passing an offset to this function depending on the micro lens and the super resolution
                ret_image_torch, azim_image_torch = self.ret_and_azim_images(volume_in, micro_lens_offset=current_offset)
                # If this is the first image, create
                if full_img_row_r is None:
                    full_img_row_r = ret_image_torch
                    full_img_row_a = azim_image_torch
                else: # Concatenate to existing image otherwise
                    if self.backend == BackEnds.NUMPY:
                        full_img_row_r = np.concatenate((full_img_row_r, ret_image_torch), 0)
                        full_img_row_a = np.concatenate((full_img_row_a, azim_image_torch), 0)
                    elif self.backend == BackEnds.PYTORCH:
                        full_img_row_r = torch.cat((full_img_row_r, ret_image_torch), 0)
                        full_img_row_a = torch.cat((full_img_row_a, azim_image_torch), 0)
            if full_img_r is None:
                full_img_r = full_img_row_r
                full_img_a = full_img_row_a
            else:
                if self.backend == BackEnds.NUMPY:
                    full_img_r = np.concatenate((full_img_r, full_img_row_r), 1)
                    full_img_a = np.concatenate((full_img_a, full_img_row_a), 1)
                elif self.backend == BackEnds.PYTORCH:
                    full_img_r = torch.cat((full_img_r, full_img_row_r), 1)
                    full_img_a = torch.cat((full_img_a, full_img_row_a), 1)
        return full_img_r, full_img_a
 
    def retardance(self, JM):
        '''Phase delay introduced between the fast and slow axis in a Jones Matrix'''
        if self.backend == BackEnds.NUMPY:
            e1, e2 = np.linalg.eigvals(JM)
            phase_diff = np.angle(e1) - np.angle(e2)
            retardance = np.abs(phase_diff)
        elif self.backend == BackEnds.PYTORCH:
            x = torch.linalg.eigvals(JM)
            retardance = (torch.angle(x[:,1]) - torch.angle(x[:,0])).abs()
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
        if self.backend==BackEnds.NUMPY:
            return self.calc_cummulative_JM_of_ray_numpy(volume_in, micro_lens_offset)
        elif self.backend==BackEnds.PYTORCH:
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
        _voxels_of_segs, ell_in_voxels = self.ray_vol_colli_indices, self.ray_vol_colli_lengths


        # Compute the 1D index of each micro-lens.
        # compute once and store for later.
        # accessing 1D arrays increases training speed by 25%
        key = str(micro_lens_offset)
        if key not in self.vox_indices_ml_shifted.keys():
            self.vox_indices_ml_shifted[key] = [[RayTraceLFM.ravel_index((vox[ix][0], vox[ix][1]+micro_lens_offset[0], vox[ix][2]+micro_lens_offset[1]), self.optical_info['volume_shape']) for ix in range(len(vox))] for vox in self.ray_vol_colli_indices]
        voxels_of_segs = self.vox_indices_ml_shifted[key]

        # Init an array to store the Jones matrices.
        JM_list = []

        assert self.optical_info == volume_in.optical_info, 'Optical info between ray-tracer and volume mismatch. This might cause issues on the border micro-lenses.'
        # Iterate the interactions of all rays with the m-th voxel
        # Some rays interact with less voxels, so we mask the rays valid
        # for this step with rays_with_voxels
        for m in range(self.ray_vol_colli_lengths.shape[1]):
            # Check which rays still have voxels to traverse
            rays_with_voxels = [len(vx)>m for vx in _voxels_of_segs]
            # How many rays at this step
            # n_rays_with_voxels = sum(rays_with_voxels)
            # The lengths these rays traveled through the current voxels
            ell = ell_in_voxels[rays_with_voxels,m]
            # The voxel coordinates each ray collides with
            vox = [vx[m] for ix,vx in enumerate(voxels_of_segs) if rays_with_voxels[ix]]
            
            # Extract the information from the volume
            # Birefringence 
            Delta_n = volume_in.Delta_n[vox]
            # And axis
            opticAxis = volume_in.optic_axis[:,vox].permute(1,0)
            # Grab the subset of precomputed ray directions that have voxels in this step
            filtered_rayDir = self.ray_direction_basis[:,rays_with_voxels,:]

            # Compute the interaction from the rays with their corresponding voxels
            JM = self.voxRayJM( Delta_n = Delta_n,
                                opticAxis = opticAxis, 
                                rayDir = filtered_rayDir,
                                ell = ell,
                                wavelength=self.optical_info['wavelength'])

            if m==0:
                material_JM = JM
            else:
                material_JM[rays_with_voxels,...] = material_JM[rays_with_voxels,...] @ JM

        polarizer = torch.from_numpy(self.optical_info['polarizer']).type(torch.complex64).to(Delta_n.device)
        analyzer = torch.from_numpy(self.optical_info['analyzer']).type(torch.complex64).to(Delta_n.device)
        effective_JM = analyzer @ material_JM @ polarizer

        return effective_JM

    def ret_and_azim_images(self, volume_in : BirefringentVolume, micro_lens_offset=[0,0]):
        '''Calculate retardance and azimuth values for a ray with a Jones Matrix'''
        if self.backend==BackEnds.NUMPY:
            return self.ret_and_azim_images_numpy(volume_in, micro_lens_offset)
        elif self.backend==BackEnds.PYTORCH:
            return self.ret_and_azim_images_torch(volume_in, micro_lens_offset)

    def ret_and_azim_images_numpy(self, volume_in : BirefringentVolume, micro_lens_offset=[0,0]):
        '''Calculate retardance and azimuth values for a ray with a Jones Matrix'''

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

        # Fetch needed variables
        pixels_per_ml = self.optic_config.mla_config.n_pixels_per_mla
        
        
        # Calculate Jones Matrices for all rays
        effective_JM = self.calc_cummulative_JM_of_ray(volume_in, micro_lens_offset)
        # Calculate retardance and azimuth
        retardance = self.retardance(effective_JM)
        azimuth = self.azimuth(effective_JM)

        # Create output images
        ret_image = torch.zeros((pixels_per_ml, pixels_per_ml), dtype=torch.float32, requires_grad=True, device=self.get_device())
        azim_image = torch.zeros((pixels_per_ml, pixels_per_ml), dtype=torch.float32, requires_grad=True, device=self.get_device())
        ret_image.requires_grad = False
        azim_image.requires_grad = False

        # Fill the values in the images
        ret_image[self.ray_valid_indices[0,:],self.ray_valid_indices[1,:]] = retardance
        azim_image[self.ray_valid_indices[0,:],self.ray_valid_indices[1,:]] = azimuth
        # Alternative version
        # ret_image = torch.sparse_coo_tensor(indices = self.ray_valid_indices, values = retardance, size=(pixels_per_ml, pixels_per_ml)).to_dense()
        # azim_image = torch.sparse_coo_tensor(indices = self.ray_valid_indices, values = azimuth, size=(pixels_per_ml, pixels_per_ml)).to_dense()

        return ret_image, azim_image


    # todo: once validated merge this with numpy function
    # todo: these are re-implemented in abstract_classes in OpticalElement
    def voxRayJM(self, Delta_n, opticAxis, rayDir, ell, wavelength):
        '''Compute Jones matrix associated with a particular ray and voxel combination'''
        if self.backend == BackEnds.NUMPY:
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
                JM = JonesMatrixGenerators.linear_retarder(ret,azim)

        elif self.backend == BackEnds.PYTORCH:
            n_voxels = opticAxis.shape[0]
            if not torch.is_tensor(opticAxis):
                opticAxis = torch.from_numpy(opticAxis).to(Delta_n.device)

            # Dot product of optical axis and 3 ray-direction vectors
            OA_dot_rayDir = torch.linalg.vecdot(opticAxis, rayDir)

            # Azimuth is the angle of the sloq axis of retardance.
            azim = 2 * torch.arctan2(OA_dot_rayDir[1,:], OA_dot_rayDir[2,:])
            ret = abs(Delta_n) * (1 - OA_dot_rayDir[0,:] ** 2) * torch.pi * ell / wavelength

            if True: # old method
                offdiag = 1j * torch.sin(azim) * torch.sin(ret)
                diag1 = torch.cos(ret) + 1j * torch.cos(azim) * torch.sin(ret)
                diag2 = torch.conj(diag1)
                # Construct Jones Matrix
                JM = torch.zeros([Delta_n.shape[0], 2, 2], dtype=torch.complex64, device=Delta_n.device)
                JM[:,0,0] = diag1
                JM[:,0,1] = offdiag
                JM[:,1,0] = offdiag
                JM[:,1,1] = diag2
            else: # Much more operations in this method
                JM = JonesMatrixGenerators.linear_retarder(ret, azim, self.backend)
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
        


###########################################################################################
    # Constructors for different types of elements
    # This methods are constructors only, they don't support torch optimization of internal variables

class JonesMatrixGenerators(BirefringentElement):
    def __init__(self, backend : BackEnds = BackEnds.NUMPY):
        super(BirefringentElement, self).__init__(backend=backend, torch_args={}, optical_info={})

    @staticmethod
    def rotator(angle, backend=BackEnds.NUMPY):
        '''2D rotation matrix
        Args:
            angle: angle to rotate by counterclockwise [radians]
        Return: Jones matrix'''
        if backend == BackEnds.NUMPY:
            s = np.sin(angle)
            c = np.cos(angle)
            R = np.array([[c, -s], [s, c]])
        elif backend == BackEnds.PYTORCH:
            s = torch.sin(angle)
            c = torch.cos(angle)
            R = torch.tensor([[c, -s], [s, c]])
        return R

    @staticmethod
    def linear_retarder(ret, azim, backend=BackEnds.NUMPY):
        '''Linear retarder
        Args:
            ret (float): retardance [radians]
            azim (float): azimuth angle of fast axis [radians]
        Return: Jones matrix    
        '''
        retarder_azim0 = JonesMatrixGenerators.linear_retarder_azim0(ret, backend=backend)
        R = JonesMatrixGenerators.rotator(azim, backend=backend)
        Rinv = JonesMatrixGenerators.rotator(-azim, backend=backend)
        return R @ retarder_azim0 @ Rinv

    @staticmethod
    def linear_retarder_azim0(ret, backend=BackEnds.NUMPY):
        '''todo'''
        if backend == BackEnds.NUMPY:
            return np.array([[np.exp(1j * ret / 2), 0], [0, np.exp(-1j * ret / 2)]])
        else:
            return torch.cat(
                (torch.cat((torch.exp(1j * ret / 2).unsqueeze(1), torch.zeros(len(ret),1)),1).unsqueeze(2),
                torch.cat((torch.zeros(len(ret),1), torch.exp(-1j * ret / 2).unsqueeze(1)),1).unsqueeze(2)),
                2
            )
            return torch.tensor([[torch.exp(1j * ret / 2), 0], [0, torch.exp(-1j * ret / 2)]])

    @staticmethod
    def linear_retarter_azim90(ret, backend=BackEnds.NUMPY):
        '''todo
        using same convention as linear_retarder_azim90'''
        if backend == BackEnds.NUMPY:
            return np.array([[np.exp(1j * ret / 2), 0], [0, np.exp(-1j * ret / 2)]])
        else:
            return torch.tensor([torch.exp(1j * ret / 2), 0], [0, torch.exp(-1j * ret / 2)])

    @staticmethod
    def quarter_waveplate(azim):
        '''Quarter Waveplate
        Linear retarder with lambda/4 or equiv pi/2 radians
        Commonly used to convert linear polarized light to circularly polarized light'''
        ret = np.pi / 2
        return JonesMatrixGenerators.linear_retarder(ret, azim)

    @staticmethod
    def half_waveplate(azim):
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
    def linear_polarizer(theta):
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
    def right_circular_polarizer():
        '''Right Circular Polarizer'''
        return 1 / 2 * np.array([[1, -1j], [1j, 1]])

    @staticmethod
    def left_circular_polarizer():
        '''Left Circular Polarizer'''
        return 1 / 2 * np.array([[1, 1j], [-1j, 1]])
    @staticmethod
    def right_circular_retarder(ret):
        '''Right Circular Retarder'''
        return JonesMatrixGenerators.rotator(-ret / 2)
    @staticmethod
    def left_circular_retarder(ret):
        '''Left Circular Retarder'''
        return JonesMatrixGenerators.rotator(ret / 2)

    @staticmethod
    def polscope_analyzer():
        '''Acts as a circular polarizer
        Inhomogeneous elements because eigenvectors are linear (-45 deg) and (right) circular polarization states
        Source: 2010 Polarized Light pg. 224'''
        return 1 / (2 * np.sqrt(2)) * np.array([[1 + 1j, 1 - 1j], [1 + 1j, 1 - 1j]])

    @staticmethod
    def universal_compensator(retA, retB):
        '''Universal Polarizer
        Used as the polarizer for the LC-PolScope'''
        return JonesMatrixGenerators.linear_retarder_azim0(retB) @ JonesMatrixGenerators.linear_retarder(retA, -np.pi / 4) @ JonesMatrixGenerators.linear_polarizer(0)

    @staticmethod
    def universal_compensator(retA, retB):
        '''Universal Polarizer
        Used as the polarizer for the LC-PolScope'''
        LP = JonesMatrixGenerators.linear_polarizer(0)
        LCA = JonesMatrixGenerators.linear_retarder(retA, -np.pi / 4)
        LCB = JonesMatrixGenerators.linear_retarder_azim0(retB)
        return LCB @ LCA @ LP

    @staticmethod
    def universal_compensator_modes(setting=0, swing=0):
        '''Settings for the LC-PolScope polarizer
        Parameters:
            setting (int): LC-PolScope setting number between 0 and 4
            swing (float): proportion of wavelength, for ex 0.03
        Returns:
            Jones matrix'''
        swing_rad = swing * 2 * np.pi
        if setting == 0:
            retA = np.pi / 2
            retB = np.pi
        elif setting == 1:
            retA = np.pi / 2 + swing_rad
            retB = np.pi
        elif setting == 2:
            retA = np.pi / 2
            retB = np.pi + swing_rad
        elif setting == 3:
            retA = np.pi / 2
            retB = np.pi - swing_rad
        elif setting == 4:
            retA = np.pi / 2 - swing_rad
            retB = np.pi
        return JonesMatrixGenerators.universal_compensator(retA, retB)


class JonesVectorGenerators(BirefringentElement):
    def __init__(self, backend : BackEnds = BackEnds.NUMPY):
        super(BirefringentElement, self).__init__(backend=backend, torch_args={}, optical_info={})

    @staticmethod
    def right_circular(backend=BackEnds.NUMPY):
        return np.array([1, -1j])

    @staticmethod
    def left_circular(backend=BackEnds.NUMPY):
        return np.array([1, 1j])

    @staticmethod
    def linear(angle):
        return JonesMatrixGenerators.rotator(angle) @ np.array([1, 0])

    @staticmethod
    def horizonal():
        return np.array([1, 0])

    @staticmethod
    def vertical():
        return np.array([0, 1])


