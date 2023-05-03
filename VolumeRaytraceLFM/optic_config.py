# Third party libraries imports
try:
    import torch.nn as nn
    import torch
    import torch.nn.functional as F
except:
    pass

class OpticBlock(nn.Module):  # pure virtual class
    """Base class containing all the basic functionality of an optic block"""

    def __init__(
        self, optic_config=None, members_to_learn=None,
    ):  # Contains a list of members which should be optimized (In case none are provided members are created without gradients)
        super(OpticBlock, self).__init__()
        self.optic_config = optic_config
        self.members_to_learn = [] if members_to_learn is None else members_to_learn
        self.device_dummy = nn.Parameter(torch.tensor([1.0]))


    def get_trainable_variables(self):
        trainable_vars = []
        for name, param in self.named_parameters():
            if name in self.members_to_learn:
                trainable_vars.append(param)
        return list(trainable_vars)
    
    def get_device(self):
        return self.device_dummy.device


# Python imports
import math

# Third party libraries imports
import torch.nn as nn
import numpy as np


class PSFConfig:
    pass

class scattering:
    pass

class MLAConfig:
    pass

class CameraConfig:
    pass

class PolConfig:
    pass

class VolumeConfig:
    pass




class OpticConfig(nn.Module):
    """Class containing the global parameters of an optical system:
    Keyword args:
    wavelenght: wavelenght of optical system (in um)
    samplingRate: sampling used when propagating wavefronts through space. Or when using a camera.
    k: wave number
    """

    @staticmethod
    def get_default_PSF_config():
        psf_config = PSFConfig()
        psf_config.M = 40  # Magnification
        psf_config.NA = 0.9  # Numerical Aperture
        psf_config.Ftl = 165000  # Tube lens focal length
        psf_config.ns = 1.33  # Specimen refractive index (RI)
        psf_config.ng0 = 1.515  # Coverslip RI design value
        psf_config.ng = 1.515  # Coverslip RI experimental value
        psf_config.ni0 = 1  # Immersion medium RI design value
        psf_config.ni = 1  # Immersion medium RI experimental value
        psf_config.ti0 = (
            150  # Microns, working distance (immersion medium thickness) design value
        )
        psf_config.tg0 = 170  # Microns, coverslip thickness design value
        psf_config.tg = 170  # Microns, coverslip thickness experimental value
        psf_config.zv = (
            0  # Offset of focal plane to coverslip, negative is closer to objective
        )
        psf_config.wvl = 0.63  # Wavelength of emission

        # Sample space information
        psf_config.voxel_size = [6.5/60, 6.5/60, 0.43] # Axial step size in sample space
        psf_config.depths = list(np.arange(-10*psf_config.voxel_size[-1], 10*psf_config.voxel_size[-1], psf_config.voxel_size[-1])) # Axial depths where we compute the PSF
            
        return psf_config
    
    @staticmethod
    def get_default_MLA_config():
        mla_config = MLAConfig()
        mla_config.use_mla = True
        mla_config.pitch = 100
        mla_config.camera_distance = 2500
        mla_config.focal_length = 2500
        mla_config.arrangement_type = "periodic"
        mla_config.n_voxels_per_ml = 1 # How many voxels per micro-lens
        mla_config.n_micro_lenses = 1

        return mla_config

    def setup_parameters(self):
        # Setup last PSF parameters
        self.PSF_config.fobj = self.PSF_config.Ftl / self.PSF_config.M
        # Calculate MLA number of pixels behind a lenslet
        self.mla_config.n_pixels_per_mla = 2 * [self.mla_config.pitch // self.camera_config.sensor_pitch]
        self.mla_config.n_pixels_per_mla = [int(n + (1 if (n % 2 == 0) else 0)) for n in self.mla_config.n_pixels_per_mla]
        # Calculate voxel size
        voxel_size_xy = self.camera_config.sensor_pitch / self.PSF_config.M
        self.PSF_config.voxel_size = [voxel_size_xy, voxel_size_xy, self.PSF_config.voxel_size[-1]]

        return
    @staticmethod
    def get_default_camera_config():
        camera_config = CameraConfig()
        camera_config.sensor_pitch = 6.5
        return camera_config

    @staticmethod
    def get_polarizers():
        pol_config = PolConfig()
        pol_config.polarizer = np.array([[1, 0], [0, 1]])
        pol_config.analyzer = np.array([[1, 0], [0, 1]])
        return pol_config
    
    def __init__(self, PSF_config=None):
        super(OpticConfig, self).__init__()
        if PSF_config is None:
            self.PSF_config = self.get_default_PSF_config()
        else:
            self.PSF_config = PSF_config
        self.set_k()
        self.mla_config = self.get_default_MLA_config()
        self.camera_config = self.get_default_camera_config()
        self.pol_config = self.get_polarizers()
        self.volume_config = VolumeConfig()


    def get_wavelenght(self):
        return self.PSF_config.wvl

    def get_medium_refractive_index(self):
        return self.PSF_config.ni

    def set_k(self):
        # Wave Number
        self.k = 2 * math.pi * self.PSF_config.ni / self.PSF_config.wvl  # wave number
        
    def get_k(self):
        return self.k


# Convert volume to single 2D MIP image, input [batch,1,xDim,yDim,zDim]
def volume_2_projections(vol_in, proj_type=torch.sum, scaling_factors=[1,1,1], depths_in_ch=True, ths=[0.0,1.0], normalize=False, border_thickness=1, add_scale_bars=True, scale_bar_vox_sizes=[40,20]):
    vol = vol_in.detach().clone().abs()
    # Normalize sets limits from 0 to 1
    if normalize:
        vol -= vol.min()
        vol /= vol.max()
    if depths_in_ch:
        vol = vol.permute(0,3,2,1).unsqueeze(1)
    if ths[0]!=0.0 or ths[1]!=1.0:
        vol_min,vol_max = vol.min(),vol.max()
        vol[(vol-vol_min)<(vol_max-vol_min)*ths[0]] = 0
        vol[(vol-vol_min)>(vol_max-vol_min)*ths[1]] = vol_min + (vol_max-vol_min)*ths[1]

    vol_size = list(vol.shape)
    vol_size[2:] = [vol.shape[i+2] * scaling_factors[i] for i in range(len(scaling_factors))]

    x_projection = proj_type(vol.float().cpu(), dim=2)
    y_projection = proj_type(vol.float().cpu(), dim=3)
    z_projection = proj_type(vol.float().cpu(), dim=4)

    out_img = z_projection.min() * torch.ones(
        vol_size[0], vol_size[1], vol_size[2] + vol_size[4] + border_thickness, vol_size[3] + vol_size[4] + border_thickness
    )

    out_img[:, :, : vol_size[2], : vol_size[3]] = z_projection
    out_img[:, :, vol_size[2] + border_thickness :, : vol_size[3]] = F.interpolate(x_projection.permute(0, 1, 3, 2), size=[vol_size[-1],vol_size[-3]], mode='nearest')
    out_img[:, :, : vol_size[2], vol_size[3] + border_thickness :] = F.interpolate(y_projection, size=[vol_size[2],vol_size[4]], mode='nearest')


    if add_scale_bars:
        line_color = out_img.max()
        # Draw white lines
        out_img[:, :, vol_size[2]: vol_size[2]+ border_thickness, ...] = line_color
        out_img[:, :, :, vol_size[3]:vol_size[3]+border_thickness, ...] = line_color
        # start = 0.02
        # out_img[:, :, int(start* vol_size[2]):int(start* vol_size[2])+4, int(0.9* vol_size[3]):int(0.9* vol_size[3])+scale_bar_vox_sizes[0]] = line_color
        # out_img[:, :, int(start* vol_size[2]):int(start* vol_size[2])+4, vol_size[2] + border_thickness + 10 : vol_size[2] + border_thickness + 10 + scale_bar_vox_sizes[1]*scaling_factors[2]] = line_color
        # out_img[:, :, vol_size[2] + border_thickness + 10 : vol_size[2] + border_thickness + 10 + scale_bar_vox_sizes[1]*scaling_factors[2], int(start* vol_size[2]):int(start* vol_size[2])+4] = line_color

    return out_img