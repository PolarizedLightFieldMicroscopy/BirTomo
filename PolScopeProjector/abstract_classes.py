import torch
import torch.nn as nn

class PolObject(nn.Module):
    # Main class containing a light propagation mode associated with an object
    def __init__(self, pol_mode):
        super(PolObject, self).__init__()
        self.light_mode_types = ['fluorescent', 'fluorecent_polarized', 'dipole', 'birefringent']


        self.pol_mode = pol_mode
        assert pol_mode in self.light_mode_types, f'Polarization mode not found, please provide one of: {self.light_mode_types}'
    
    # Helper function to test if an object has the same mode as this one
    def check_pol_mode_compatibility(self, obj):
        return obj.pol_mode == self.pol_mode


class PolVolume(PolObject):
    # A volume asociated with a light propagation mode
    def __init__(self, shape, pol_mode):
        super(PolVolume, self).__init__(pol_mode)

    def forward(self, volume):
        raise NotImplementedError

class PolVolumeProjector(PolObject):
    # A Forward Projector abstract operation that evaluates light traveling through a volume
    # For example: rays running through the volume, emmiter dipoles, wave-optics scattering models
    def __init__(self, shape, pol_mode):
        super(PolVolumeProjector, self).__init__(pol_mode)

    def forward(self, volume : PolVolume):
        raise NotImplementedError



