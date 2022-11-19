import torch
import torch.nn as nn
from abstract_classes import *

class PolRayProjector(PolVolumeProjector):
    # Ray projector able to compute intersection with volume voxels with the Siddon algorithm
    def __init__(self, shape, pol_mode, ray_geometry : dict):
        super(PolRayProjector, self).__init__(pol_mode)

        ## Todo: precompute ray-volume interesction and store it in self
        # self.ray_geometry = pass

    def forward(self, volume : PolVolume):
        raise NotImplementedError   

##### Implementations for different types

class PolRayProjectorFluorescence(PolRayProjector):
    # An Ray handler that computes raytracing through a volume assuming fluorescence emmision, without polarization
    def __init__(self, shape, pol_mode, ray_geometry : dict):
        super(PolRayProjectorFluorescence, self).__init__(pol_mode, ray_geometry)

    def forward(self, volume : PolVolume):
        raise NotImplementedError

class PolRayProjectorPolarizedFluorescence(PolRayProjector):
    # An Ray handler that computes raytracing through a volume assuming fluorescence emmision, without polarization
    def __init__(self, shape, pol_mode, ray_geometry : dict):
        super(PolRayProjectorPolarizedFluorescence, self).__init__(pol_mode, ray_geometry)

        ## Todo: precompute ray-volume interesction and store it in self
        
    def forward(self, volume : PolVolume):
        raise NotImplementedError

class PolRayProjectorBirefringence(PolRayProjector):
    # An Ray handler that computes raytracing through a volume assuming fluorescence emmision, without polarization
    def __init__(self, shape, pol_mode, ray_geometry : dict):
        super(PolRayProjectorBirefringence, self).__init__(pol_mode, ray_geometry)

        ## Todo: precompute ray-volume interesction and store it in self

    def compute_ray_jones_vectors_from_birefringence(self):
        # self.jones_vectors = self.ray_geometry * ... .... todo
        pass

    def forward(self, volume : PolVolume):
        raise NotImplementedError