from VolumeRaytraceLFM.abstract_classes import (
    BackEnds, OpticalElement, SimulType
)


class BirefringentElement(OpticalElement):
    ''' Birefringent element, such as voxel, raytracer, etc,
    extending optical element, so it has a back-end and optical information'''

    def __init__(self, backend: BackEnds = BackEnds.NUMPY, torch_args={},
                 optical_info=None):
        super(BirefringentElement, self).__init__(backend=backend,
                                                  torch_args=torch_args,
                                                  optical_info=optical_info
                                                  )
        self.simul_type = SimulType.BIREFRINGENT
