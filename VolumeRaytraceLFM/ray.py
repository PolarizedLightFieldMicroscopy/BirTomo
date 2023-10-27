import time
import torch
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume, BirefringentRaytraceLFM

try:
    import torch
    DEVICE = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
except:
    DEVICE = "cpu"

def setup_raytracer(optical_info, backend=BackEnds.PYTORCH):
    """Initialize Birefringent Raytracer."""
    print(f'For raytracing, using computing device: cpu')
    rays = BirefringentRaytraceLFM(backend=backend, optical_info=optical_info).to('cpu')
    start_time = time.time()
    rays.compute_rays_geometry()
    print(f'Ray-tracing time in seconds: {time.time() - start_time}')
    return rays.to(DEVICE)
