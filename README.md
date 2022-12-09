# forward-model
forward model using mainly ray tracing

## How to use
For the forward model, the main script is main_forward_projection.py. The workflow within that script is the following:

1. Create a birefringent raytracer.
1. Compute the ray geometry.
1. Create a volume to image.
1. Raytrace through the volume.

For the iterative reconstruction, the main script is main_3D_bire_recon_pytorch.py.

## Requirements
virtual environment should include the waveblocks requirements

See environment.txt and environment.yml files.