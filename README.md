# forward-model
forward model using mainly ray tracing

## How to use
For the forward model, the main script is main_forward_projection.py. The workflow within that script is the following:

1. Create a birefringent raytracer.
2. Compute the ray geometry.
3. Create a volume to image.
3. Raytrace through the volume.

For the iterative reconstruction, the main script is main_3D_bire_recon_pytorch.py.

## Requirements
virtual environment should include the waveblocks requirements