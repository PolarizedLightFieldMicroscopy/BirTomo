![python version](https://img.shields.io/badge/python-3.10-blue)
[![GitHub Actions Demo](https://github.com/PolarizedLightFieldMicroscopy/forward-model/actions/workflows/github-actions-demo.yml/badge.svg)](https://github.com/PolarizedLightFieldMicroscopy/forward-model/actions/workflows/github-actions-demo.yml)
# forward-model
forward model using mainly ray tracing

## How to use
For the forward model, the main script is main_forward_projection.py. The workflow within that script is the following:

1. Create a birefringent raytracer.
1. Compute the ray geometry.
1. Create a volume to image.
1. Raytrace through the volume.

For the iterative reconstruction, the main script is main_3d_reconstruction.py.

You can also use our streamlit app that is only the streamlit cloud: https://polarizedlightfieldmicroscopy-forward-mo-user-interface-dc1r85.streamlit.app/

## Requirements
virtual environment should include the waveblocks requirements

See environment.txt and environment.yml files.

### Running forward projection using numpy method
Necessary packages:
- matplotlib (numpy is included)
- tqdm

## Testing
- Run pytest in terminal to check that all the tests pass.
- Run ```pytest -v test_jones.py::test_polscope``` to run a particular test.