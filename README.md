![python versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
[![Run Pytest](https://github.com/PolarizedLightFieldMicroscopy/forward-model/actions/workflows/pytest-action.yml/badge.svg)](https://github.com/PolarizedLightFieldMicroscopy/forward-model/actions/workflows/pytest-action.yml)
# GeoBirT
Polarized light field microscopy forward model and inverse model using geometrical optics and Jones Calculus.

## Installation
Run the following code to create a virtual environment will all the necessary and relevant packages:
```
conda create --name bir-tomo python=3.11 --yes
conda activate bir-tomo
pip install -e .
```
If you have a CUDA on your computer, or having issues with pytorch, try following the instructions [here](https://pytorch.org/get-started/locally/) for installing pytorch.

To download an editable installation in developer mode:
```
pip install -e .[dev]
```

### Requirements
See `pyproject.toml` for the dependencies.

Necessary packages:
- matplotlib (numpy is included)
- tqdm
- torch
- h5py (for reading and saving volumes)
- tifffile (for reading and saving images)
- plotly (for visualizing volumes)
- ipykernel (for using jupyter notebooks)
- streamlit (for running the streamlit page locally)
- pytest (for testing code during development)
- scikit-image (for analyzing images)

## Birefringence tomography
*To be updated soon...*

For the forward model, the main script is `run_simulations.py`. The workflow within that script is the following:

1. Create a birefringent raytracer.
1. Compute the ray geometry.
1. Create a volume to image.
1. Raytrace through the volume.

For the iterative reconstruction, the main script is `run_recon.py`.
The workflow within that script is the following:
1. Generate birefringence and retardance images with forward model that will serve as the ground truth (measurement) images.
    1. Create a birefringent raytracer.
    1. Compute the ray geometry.
    1. Create a volume to image.
    1. Raytrace through the volume.
1. Make an initial guess of the volume.
1. Create an optimizer.
    1. Choose to optimize for the birefringence $\Delta n$ and optic axis $\hat{a}$
    1. Choose an optimization method, such as gradient descent.
1. Define a loss function to be minimized.
1. Perform many iterations of the estimated volume being updated from the gradients of the loss function with respect to the estimated volume.

Open the streamlit page locally with
```
streamlit run User_Interface.py
```

You can also use our streamlit app that runs on the streamlit cloud and uses the code in the *stream* branch: https://polarizedlightfieldmicroscopy-forward-mo-user-interface-dc1r85.streamlit.app/

## Testing
- Run ```pytest``` in the terminal to check that all the tests pass.
- An example of running a particular test is ```pytest -v tests/test_jones.py::test_polscope```.
