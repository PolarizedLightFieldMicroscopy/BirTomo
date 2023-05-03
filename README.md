![python version](https://img.shields.io/badge/python-3.10-blue)
[![GitHub Actions Demo](https://github.com/PolarizedLightFieldMicroscopy/forward-model/actions/workflows/github-actions-demo.yml/badge.svg)](https://github.com/PolarizedLightFieldMicroscopy/forward-model/actions/workflows/github-actions-demo.yml)
# forward-model
Polarized light field microscopy forward model and inverse model using geometrical optics and Jones Calculus.

## How to use
For the forward model, the main script is main_forward_projection.py. The workflow within that script is the following:

1. Create a birefringent raytracer.
1. Compute the ray geometry.
1. Create a volume to image.
1. Raytrace through the volume.

For the iterative reconstruction, the main script is main_3d_reconstruction.py.
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

You can also use our streamlit app that runs on the streamlit cloud: https://polarizedlightfieldmicroscopy-forward-mo-user-interface-dc1r85.streamlit.app/

## Requirements

<!-- See environment.txt and environment.yml files. -->

Necessary packages:
- matplotlib (numpy is included)
- tqdm
- torch
- h5py (for reading and saves volumes)
- plotly (for visualizing volumes)
- os (for saving images)

Run the following code to create a virtual environment will all the necessary and relevant packages:
```
conda create --name model python=3.10 tqdm matplotlib h5py --yes
conda activate model
conda install -c conda-forge pytorch --yes
conda install -c plotly plotly --yes
pip install streamlit
conda install -c anaconda pytest --yes
```

## Testing
- Run ```pytest``` in the terminal to check that all the tests pass.
- An example of running a particular test is ```pytest -v test_jones.py::test_polscope```.