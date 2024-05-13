"""This script contains the ForwardModel class."""
import os
import time
import matplotlib.pyplot as plt
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import (
    BirefringentVolume, BirefringentRaytraceLFM
)
from VolumeRaytraceLFM.visualization.plotting_ret_azim import (
    plot_retardance_orientation
)
from VolumeRaytraceLFM.visualization.plotting_intensity import (
    plot_intensity_images
)
from VolumeRaytraceLFM.jones.jones_calculus import JonesMatrixGenerators


class ForwardModel:
    """
    Simulates optical behavior of birefringent materials.

    Attributes:
        backend (BackEnds): Computational backend (PyTorch, NumPy).
        optical_info (dict): Optical system properties.
        rays (BirefringentRaytraceLFM): Raytracer instance.
        ret_img (Tensor/array): Simulated retardance image.
        azim_img (Tensor/array): Simulated azimuth image.
        volume_GT (Any): Ground truth for volume (optional).
        savedir (str): Directory for saving results.
        forward_img_dir (str): Directory for forward images.
        base_dir (str): Base directory for files.
        img_list (list): List of intensity images.
        ray_geometry_computation_time (float):
            Time for ray geometry computation.

    Methods:
        __init__: Initializes ForwardModel.
        to_device: Moves tensors to a computing device.
        is_pytorch_backend: Checks if backend is PyTorch.
        is_numpy_backend: Checks if backend is NumPy.
        convert_to_numpy: Converts to NumPy arrays.
        is_pytorch_tensor: Checks for PyTorch tensor.
        setup_raytracer: Sets up the raytracer.
        view_images: Displays retardance, azimuth images.
        view_intensity_image: Placeholder for intensity.
        save_ret_azim_images: Saves retardance, azimuth.
        save_intensity_image: Placeholder for saving.
        add_polscope_components: Adds polarizers, analyzers.
        plot_rays: Plots rays in 3D.
        create_savedir: Creates directories for saving.
        forward_model: Computes, updates simulation images.
    """

    def __init__(self, optical_system, backend, device='cpu'):
        self.backend = backend
        # Linking with the optical system
        self.optical_info = optical_system['optical_info']
        self.rays = self.setup_raytracer(device=device)
        self.rays.use_lenslet_based_filtering = False
        # Placeholders
        self.ret_img = None
        self.azim_img = None
        self.volume_GT = None
        self.savedir = None
        self.forward_img_dir = None
        self.base_dir = ""
        self.img_list = [None] * 5

        # Set up directories
        if False:
            self.create_savedir()

    def to_device(self, device):
        """Move all tensors to the specified device."""
        if self.is_pytorch_backend():
            if self.ret_img is not None:
                self.ret_img = self.ret_img.to(device)
            if self.azim_img is not None:
                self.azim_img = self.azim_img.to(device)
            if self.volume_GT is not None:
                self.volume_GT = self.volume_GT.to(device)
            self.rays.to_device(device)
        return self

    def is_pytorch_backend(self):
        """Check if the backend is pytorch."""
        return self.backend == BackEnds.PYTORCH

    def is_numpy_backend(self):
        """Check if the backend is numpy."""
        return self.backend == BackEnds.NUMPY

    def convert_to_numpy(self, data):
        """Convert the data to numpy array if it is a pytorch tensor.
        Otherwise, return the data as is. This method is useful for plotting
        and saving images.

        Note that the torch module does not need to be imported, as it would be
        if torch.is_tensor() is used to check if the data is a tensor.
        An alternative would be to use self.is_pytorch_backend().

        Args:
            data (torch.Tensor or np.array): Data to convert to numpy array.
        Returns:
            np.array: The converted data.
        """
        # Alternative is to use self.is_pytorch_backend()
        # This method avoid importing the torch module
        if self.is_pytorch_tensor(data):
            return data.detach().cpu().numpy()
        return data

    def is_pytorch_tensor(self, obj):
        """Check if the object is a pytorch tensor."""
        return "torch.Tensor" in str(type(obj))

    def setup_raytracer(self, device='cpu'):
        """Initialize Birefringent Raytracer."""
        print(f'For raytracing, using computing device {device}')
        rays = BirefringentRaytraceLFM(
            backend=self.backend, optical_info=self.optical_info
        )
        if self.is_pytorch_backend():
            rays.to_device(device)  # Move the rays to the specified device
        start_time = time.time()
        rays.compute_rays_geometry()
        self.ray_geometry_computation_time = time.time() - start_time
        print(f'Raytracing time in seconds: {self.ray_geometry_computation_time:.2f}')
        return rays

    def view_images(self, azimuth_plot_type='hsv'):
        """View the simulated images,
        and pause until the user closes the figure.
        Args:
            azimuth_plot_type (str): 'hsv' or 'lines'
        """
        ret_image = self.convert_to_numpy(self.ret_img)
        azim_image = self.convert_to_numpy(self.azim_img)
        my_fig = plot_retardance_orientation(
            ret_image, azim_image, azimuth_plot_type, include_labels=True
        )
        # my_fig.tight_layout()
        plt.pause(0.2)
        plt.show(block=True)

    def view_intensity_images(self):
        """View the simulated intensity images."""
        for i in range(len(self.img_list)):
            self.img_list[i] = self.convert_to_numpy(self.img_list[i])
        my_fig = plot_intensity_images(self.img_list)
        my_fig.tight_layout()
        plt.pause(0.2)
        plt.show(block=True)

    def save_ret_azim_images(self):
        """Save the simulated retardance and azimuth images."""
        self.create_savedir()
        ret_image = self.convert_to_numpy(self.ret_img)
        azim_image = self.convert_to_numpy(self.azim_img)
        my_fig = plot_retardance_orientation(
            ret_image, azim_image, 'hsv', include_labels=True)
        my_fig.savefig(self.savedir + '/ret_azim.png',
                       bbox_inches='tight', dpi=300)

    def save_intensity_images(self):
        """Save the simulated intensity images."""
        pass

    def add_polscope_components(self):
        """Add the polarizers and analyzers to the optical system. Non-identity
        polarizers and analyzers are used that model the LC-PolScope setup."""
        self.optical_info['polarizer'] = JonesMatrixGenerators.polscope_analyzer()
        self.optical_info['analyzer'] = JonesMatrixGenerators.universal_compensator_modes(
            setting=0, swing=0)

    def plot_rays(self):
        """Plot the rays in 3D."""
        self.rays.plot_rays()

    def create_savedir(self):
        """
        Create the directory where the forward images and possibly other
        results will be saved.
        """
        # Here's a basic structure; customize as needed
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        self.savedir = os.path.join(self.base_dir, "forward_images")
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)

    def forward_model(self, volume: BirefringentVolume,
                      intensity=False, all_lenslets=False):
        """
        Compute the forward model for a given volume using the simulator's
        attributes. This function updates the instance parameters with the
        computed images.
        Args:
            volume (BirefringentVolume): The volume to use for the forward
                                         model computation.
            intensity (bool): If True, compute the intensity images in addition
                              to the default retardance and azimuth images.
        Returns:
            None: This function does not return any value but updates instance
                  parameters.
        Creates/Updates:
            self.ret_img (torch.Tensor or np.array):
                The retardance image calculated by the model.
            self.azim_img (torch.Tensor or np.array):
                The azimuth image calculated by the model.
            self.img_list (list of np.arrays):
                List of intensity images, created only if 'intensity' is True.
        """
        ret_image, azim_image = self.rays.ray_trace_through_volume(
                                    volume, all_rays_at_once=all_lenslets)
        # print("Retardance and azimuth images computed with LC-PolScope setup")
        self.ret_img = ret_image
        self.azim_img = azim_image

        if intensity:
            self.optical_info['analyzer'] = JonesMatrixGenerators.left_circular_polarizer()
            self.optical_info['polarizer_swing'] = 0.03
            self.img_list = self.rays.ray_trace_through_volume(
                volume, intensity=True
            )
            print("Intensity images computed according to the LC-PolScope setup")
