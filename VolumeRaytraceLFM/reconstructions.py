import torch
import time
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume, BirefringentRaytraceLFM
from VolumeRaytraceLFM.optic_config import volume_2_projections
from plotting_tools import plot_iteration_update, plot_retardance_orientation

class ReconstructionConfig:
    def __init__(self, optical_info, ret_image, azim_image, initial_volume, iteration_params, loss_function=None):
        """
        Initialize the ReconstructorConfig with the provided parameters.

        optical_info: The optical parameters for the reconstruction process.
        retardance_image: Measured retardance image.
        azimuth_image: Measured azimuth image.
        initial_volume: An initial estimation of the volume.
        """
        assert isinstance(optical_info, dict), "Expected optical_info to be a dictionary"
        assert isinstance(ret_image, torch.Tensor), "Expected ret_image to be a PyTorch Tensor"
        assert isinstance(azim_image, torch.Tensor), "Expected azim_image to be a PyTorch Tensor"
        assert isinstance(initial_volume, BirefringentVolume), "Expected initial_volume to be of type BirefringentVolume"
        assert isinstance(iteration_params, dict), "Expected iteration_params to be a dictionary"
        if loss_function:
            assert callable(loss_function), "Expected loss_function to be callable"

        self.optical_info = optical_info
        self.retardance_image = ret_image.detach()
        self.azimuth_image = azim_image.detach()
        self.initial_volume = initial_volume
        self.interation_parameters = iteration_params



class Reconstructor:
    def __init__(self, recon_info: ReconstructionConfig):
        """
        Initialize the Reconstructor with the provided parameters.

        iteration_params (class): containing reconstruction parameters
        """
        self.backend = BackEnds.PYTORCH
        self.optical_info = recon_info.optical_info
        self.ret_img_meas = recon_info.retardance_image
        self.azim_img_meas = recon_info.azimuth_image
        self.initial_vol = recon_info.initial_volume # if initial_volume is not None else self._initialize_volume()
        self.iteration_params = recon_info.interation_parameters

        self.rays = self.setup_raytracer()

    def _initialize_volume(self):
        """
        Method to initialize volume if it's not provided.
        Here, we can return a default volume or use some initialization strategy.
        """
        # Placeholder for volume initialization
        default_volume = None
        return default_volume

    def setup_raytracer(self):
        """Initialize Birefringent Raytracer."""
        print(f'For raytracing, using default computing device, likely cpu')
        rays = BirefringentRaytraceLFM(backend=self.backend, optical_info=self.optical_info)
        start_time = time.time()
        rays.compute_rays_geometry()
        print(f'Ray-tracing time in seconds: {time.time() - start_time}')
        return rays

    def setup_initial_volume(self):
        """Setup initial estimated volume."""
        initial_volume = BirefringentVolume(backend=BackEnds.PYTORCH,
                                            optical_info=self.optical_info,
                                            volume_creation_args = {'init_mode' : 'random'}
                                            )
        # Let's rescale the random to initialize the volume
        initial_volume.Delta_n.requires_grad = False
        initial_volume.optic_axis.requires_grad = False
        initial_volume.Delta_n *= -0.01
        # # And mask out volume that is outside FOV of the microscope
        mask = self.rays.get_volume_reachable_region()
        initial_volume.Delta_n[mask.view(-1)==0] = 0
        initial_volume.Delta_n.requires_grad = True
        initial_volume.optic_axis.requires_grad = True
        # Indicate to this object that we are going to optimize Delta_n and optic_axis
        initial_volume.members_to_learn.append('Delta_n')
        initial_volume.members_to_learn.append('optic_axis')
        return initial_volume

    def mask_outside_rays(self):
        # Mask out volume that is outside FOV of the microscope
        self.initial_vol.Delta_n.requires_grad = False
        self.initial_vol.optic_axis.requires_grad = False
        mask = self.rays.get_volume_reachable_region()
        self.initial_vol.Delta_n[mask.view(-1)==0] = 0
        self.initial_vol.Delta_n.requires_grad = True
        self.initial_vol.optic_axis.requires_grad = True

    def specify_variables_to_learn(self, learning_vars=None):
        """
        Specify which variables of the initial volume object should be considered for learning.
        This method updates the 'members_to_learn' attribute of the initial volume object, ensuring
        no duplicates are added.
        Parameters:
            learning_vars (list): Variable names to be appended for learning.
                                    Defaults to ['Delta_n', 'optic_axis'].
        """
        volume = self.initial_vol
        if learning_vars is None:
            learning_vars = ['Delta_n', 'optic_axis']
        for var in learning_vars:
            if var not in volume.members_to_learn:
                volume.members_to_learn.append(var)  

    def optimizer_setup(self, volume_estimation, training_params):
        """Setup optimizer."""
        trainable_parameters = volume_estimation.get_trainable_variables()
        return torch.optim.Adam(trainable_parameters, lr=training_params['lr'])

    def compute_losses(self, ret_image_measured, azim_image_measured, ret_image_current, azim_image_current, volume_estimation, training_params):
        # Vector difference GT
        co_gt, ca_gt = ret_image_measured * torch.cos(azim_image_measured), ret_image_measured * torch.sin(azim_image_measured)
        # Compute data term loss
        co_pred, ca_pred = ret_image_current * torch.cos(azim_image_current), ret_image_current * torch.sin(azim_image_current)
        data_term = ((co_gt - co_pred) ** 2 + (ca_gt - ca_pred) ** 2).mean()

        # Compute regularization term
        delta_n = volume_estimation.get_delta_n()
        TV_reg = (
            (delta_n[1:, ...] - delta_n[:-1, ...]).pow(2).sum() +
            (delta_n[:, 1:, ...] - delta_n[:, :-1, ...]).pow(2).sum() +
            (delta_n[:, :, 1:] - delta_n[:, :, :-1]).pow(2).sum()
        )
        axis_x = volume_estimation.get_optic_axis()[0, ...]
        TV_reg_axis_x = (
            (axis_x[1:, ...] - axis_x[:-1, ...]).pow(2).sum() +
            (axis_x[:, 1:, ...] - axis_x[:, :-1, ...]).pow(2).sum() +
            (axis_x[:, :, 1:] - axis_x[:, :, :-1]).pow(2).sum()
        )
        # regularization_term = TV_reg + 1000 * (volume_estimation.Delta_n ** 2).mean() + TV_reg_axis_x / 100000
        regularization_term = TV_reg + 1000 * (volume_estimation.Delta_n ** 2).mean()

        # Total loss
        L = data_term + training_params['regularization_weight'] * regularization_term
        return L, data_term, regularization_term

    def one_iteration(self, ret_image_measured, azim_image_measured, optimizer, rays, volume_estimation, training_params):
        optimizer.zero_grad()

        # Forward project
        [ret_image_current, azim_image_current] = rays.ray_trace_through_volume(volume_estimation)
        L, data_term, regularization_term = self.compute_losses(ret_image_measured, azim_image_measured, ret_image_current, azim_image_current, volume_estimation, training_params)

        L.backward()
        optimizer.step()

        return L.item(), data_term.item(), regularization_term.item(), ret_image_current, azim_image_current


    def handle_visualization_and_saving(self, ep, Delta_n_GT, ret_image_measured, azim_image_measured, 
                                        volume_estimation, figure, output_dir, 
                                        ret_image_current, azim_image_current,
                                        losses, data_term_losses, regularization_term_losses):
        if ep % 1 == 0:
            plt.clf()
            plot_iteration_update(
                volume_2_projections(Delta_n_GT.unsqueeze(0))[0, 0].detach().cpu().numpy(),
                ret_image_measured.detach().cpu().numpy(),
                azim_image_measured.detach().cpu().numpy(),
                volume_2_projections(volume_estimation.get_delta_n().unsqueeze(0))[0, 0].detach().cpu().numpy(),
                    ret_image_current.detach().cpu().numpy(),
                    azim_image_current.detach().cpu().numpy(),
                    losses,
                    data_term_losses,
                    regularization_term_losses
            )
            figure.canvas.draw()
            figure.canvas.flush_events()
            time.sleep(0.1)
            if False:
                plt.savefig(f"{output_dir}/Optimization_ep_{'{:02d}'.format(ep)}.pdf")
            time.sleep(0.1)

        # if ep % 100 == 0:
        #     volume_estimation.save_as_file(f"{output_dir}/volume_ep_{'{:02d}'.format(ep)}.h5")
        return

    def reconstruct(self):
        """
        Method to perform the actual reconstruction based on the provided parameters.
        """
        Delta_n_GT = copy.deepcopy(self.initial_vol).get_delta_n().detach()
        output_dir = ''
        # self.mask_outside_rays()
        self.specify_variables_to_learn()
        volume_estimation = copy.deepcopy(self.initial_vol)
        self.initial_vol.Delta_n.requires_grad = False
        self.initial_vol.optic_axis.requires_grad = False
        optimizer = self.optimizer_setup(volume_estimation, self.iteration_params)
        figure = setup_visualization()

        # Lists to store losses
        losses = []
        data_term_losses = []
        regularization_term_losses = []


        # Training loop
        for ep in tqdm(range(self.iteration_params['n_epochs']), "Minimizing"):
            loss, data_term_loss, regularization_term_loss, ret_image_current, azim_image_current = self.one_iteration(
                self.ret_img_meas, self.azim_img_meas, optimizer, self.rays, volume_estimation, self.iteration_params
                )
            # loss, data_term_loss, regularization_term_loss, ret_image_current, azim_image_current = one_iteration_og(co_gt, ca_gt, optimizer, rays, volume_estimation, training_params)
            # Record losses
            losses.append(loss)
            data_term_losses.append(data_term_loss)
            regularization_term_losses.append(regularization_term_loss)

            # Visualization and saving
            self.handle_visualization_and_saving(ep, Delta_n_GT, self.ret_img_meas, self.azim_img_meas,
                                            volume_estimation, figure, output_dir,
                                            ret_image_current, azim_image_current,
                                            losses, data_term_losses, regularization_term_losses
                                            )
        # Final visualizations after training completes
        if False:
            plt.savefig(f"{output_dir}/Optimization_final.pdf")
        plt.show()

def setup_visualization():
    plt.ion()
    figure = plt.figure(figsize=(18, 9))
    plt.rcParams['image.origin'] = 'lower'
    if False:
        manager = plt.get_current_fig_manager()
        manager.window.setGeometry(4000, 600, 1800, 900)
    return figure
