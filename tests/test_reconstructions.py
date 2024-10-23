"""Tests for the reconstructions module."""

import numpy as np
import pytest
from tests.fixtures_optical_info import set_optical_info
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume
from VolumeRaytraceLFM.reconstructions import (
    ReconstructionConfig,
    Reconstructor,
)


@pytest.fixture
def recon_info():
    optical_info = set_optical_info([3, 5, 5], 17, 1)
    ret_image_meas = np.random.rand(17, 17)
    azim_image_meas = np.random.rand(17, 17)
    initial_volume = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=optical_info,
        volume_creation_args={
            "init_mode": "single_voxel",
            "init_args": {"delta_n": -0.05, "offset": [0, 0, 0]},
        },
    )
    iteration_params = {
        "num_iterations": 31,
        "azimuth_weight": 0.5,
        "regularization_weight": 0.1,
        "lr": 1e-3,
        "output_posfix": "",
    }
    recon_config = ReconstructionConfig(
        optical_info, ret_image_meas, azim_image_meas, initial_volume, iteration_params
    )
    return recon_config


@pytest.fixture
def reconstructor(recon_info):
    recon_info.recon_directory = ""
    return Reconstructor(recon_info)


def test_reconstructor_initialization(reconstructor):
    assert isinstance(reconstructor, Reconstructor)
    assert reconstructor.backend == BackEnds.PYTORCH


# def test_reconstruction_config():
#     # Test ReconstructionConfig initialization
#     optical_info = {'wavelength': 532e-9, 'refractive_index': 1.33}
#     ret_image = np.random.rand(10, 10)
#     azim_image = np.random.rand(10, 10)
#     initial_vol = BirefringentVolume(
#         backend=BackEnds.PYTORCH,
#         optical_info=recon_optical_info,
#         volume_creation_args = volume_args.random_args
#     )
#     iteration_params = {'num_iterations': 10, 'learning_rate': 0.01}
#     recon_config = ReconstructionConfig(optical_info, ret_image, azim_image, initial_vol, iteration_params)
#     assert recon_config.optical_info == optical_info
#     assert np.array_equal(recon_config.ret_image, ret_image)
#     assert np.array_equal(recon_config.azim_image, azim_image)
#     assert np.array_equal(recon_config.initial_vol, initial_vol)
#     assert recon_config.iteration_params == iteration_params

#     # Test ReconstructionConfig save and load
#     parent_directory = 'test_recon_config'
#     recon_config.save(parent_directory)
#     loaded_config = ReconstructionConfig.load(parent_directory)
#     assert loaded_config.optical_info == optical_info
#     assert np.array_equal(loaded_config.ret_image, ret_image)
#     assert np.array_equal(loaded_config.azim_image, azim_image)
#     assert np.array_equal(loaded_config.initial_vol, initial_vol)
#     assert loaded_config.iteration_params == iteration_params
#     os.rmdir(parent_directory)

# def test_reconstructor():
#     # Test Reconstructor initialization
#     optical_info = {'wavelength': 532e-9, 'refractive_index': 1.33}
#     ret_image = np.random.rand(10, 10)
#     azim_image = np.random.rand(10, 10)
#     initial_vol = np.random.rand(10, 10, 10)
#     iteration_params = {'num_iterations': 10, 'learning_rate': 0.01}
#     recon_config = ReconstructionConfig(optical_info, ret_image, azim_image, initial_vol, iteration_params)
#     reconstructor = Reconstructor(recon_config)
#     assert reconstructor.recon_config == recon_config

#     # Test Reconstructor setup_raytracer
#     reconstructor.setup_raytracer()
#     assert isinstance(reconstructor.raytracer, BirefringentRaytraceLFM)

#     # Test Reconstructor setup_initial_volume
#     reconstructor.setup_initial_volume()
#     assert np.array_equal(reconstructor.volume_estimation, initial_vol)

#     # Test Reconstructor mask_outside_rays
#     reconstructor.mask_outside_rays()
#     assert np.array_equal(reconstructor.volume_estimation, initial_vol)

#     # Test Reconstructor specify_variables_to_learn
#     reconstructor.specify_variables_to_learn()
#     assert len(reconstructor.variables_to_learn) == 1

#     # Test Reconstructor optimizer_setup
#     volume_estimation = torch.tensor(initial_vol, requires_grad=True)
#     training_params = {'optimizer': 'Adam', 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False}
#     reconstructor.optimizer_setup(volume_estimation, training_params)
#     assert isinstance(reconstructor.optimizer, torch.optim.Adam)

#     # Test Reconstructor compute_losses
#     ret_image_measured = np.random.rand(10, 10)
#     azim_image_measured = np.random.rand(10, 10)
#     ret_image_current = np.random.rand(10, 10)
#     azim_image_current = np.random.rand(10, 10)
#     losses = reconstructor.compute_losses(ret_image_measured, azim_image_measured, ret_image_current, azim_image_current, volume_estimation, training_params)
#     assert isinstance(losses, dict)
#     assert 'total_loss' in losses

#     # Test Reconstructor one_iteration
#     optimizer = reconstructor.optimizer
#     reconstructor.one_iteration(optimizer, volume_estimation)
#     assert np.array_equal(reconstructor.volume_estimation.detach().numpy(), initial_vol)

#     # Test Reconstructor visualize_and_save
#     fig = reconstructor.visualize_and_save(0, None, 'test_reconstructor')
#     assert fig is not None
#     os.rmdir('test_reconstructor')
