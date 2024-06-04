import json
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume


def setup_optical_parameters(config_file=None):
    """Setup optical parameters based on a configuration file."""
    optical_info = BirefringentVolume.get_optical_info_template()
    if config_file is not None:
        with open(config_file, "r") as f:
            config = json.load(f)
        optical_info.update(config)
    return optical_info


def setup_iteration_parameters(config_file=None):
    """Setup iteration parameters based on a configuration file."""
    if config_file is not None:
        with open(config_file, "r") as f:
            iteration_params = json.load(f)
    else:
        iteration_params = {
            "n_epochs": 201,
            "azimuth_weight": 0.5,
            "regularization_weight": 0.1,
            "lr": 1e-3,
            "output_posfix": "",
        }
    return iteration_params
