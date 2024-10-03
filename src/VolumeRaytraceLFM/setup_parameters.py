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
            "general": {
                "num_iterations": 1000,
                "save_freq": 100
            },

            "learning_rates": {
                "birefringence": 1e-4,
                "optic_axis": 1e-1
            },

            "regularization": {
                "weight": 1.0,
                "functions": [
                    ["birefringence active L2", 100]
                ]
            },
        }
    return iteration_params
