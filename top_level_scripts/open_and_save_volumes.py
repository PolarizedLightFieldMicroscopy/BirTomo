"""Open, visualize, and save a phantom."""

import os
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume
from VolumeRaytraceLFM.setup_parameters import setup_optical_parameters
from VolumeRaytraceLFM.volumes import volume_args
from VolumeRaytraceLFM.visualization.plotting_volume import visualize_volume

BACKEND = BackEnds.NUMPY
BASE_DIR = "data/phantoms"
OPTICAL_CONFIG_FILENAME = "optical_config_volume.json"
SAVE_DIR = "Oct30"
VOL_NAME = "volume"

if __name__ == "__main__":
    optical_info = setup_optical_parameters(
        os.path.join(BASE_DIR, OPTICAL_CONFIG_FILENAME)
    )
    print(f"The volume shape is {optical_info['volume_shape']}.")
    # Create a volume from specific parameters
    volume = BirefringentVolume(
        backend=BACKEND,
        optical_info=optical_info,
        volume_creation_args=volume_args.shell_args,
    )
    # Remove half of the volume to turn the ellipsoid into a shell
    volume.get_delta_n()[: optical_info["volume_shape"][0] // 2 + 2, ...] = 0
    # Visualize the volume
    visualize_volume(volume, optical_info)
    # Save the volume
    file_dir = os.path.join(BASE_DIR, SAVE_DIR)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
        print(f"Directory '{file_dir}' was created.")
    filename = os.path.join(BASE_DIR, SAVE_DIR, VOL_NAME + ".h5")
    volume.save_as_file(filename, description="shell")
    # Open the same volume. Note that if optical_info['volume_shape'] is
    #   different than the volume dimensions, cropping or padding will occur.
    volume_from_file = BirefringentVolume.init_from_file(
        filename, backend=BACKEND, optical_info=optical_info
    )
    # Visualize the volume
    visualize_volume(volume_from_file, optical_info)
