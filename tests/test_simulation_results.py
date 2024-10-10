"""Tests for the simulation results.

Images to compare against are generated and saved with:

    images = run_simulation("shell_small", [7, 18, 18], 16, 9)
    filename = generate_filename("shell_small", [7, 18, 18], 16, 9)
    save_images(images, filename)
"""

import pytest
import torch
from VolumeRaytraceLFM.simulations import ForwardModel
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume
from VolumeRaytraceLFM.volumes import volume_args
from tests.test_all import check_azimuth_images
from tests.fixtures_optical_info import set_optical_info

BACKEND = BackEnds.PYTORCH
PARAMETER_SETS = [
    ("voxel", [3, 5, 5], 16, 1),
    ("voxel", [3, 9, 9], 16, 5),
    ("sphere2", [11, 30, 30], 16, 11),
    ("plane", [4, 8, 8], 16, 4),
    ("shell_small", [7, 18, 18], 16, 9),
]

# Unit test idea: images of a (shifted) voxel should be the same for
#                   all odd axial dimension volumes

def create_simulator(optical_info, backend):
    optical_system = {"optical_info": optical_info}
    simulator = ForwardModel(optical_system, backend)
    return simulator


def run_simulation(vol_type, vol_shape, pixels_per_ml, n_lenslets):
    optical_info = set_optical_info(vol_shape, pixels_per_ml, n_lenslets)
    simulator = create_simulator(optical_info, BACKEND)
    if vol_type == "voxel":
        vol_args = volume_args.voxel_args
    elif vol_type == "sphere2":
        vol_args = volume_args.sphere_args2
    elif vol_type == "plane":
        vol_args = volume_args.plane_args
    elif vol_type == "shell":
        vol_args = volume_args.shell_args
    elif vol_type == "shell_small":
        vol_args = volume_args.shell_small_args
    else:
        vol_args = volume_args.random_args
    with torch.no_grad():
        volume = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=vol_args,
        )
        simulator.rays.prepare_for_all_rays_at_once()
        simulator.forward_model(volume, all_lenslets=True)
        # simulator.view_images()
        images = simulator.ret_img, simulator.azim_img
    return images


def generate_filename(vol_type, vol_shape, pixels_per_ml, n_lenslets):
    return f"precomputed_images_{vol_type}_{vol_shape[0]}_{vol_shape[1]}_{vol_shape[2]}_{pixels_per_ml}_{n_lenslets}.pt"


def save_images(images, filename):
    filename = f"tests/test_data/{filename}"
    torch.save(images, filename)
    print(f"Images saved to {filename}")


def compare_images(generated_images, saved_images):
    assert torch.allclose(
        generated_images[0], saved_images[0], atol=5e-4
    ), "Retardance images differ"
    check_azimuth_images(generated_images[1], saved_images[1])
    print("Images match the saved images.")


@pytest.mark.parametrize(
    "vol_type, vol_shape, pixels_per_ml, n_lenslets",
    PARAMETER_SETS,
)
@pytest.mark.slow
def test_simulation(vol_type, vol_shape, pixels_per_ml, n_lenslets):
    torch.set_default_dtype(torch.float32)
    torch.set_grad_enabled(False)
    images = run_simulation(vol_type, vol_shape, pixels_per_ml, n_lenslets)
    filename = generate_filename(vol_type, vol_shape, pixels_per_ml, n_lenslets)
    filepath = f"tests/test_data/{filename}"
    try:
        saved_images = torch.load(filepath)
        compare_images(images, saved_images)
    except FileNotFoundError:
        print(f"Saved images not found at {filepath}")
    except Exception as e:
        print(f"Failed to compare images: {e}")
        raise

if __name__ == "__main__":
    from fixtures_optical_info import set_optical_info
    # images = run_simulation("shell_small", [7, 18, 18], 16, 9)

    # Loop through the parameter sets and execute functions
    for vol_type, vol_shape, pixels_per_ml, n_lenslets in PARAMETER_SETS:
        images = run_simulation(vol_type, vol_shape, pixels_per_ml, n_lenslets)
        filename = generate_filename(vol_type, vol_shape, pixels_per_ml, n_lenslets)
        save_images(images, filename)
