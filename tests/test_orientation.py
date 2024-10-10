"""Test the azimuth image for different optic axis configurations."""

import pytest
import torch
import math
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.simulations import ForwardModel
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume

# Setting up constants
BACKEND = BackEnds.PYTORCH

@pytest.fixture
def optical_system_fixture():
    optical_info = {
        "volume_shape": [1, 3, 3],
        "axial_voxel_size_um": 1.0,
        "cube_voxels": True,
        "pixels_per_ml": 1,
        "n_micro_lenses": 1,
        "n_voxels_per_ml": 1,
        "M_obj": 60,
        "na_obj": 1.2,
        "n_medium": 1.35,
        "wavelength": 0.550,
        "aperture_radius_px": 1,
        "camera_pix_pitch": 6.5,
        "polarizer": [[1, 0], [0, 1]],
        "analyzer": [[1, 0], [0, 1]],
        "polarizer_swing": 0.03
    }
    return {"optical_info": optical_info}


@pytest.fixture(scope="function")
def setup_simulator(optical_system_fixture):
    simulator = ForwardModel(optical_system_fixture, backend=BACKEND)
    simulator.rays.prepare_for_all_rays_at_once()
    return simulator


# Function to test the azimuth image for a given volume
def compare_azimuth_image(simulator, volume, expected_output):
    simulator.forward_model(volume, all_lenslets=True)
    azim = simulator.azim_img
    print(f"  - Computed Azimuth: {azim.item():.4f}\n"
          f"  - Expected Output: {expected_output.item():.4f}")
    azim = torch.where(torch.isclose(azim, torch.tensor(torch.pi), atol=1e-8), torch.tensor(0.0), azim)
    # Check if the azimuth image matches the expected output
    assert torch.allclose(azim, expected_output.float(), atol=1e-6), f"Azimuth image is not as expected. Got {azim}, expected {expected_output}"


# Function to create a birefringent volume with given optic axis and delta_n
def create_birefringent_volume(optical_system_fixture, optic_axis, delta_n=-0.05):
    voxel_args = {
        "init_mode": "single_voxel",
        "init_args": {"delta_n": delta_n, "offset": [0, 0, 0], "optic_axis": optic_axis},
    }
    volume = BirefringentVolume(
        backend=BACKEND,
        optical_info=optical_system_fixture["optical_info"],
        volume_creation_args=voxel_args,
    )
    volume.set_requires_grad(False)
    return volume
    

# Parametrize the test with different configurations
@pytest.mark.parametrize(
    "optic_axis, delta_n, expected_output", [
        # Positive Birefringence (delta_n > 0)
        ([1, 0, 0], 0.05, torch.tensor([[0]])),               # azim = 0
        ([-1, 0, 0], 0.05, torch.tensor([[0]])),              # azim = 0
        ([0, 1, 0], 0.05, torch.tensor([[math.pi/2]])),       # azim = pi/2
        ([0, -1, 0], 0.05, torch.tensor([[math.pi/2]])),      # azim = pi/2
        ([0, 0, 1], 0.05, torch.tensor([[0]])),               # azim = 0
        ([0, 0, -1], 0.05, torch.tensor([[0]])),              # azim = 0
        ([0, 1, 1], 0.05, torch.tensor([[math.pi/4]])),       # azim = pi/4
        ([0, -1, -1], 0.05, torch.tensor([[math.pi/4]])),     # azim = pi/4
        ([0, -1, 1], 0.05, torch.tensor([[3*math.pi/4]])),    # azim = 3pi/4
        ([0, 1, -1], 0.05, torch.tensor([[3*math.pi/4]])),    # azim = 3pi/4

        # Negative Birefringence (delta_n < 0)
        ([1, 0, 0], -0.05, torch.tensor([[0]])),               # azim = 0
        ([-1, 0, 0], -0.05, torch.tensor([[0]])),              # azim = 0
        ([0, 1, 0], -0.05, torch.tensor([[0]])),               # azim flipped to 0
        ([0, -1, 0], -0.05, torch.tensor([[0]])),              # azim flipped to 0
        ([0, 0, 1], -0.05, torch.tensor([[math.pi/2]])),       # azim = pi/2
        ([0, 0, -1], -0.05, torch.tensor([[math.pi/2]])),      # azim = pi/2
        ([0, 1, 1], -0.05, torch.tensor([[3*math.pi/4]])),      # azim flipped to 3pi/4
        ([0, -1, -1], -0.05, torch.tensor([[3*math.pi/4]])),    # azim flipped to 3pi/4
        ([0, -1, 1], -0.05, torch.tensor([[math.pi/4]])),       # azim flipped to pi/4
        ([0, 1, -1], -0.05, torch.tensor([[math.pi/4]])),       # azim flipped to pi/4
    ]
)
def test_azimuth_images(optic_axis, delta_n, expected_output, setup_simulator, optical_system_fixture):
    """Test different optic axis configurations with positive and negative birefringence."""
    # Create the simulator (only once for all tests)
    simulator = setup_simulator
    
    # Create a birefringent volume for the test
    volume = create_birefringent_volume(optical_system_fixture,optic_axis, delta_n=delta_n)
    
    # Run the test on the azimuth image
    compare_azimuth_image(simulator, volume, expected_output)


if __name__ == "__main__":
    # Define 10 different configurations for volumes
    volumes_config_positive_birefringent = [
        {"optic_axis": [1, 0, 0], "expected_output": torch.tensor([[0]])},               # azim = 0
        {"optic_axis": [-1, 0, 0], "expected_output": torch.tensor([[0]])},              # azim = 0
        {"optic_axis": [0, 1, 0], "expected_output": torch.tensor([[math.pi/2]])},       # azim = pi/2
        {"optic_axis": [0, -1, 0], "expected_output": torch.tensor([[math.pi/2]])},     # azim = pi/2
        {"optic_axis": [0, 0, 1], "expected_output": torch.tensor([[0]])},              # azim = 0
        {"optic_axis": [0, 0, -1], "expected_output": torch.tensor([[0]])},             # azim = 0
        {"optic_axis": [0, 1, 1], "expected_output": torch.tensor([[math.pi/4]])},       # azim = pi/4
        {"optic_axis": [0, -1, -1], "expected_output": torch.tensor([[math.pi/4]])},     # azim = pi/4
        {"optic_axis": [0, -1, 1], "expected_output": torch.tensor([[3*math.pi/4]])},    # azim = 3pi/4
        {"optic_axis": [0, 1, -1], "expected_output": torch.tensor([[3*math.pi/4]])},    # azim = 3pi/4
    ]

    volumes_config_negative_birefringent = [
        {"optic_axis": [1, 0, 0], "expected_output": torch.tensor([[0]])},               
        {"optic_axis": [-1, 0, 0], "expected_output": torch.tensor([[0]])},
        {"optic_axis": [0, 1, 0], "expected_output": torch.tensor([[0]])},      
        {"optic_axis": [0, -1, 0], "expected_output": torch.tensor([[0]])},     
        {"optic_axis": [0, 0, 1], "expected_output": torch.tensor([[math.pi/2]])},             
        {"optic_axis": [0, 0, -1], "expected_output": torch.tensor([[math.pi/2]])},              
        {"optic_axis": [0, 1, 1], "expected_output": torch.tensor([[3*math.pi/4]])},     
        {"optic_axis": [0, -1, -1], "expected_output": torch.tensor([[3*math.pi/4]])},   
        {"optic_axis": [0, -1, 1], "expected_output": torch.tensor([[math.pi/4]])},
        {"optic_axis": [0, 1, -1], "expected_output": torch.tensor([[math.pi/4]])},
    ]


    # optical_system["optical_info"]["pixels_per_ml"] = 4
    # optical_system["optical_info"]["aperture_radius_px"] = 3
    simulator = setup_simulator(optical_system_fixture)
    
    birefringence = 0.05
    print("Birefringence: ", birefringence)
    if birefringence > 0:
        volumes_config = volumes_config_positive_birefringent[::2]
    else:
        volumes_config = volumes_config_negative_birefringent[::2]

    single_only = False
    if not single_only:
        # Iterate through the 10 volume configurations, run the tests, and check results
        for idx, config in enumerate(volumes_config):
            volume = create_birefringent_volume(config["optic_axis"], delta_n=birefringence)
            print(f"Testing volume {idx + 1} with optic_axis: {config['optic_axis']}")
            compare_azimuth_image(simulator, volume, config["expected_output"])
            # print(f"Test {idx + 1} passed.")
    else:
        volume = create_birefringent_volume([1, 0, 0], delta_n=birefringence)
        simulator.forward_model(volume, all_lenslets=True)
        simulator.view_images()
