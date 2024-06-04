"""Forward model fixtures for testing."""

import pytest
from VolumeRaytraceLFM.simulations import ForwardModel
from tests.fixtures_optical_info import set_optical_info


@pytest.fixture
def forward_model_fixture(backend_fixture):
    """Create forward model fixture for testing."""
    optical_info = set_optical_info([3, 5, 5], 17, 1)
    optical_system = {"optical_info": optical_info}

    return ForwardModel(optical_system, backend_fixture)
