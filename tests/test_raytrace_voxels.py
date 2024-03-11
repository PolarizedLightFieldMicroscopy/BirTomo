"""Tests for identifying voxels to raytrace."""
import pytest
from tests.fixtures_optical_info import set_optical_info
from VolumeRaytraceLFM.simulations import (
    BackEnds, ForwardModel
)

def has_common_number(lists):
    """Check if there exists a number that is in every list within the list of lists."""
    # Use set intersection to find common elements
    common_elements = set(lists[0])  # Start with elements from the first list
    for lst in lists[1:]:
        common_elements &= set(lst)  # Update the set with intersection
        if not common_elements:
            return False  # Early return if no common elements found
    return True if common_elements else False


def has_common_tuple(lists_of_tuples):
    """Check if there exists a tuple that is in every list within the list of lists of tuples."""
    # Use set intersection to find common elements
    common_elements = set(lists_of_tuples[0])  # Start with elements from the first list
    for lst in lists_of_tuples[1:]:
        common_elements &= set(lst)  # Update the set with intersection
        if not common_elements:
            return False  # Early return if no common elements found
    return True if common_elements else False


@pytest.mark.parametrize("num_microlenses", [1, 3, 5, 7, 9])
def test_raytrace_common_voxel(num_microlenses):
    optical_info = set_optical_info([3, 7, 7], 17, num_microlenses)
    optical_system = {"optical_info": optical_info}
    simulator = ForwardModel(optical_system, BackEnds.PYTORCH)
    simulator.rays.store_shifted_vox_indices()
    vox_indices_list = simulator.rays.vox_indices_by_mla_idx[(0, 0)]
    err_msg = "All rays should pass through a common voxel for a single microlens."
    assert has_common_number(vox_indices_list) == True, err_msg
    ctr_idx = num_microlenses // 2
    vox_indices_list = simulator.rays.vox_indices_by_mla_idx[(ctr_idx, ctr_idx)]
    assert has_common_number(vox_indices_list) == True, err_msg


@pytest.mark.parametrize("num_microlenses", [1, 3, 5, 7, 9])
def test_raytrace_common_collision_voxel(num_microlenses):
    """The common element should be:
            (1, num_microlenses // 2, num_microlenses // 2).
    """
    optical_info = set_optical_info([3, 7, 7], 17, num_microlenses)
    optical_system = {"optical_info": optical_info}
    simulator = ForwardModel(optical_system, BackEnds.PYTORCH)
    vox_indices_list = simulator.rays.ray_vol_colli_indices
    err_msg = "All rays should pass through a common voxel for a single microlens."
    assert has_common_tuple(vox_indices_list) == True, err_msg
