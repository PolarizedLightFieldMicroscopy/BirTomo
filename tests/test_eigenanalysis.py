import torch
import pytest
from VolumeRaytraceLFM import JonesMatrixGenerators, BackEnds
from VolumeRaytraceLFM.jones.eigenanalysis import (
    retardance_from_jones, retardance_from_su2
)


def generate_jones_matrix(batch_size=10):
    ret = torch.rand(batch_size) * 2 * torch.pi  # Tensor of shape [batch_size]
    azim = torch.rand(batch_size) * 2 * torch.pi  # Tensor of shape [batch_size]
    # Generate the batch of Jones matrices using the linear_retarder method
    jones_matrices = []
    for i in range(batch_size):
        jones_matrix = JonesMatrixGenerators.linear_retarder(ret[i], azim[i], backend=BackEnds.PYTORCH)
        jones_matrices.append(jones_matrix)
    jones_matrices = torch.stack(jones_matrices, dim=0)
    return jones_matrices


# @pytest.mark.parametrize("jones", [generate_jones_matrix() for _ in range(5)])
def test_retardance_equivalence():
    # Test to ensure both methods for calculating retardance are equivalent
    batch_jones_matrices = generate_jones_matrix(batch_size=10)
    jones = batch_jones_matrices
    retardance_su2 = retardance_from_su2(jones)
    retardance_jones = retardance_from_jones(jones, su2_method=False)
    err_msg = "The retardance calculations do not match!"
    assert torch.allclose(retardance_su2, retardance_jones), err_msg
