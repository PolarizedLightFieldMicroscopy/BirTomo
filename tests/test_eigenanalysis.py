"""Tests for the eigenanalysis module of the VolumeRaytraceLFM.jones package."""

import os
import torch
import pytest
from VolumeRaytraceLFM import JonesMatrixGenerators, BackEnds
from VolumeRaytraceLFM.jones.eigenanalysis import (
    retardance_from_jones,
    retardance_from_su2,
    calc_theta,
)


# Additional debugging helper
@pytest.hookimpl(tryfirst=True)
def pytest_exception_interact(node, call, report):
    if report.failed:
        # Only capture output if the test failed
        if node.name == "test_calc_theta_differentiability":
            print("Captured Environment State:")
            print(f"os.environ: {os.environ}")
            print(f"torch.get_rng_state(): {torch.get_rng_state()}")
            print(
                f"torch.cuda.get_rng_state_all(): {torch.cuda.get_rng_state_all() if torch.cuda.is_available() else 'N/A'}"
            )


@pytest.mark.parametrize(
    "grad_output",
    [torch.ones(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)],
)
def test_calc_theta_differentiability(grad_output):
    torch.set_grad_enabled(True)
    jones = torch.eye(2, dtype=torch.complex64).unsqueeze(0).requires_grad_(True)
    theta = calc_theta(jones)

    print(f"jones.requires_grad: {jones.requires_grad}")
    print(f"theta.requires_grad: {theta.requires_grad}")
    print(f"theta.grad_fn: {theta.grad_fn}")
    print(f"Initial jones.grad: {jones.grad}")

    # Check if gradients can be computed
    try:
        theta.backward(grad_output)

        print(f"Computed jones.grad: {jones.grad}")

        assert jones.grad is not None, "Gradients are None"
        assert not torch.isnan(jones.grad).any(), "Gradients contain NaNs"
        assert not torch.isinf(jones.grad).any(), "Gradients contain Infs"
    except RuntimeError as e:
        pytest.fail(f"Backward pass failed with error: {e}")


def generate_jones_matrix(batch_size=10):
    ret = torch.rand(batch_size) * 2 * torch.pi  # Tensor of shape [batch_size]
    azim = torch.rand(batch_size) * 2 * torch.pi  # Tensor of shape [batch_size]
    # Generate the batch of Jones matrices using the linear_retarder method
    jones_matrices = []
    for i in range(batch_size):
        jones_matrix = JonesMatrixGenerators.linear_retarder(
            ret[i], azim[i], backend=BackEnds.PYTORCH
        )
        jones_matrices.append(jones_matrix)
    jones_matrices = torch.stack(jones_matrices, dim=0)
    return jones_matrices


def test_retardance_equivalence():
    # Test to ensure both methods for calculating retardance are equivalent
    batch_jones_matrices = generate_jones_matrix(batch_size=10)
    jones = batch_jones_matrices
    retardance_su2 = retardance_from_su2(jones)
    retardance_jones = retardance_from_jones(jones, su2_method=False)
    err_msg = "The retardance calculations do not match!"
    assert torch.allclose(retardance_su2, retardance_jones), err_msg


def test_calc_theta_identity():
    jones = torch.eye(2).unsqueeze(0)
    theta = calc_theta(jones)
    err_msg = f"Expected theta to be 0, but got {theta}"
    assert torch.allclose(theta, torch.zeros_like(theta), atol=1e-7), err_msg


def test_retardance_of_identity():
    # Test to ensure that the retardance of the identity matrix is zero
    identity_matrix = torch.eye(2).unsqueeze(0)  # .repeat(2, 1, 1)
    # identity_matrix = torch.eye(2, dtype=torch.complex128).unsqueeze(0)
    retardance = retardance_from_su2(identity_matrix)
    err_msg = "Retardance of an identity Jones matrix is not zero."
    assert torch.allclose(retardance, torch.zeros_like(retardance), atol=3e-8), err_msg
