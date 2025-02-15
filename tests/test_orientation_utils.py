import pytest
import torch
import numpy as np
from VolumeRaytraceLFM.utils.orientation_utils import transpose_and_flip, undo_transpose_and_flip


@pytest.mark.parametrize("data, expected", [
    (np.array([[1, 2, 3], [4, 5, 6]]), np.array([[4, 1], [5, 2], [6, 3]])),
    (torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[4, 1], [5, 2], [6, 3]])),
])
def test_transpose_and_flip(data, expected):
    """Test that the transpose_and_flip function correctly transposes and flips the input data."""
    result = transpose_and_flip(data)
    if isinstance(data, np.ndarray):
        np.testing.assert_array_equal(result, expected, err_msg="The numpy array was not transposed and flipped correctly.")
    elif isinstance(data, torch.Tensor):
        torch.testing.assert_close(result, expected)

@pytest.mark.parametrize("data", [
    np.random.random((3, 3)),
    torch.randn(3, 3),
])
def test_undo_transpose_and_flip(data):
    """Test that the undo_transpose_and_flip function correctly undoes
    the transpose_and_flip function."""
    transformed = transpose_and_flip(data)
    reverted = undo_transpose_and_flip(transformed)
    if isinstance(data, np.ndarray):
        np.testing.assert_array_equal(data, reverted)
    elif isinstance(data, torch.Tensor):
        torch.testing.assert_close(data, reverted)
