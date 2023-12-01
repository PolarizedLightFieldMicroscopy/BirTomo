import torch
import pytest
from VolumeRaytraceLFM.utils.dimensions_utils import (
    get_region_of_ones_shape,
    crop_3d_tensor,
    reshape_crop_and_flatten_parameter,
    reshape_and_crop,
    store_as_pytorch_parameter
)

def test_get_region_of_ones_shape():
    # Test with a simple case
    mask = torch.tensor([[0, 1], [1, 0]])
    expected_shape = torch.tensor([2, 2])
    assert torch.all(get_region_of_ones_shape(mask) == expected_shape)

    # Test with no ones in the mask
    mask = torch.zeros((2, 2))
    with pytest.raises(ValueError):
        get_region_of_ones_shape(mask)

def test_crop_3d_tensor():
    tensor = torch.randn(4, 4, 4)
    new_shape = (2, 2, 2)
    cropped_tensor = crop_3d_tensor(tensor, new_shape)
    assert cropped_tensor.shape == torch.Size(new_shape)

def test_reshape_crop_and_flatten_parameter():
    flattened_param = torch.randn(4*4*4)
    original_shape = (4, 4, 4)
    new_shape = (2, 2, 2)
    parameter = reshape_crop_and_flatten_parameter(flattened_param, original_shape, new_shape)
    assert parameter.shape == torch.Size([8])

def test_reshape_and_crop():
    flattened_param = torch.randn(4*4*4)
    original_shape = (4, 4, 4)
    new_shape = (2, 2, 2)
    tensor = reshape_and_crop(flattened_param, original_shape, new_shape)
    assert tensor.shape == torch.Size(new_shape)

def test_store_as_pytorch_parameter():
    tensor = torch.tensor([1.0, 2.0, 3.0])
    scalar_param = store_as_pytorch_parameter(tensor, 'scalar')
    vector_param = store_as_pytorch_parameter(tensor, 'vector')

    assert isinstance(scalar_param, torch.nn.Parameter)
    assert scalar_param.shape == torch.Size([3])

    assert isinstance(vector_param, torch.nn.Parameter)
    assert vector_param.shape == torch.Size([3, 1])
