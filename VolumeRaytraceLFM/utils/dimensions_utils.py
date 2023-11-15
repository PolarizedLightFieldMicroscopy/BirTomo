import torch

def get_region_of_ones_shape(mask):
    indices = torch.nonzero(mask)
    min_indices = indices.min(dim=0)[0]
    max_indices = indices.max(dim=0)[0]
    shape = max_indices - min_indices + 1
    return shape
