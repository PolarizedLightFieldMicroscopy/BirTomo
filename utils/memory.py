"""Functions for memory management"""
import sys
import torch
import numpy as np

def tensor_memory_size(tensor):
    """Return the memory size of a tensor in bytes."""
    num_elements = tensor.numel()
    # Get the size of each element in bytes
    element_size = tensor.element_size()
    total_memory_bytes = num_elements * element_size
    return total_memory_bytes


def calculate_tensor_memory_usage(instance):
    """
    Calculate and print the memory usage of tensor attributes
    for any class instance.
    Args:
        instance: Any class instance to evaluate.
    Returns:
        total_memory: Total memory used by all tensor attributes in bytes.
    """
    total_memory = 0
    for attr_name, attr_value in vars(instance).items():
        if isinstance(attr_value, torch.Tensor):
            mem_size = tensor_memory_size(attr_value)
            print(f'Memory used by {attr_name}: {mem_size} bytes or {mem_size / 1024} KB')
            total_memory += mem_size
    print(f'Total tensor memory in the instance: {total_memory} bytes or {total_memory / 1024} KB')
    return total_memory


def deep_getsizeof(o, ids):
    """Find the memory footprint of a Python object, including the
    contents of its attributes, with special handling for NumPy arrays.
    
    Example usage: deep_getsizeof(obj, set())
    """
    if id(o) in ids:
        return 0
    ids.add(id(o))

    size = sys.getsizeof(o)

    if isinstance(o, dict):
        size += sum(deep_getsizeof(k, ids) + deep_getsizeof(v, ids) for k, v in o.items())
    elif isinstance(o, (list, tuple, set, frozenset)):
        size += sum(deep_getsizeof(item, ids) for item in o)
    elif isinstance(o, np.ndarray):
        # Use the nbytes attribute for numpy arrays to get the actual data size
        size += o.nbytes
    elif hasattr(o, '__dict__'):
        size += deep_getsizeof(o.__dict__, ids)
    elif hasattr(o, '__iter__') and not isinstance(o, (str, bytes, bytearray, np.ndarray)):
        size += sum(deep_getsizeof(item, ids) for item in o)
    return size


def calculate_total_memory(instance):
    """Calculate the total memory usage of all attributes in a class instance."""
    total_memory = 0
    ids = set()
    for attr_name, attr_value in vars(instance).items():
        if isinstance(attr_value, (torch.Tensor, torch.nn.Parameter)):
            mem_size = tensor_memory_size(attr_value)
        else:
            mem_size = deep_getsizeof(attr_value, ids)
        print(f"Memory used by {attr_name}: {mem_size} bytes")
        total_memory += mem_size
    print(f"Total memory used by the instance: {total_memory} bytes")
    return total_memory
