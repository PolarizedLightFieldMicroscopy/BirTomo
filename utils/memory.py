"""Functions for memory management"""

import torch

def tensor_memory_size(tensor):
    """Return the memory size of a tensor in kilobytes."""
    num_elements = tensor.numel()
    # Get the size of each element in bytes
    element_size = tensor.element_size()
    total_memory_bytes = num_elements * element_size
    # Convert bytes to kilobytes
    total_memory_kb = total_memory_bytes / 1024
    return total_memory_kb


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
            print(f'Memory used by {attr_name}: {mem_size} KB')
            total_memory += mem_size
    print(f'Total tensor memory in the instance: {total_memory} KB')
    return total_memory
