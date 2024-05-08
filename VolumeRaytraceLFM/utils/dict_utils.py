"""Ultility functions for dictionaries."""
import torch

def extract_numbers_from_dict_of_lists(input_dict):
    """
    Extracts all the unique numeric values from a dictionary.

    Args:
        input_dict (dict): keys as tuples and values as a list of lists

    Returns:
        set: A set of unique numeric values found in the dictionary.
    """
    num_set = set()

    for lists in input_dict.values():
        for sublist in lists:
            num_set.update(sublist)  # Assuming each sublist contains only numbers

    return num_set


def transform_dict_list_to_set(input_dict):
    """
    Transforms dict values from list of lists to sets.
    """
    result = {}
    for key, lists in input_dict.items():
        # Flatten list of lists and convert to set
        result[key] = set(x for sublist in lists for x in sublist)
    return result


def filter_keys_by_count(counter, count_ths):
    """
    Filters keys in a Counter by a count threshold.

    Args:
        counter (Counter): Counter with integer counts.
        count_ths (int): Min count for keys to be included.

    Returns:
        list: Keys with counts >= count_ths.
    """
    filtered_list = [key for key, count in counter.items() 
                    if count > count_ths - 1]
    return sorted(filtered_list)


def idx_dict_to_tensor(idx_dict):
    max_key = max(idx_dict.keys())
    # Initialize all to -1
    idx_tensor = torch.full((max_key + 1,), -1, dtype=torch.long)
    for k, v in idx_dict.items():
        idx_tensor[k] = v
    return idx_tensor
