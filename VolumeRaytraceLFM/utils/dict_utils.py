"""Ultility functions for dictionaries."""

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
