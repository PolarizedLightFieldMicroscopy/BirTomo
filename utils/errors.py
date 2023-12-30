import numpy as np


def compare_dicts(dict1, dict2):
    # Check if both dictionaries have the same keys
    if dict1.keys() != dict2.keys():
        raise ValueError

    # Loop through each key in the dictionary
    for key in dict1:
        if isinstance(dict1[key], np.ndarray) or isinstance(dict2[key], np.ndarray):
            if not np.array_equal(dict1[key], dict2[key]):
                raise ValueError
        else:
            raise ValueError
    return
