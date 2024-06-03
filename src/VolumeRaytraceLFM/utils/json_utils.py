import json
import numpy as np

"""Example of encoding complex numbers and numpy arrays to JSON.
# Example numpy array of complex numbers
complex_array = np.array([[0.5+0j, 0.5j], [-0.5j, 0.5+0j]])

# Encode the numpy array
json_data = json.dumps(complex_array, cls=ComplexArrayEncoder)
print(json_data)

# Decode the JSON data back into a numpy array
decoded_array = json_to_complex_array(json_data)
print(decoded_array)
"""

class ComplexArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        # Let numpy arrays be handled by converting them to lists
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



def decode_complex_list(dct):
    if "real" in dct and "imag" in dct:
        return complex(dct["real"], dct["imag"])
    return dct


def json_to_complex_array(json_data):
    data = json.loads(json_data, object_hook=decode_complex_list)
    # Convert lists (possibly containing complex numbers) back to a numpy array
    return np.array(data)
