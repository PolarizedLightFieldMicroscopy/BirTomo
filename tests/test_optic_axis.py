import numpy as np
from VolumeRaytraceLFM.volumes.optic_axis import spherical_to_unit_vector, unit_vector_to_spherical

def test_spherical_to_unit_vector_and_back():
    # Test angles
    theta = np.pi / 4
    phi = np.pi / 6

    # Convert to unit vector and back
    unit_vector = spherical_to_unit_vector(theta, phi)
    theta_back, phi_back = unit_vector_to_spherical(unit_vector)

    # Allow for some numerical tolerance
    assert np.isclose(theta, theta_back, atol=1e-6)
    assert np.isclose(phi, phi_back, atol=1e-6)

def test_unit_vector_to_spherical_and_back():
    # Test unit vector
    vector = np.array([0.5, 0.5, np.sqrt(2)/2])

    # Convert to spherical angles and back
    theta, phi = unit_vector_to_spherical(vector)
    vector_back = spherical_to_unit_vector(theta, phi)

    # Allow for some numerical tolerance
    assert np.allclose(vector, vector_back, atol=1e-6)
