import numpy as np
import torch
from VolumeRaytraceLFM.volumes.optic_axis import spherical_to_unit_vector_np, unit_vector_to_spherical, spherical_to_unit_vector_torch

def test_spherical_to_unit_vector_and_back():
    # Test angles
    theta = np.pi / 4
    phi = np.pi / 6

    # Convert to unit vector and back
    unit_vector = spherical_to_unit_vector_np(theta, phi)
    theta_back, phi_back = unit_vector_to_spherical(unit_vector)

    # Allow for some numerical tolerance
    assert np.isclose(theta, theta_back, atol=1e-6)
    assert np.isclose(phi, phi_back, atol=1e-6)


def test_unit_vector_to_spherical_and_back():
    # Test unit vector
    vector = np.array([0.5, 0.5, np.sqrt(2)/2])

    # Convert to spherical angles and back
    theta, phi = unit_vector_to_spherical(vector)
    vector_back = spherical_to_unit_vector_np(theta, phi)

    # Allow for some numerical tolerance
    assert np.allclose(vector, vector_back, atol=1e-6)


def test_spherical_to_unit_vector():
    # Generate random angles
    theta = np.random.uniform(0, 2 * np.pi, 100)
    phi = np.random.uniform(0, np.pi / 2, 100)

    # Convert to unit vectors using numpy
    unit_vectors_np = np.array([spherical_to_unit_vector_np(t, p) for t, p in zip(theta, phi)])

    # Convert to unit vectors using torch
    angles_torch = torch.tensor(np.stack([theta, phi], axis=-1), dtype=torch.float32)
    unit_vectors_torch = spherical_to_unit_vector_torch(angles_torch).numpy()

    # Compare the results
    assert np.allclose(unit_vectors_np, unit_vectors_torch, atol=1e-6)
