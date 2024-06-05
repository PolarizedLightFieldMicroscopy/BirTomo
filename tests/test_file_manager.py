"""Tests for VolumeFileManager class"""

import numpy as np
import h5py
import os
from unittest.mock import Mock
from VolumeRaytraceLFM.file_manager import VolumeFileManager


def mock_h5_file(return_value):
    def mock(*args, **kwargs):
        class MockH5File:
            def __getitem__(self, item):
                return return_value[item]

            def __enter__(self, *args, **kwargs):
                return self

            def __exit__(self, *args, **kwargs):
                pass

        return MockH5File()

    return mock


def test_extract_data_from_h5(monkeypatch):
    test_file_path = "test_file.h5"
    expected_delta_n = np.array([1, 2, 3])
    expected_optic_axis = np.array([4, 5, 6])

    # Mocking the h5py.File call
    monkeypatch.setattr(
        h5py,
        "File",
        mock_h5_file(
            {"data/delta_n": expected_delta_n, "data/optic_axis": expected_optic_axis}
        ),
    )

    vfm = VolumeFileManager()
    delta_n, optic_axis = vfm.extract_data_from_h5(test_file_path)

    assert np.array_equal(delta_n, expected_delta_n)
    assert np.array_equal(optic_axis, expected_optic_axis)


def test_extract_all_data_from_h5(monkeypatch):
    """Verify that the extract_all_data_from_h5 method correctly
    extracts data and optical information from an H5 file"""
    test_file_path = "test_file.h5"
    expected_delta_n = np.array([1, 2, 3])
    expected_optic_axis = np.array([4, 5, 6])
    expected_volume_shape = np.array([7, 8, 9])
    expected_voxel_size_um = np.array([10, 11, 12])

    # Mocking the h5py.File call
    monkeypatch.setattr(
        h5py,
        "File",
        mock_h5_file(
            {
                "data/delta_n": expected_delta_n,
                "data/optic_axis": expected_optic_axis,
                "optical_info/volume_shape": expected_volume_shape,
                "optical_info/voxel_size_um": expected_voxel_size_um,
            }
        ),
    )

    vfm = VolumeFileManager()
    delta_n, optic_axis, volume_shape, voxel_size_um = vfm.extract_all_data_from_h5(
        test_file_path
    )

    assert np.array_equal(delta_n, expected_delta_n)
    assert np.array_equal(optic_axis, expected_optic_axis)
    assert np.array_equal(volume_shape, expected_volume_shape)
    assert np.array_equal(voxel_size_um, expected_voxel_size_um)


def test_save_as_channel_stack_tiff(monkeypatch):
    filename = "test.tiff"
    shape = (3, 1, 5, 5)
    delta_n = np.random.random(shape[1:])
    optic_axis = np.random.random(shape)
    norms = np.linalg.norm(optic_axis, axis=0)
    optic_axis /= norms

    # Create a mock for the imwrite function
    mock_imwrite = Mock()
    # Use monkeypatch to replace imwrite with the mock
    monkeypatch.setattr("tifffile.imwrite", mock_imwrite)

    # Create an instance of VolumeFileManager and call the method
    vfm = VolumeFileManager()
    vfm.save_as_channel_stack_tiff(filename, delta_n, optic_axis)

    assert mock_imwrite.called, "imwrite was not called"

    # Extract the actual arguments with which imwrite was called
    actual_args, _ = mock_imwrite.call_args
    actual_filename, actual_data = actual_args

    assert actual_filename == filename, "Filename does not match"

    # Check if the data matches within a tolerance
    expected_data = np.stack(
        [delta_n, optic_axis[0], optic_axis[1], optic_axis[2]], axis=0
    )
    assert np.allclose(
        actual_data, expected_data
    ), "Data does not match within tolerance"


def test_save_as_h5():
    # Mock data for testing
    mock_h5_file_path = "test_saving_h5_file.h5"
    mock_delta_n = np.array([0.1])
    mock_optic_axis = np.array([0, 1, 0])
    mock_optical_info = {"volume_shape": [1, 1, 1], "voxel_size_um": 1.0}
    mock_description = "Test data for volume file manager"
    mock_optical_all = True

    manager = VolumeFileManager()
    manager.save_as_h5(
        mock_h5_file_path,
        mock_delta_n,
        mock_optic_axis,
        mock_optical_info,
        mock_description,
        mock_optical_all,
    )
    assert os.path.exists(mock_h5_file_path)
    with h5py.File(mock_h5_file_path, "r") as f:
        assert "data" in f and "optical_info" in f
        assert "delta_n" in f["data"] and "optic_axis" in f["data"]
        assert "description" in f["optical_info"]

        # Verify the contents of the datasets
        assert np.array_equal(f["data"]["delta_n"][:], mock_delta_n)
        assert np.array_equal(f["data"]["optic_axis"][:], mock_optic_axis)
        description_bytes = f["optical_info"]["description"][()]
        description_string = description_bytes.decode("utf-8")
        assert description_string == mock_description

        # Verify the presence of additional optical metadata if `optical_all` is True
        if mock_optical_all:
            for key, value in mock_optical_info.items():
                assert key in f["optical_info"]
                assert np.array_equal(f["optical_info"][key][()], value)
    os.remove(mock_h5_file_path)
