import numpy as np
import h5py
import tifffile


class VolumeFileManager:
    def __init__(self):
        """Initializes the VolumeFileManager class."""
        pass

    def extract_data_from_h5(self, file_path):
        """
        Extracts birefringence (delta_n) and optic axis data from an H5 file.

        Args:
        - file_path (str): Path to the H5 file from which data is to be extracted.

        Returns:
        - tuple: A tuple containing numpy arrays for delta_n and optic_axis.
        """
        volume_file = h5py.File(file_path, "r")
        delta_n = np.array(volume_file["data/delta_n"])
        optic_axis = np.array(volume_file["data/optic_axis"])

        return delta_n, optic_axis

    def extract_all_data_from_h5(self, file_path):
        """
        Extracts birefringence (delta_n), optic axis data, and optical information from an H5 file.

        Args:
        - file_path (str): Path to the H5 file from which data is to be extracted.

        Returns:
        - tuple: A tuple containing numpy arrays for
                    delta_n, optic_axis, volume_shape, and voxel_size_um.
        """
        volume_file = h5py.File(file_path, "r")

        # Fetch birefringence and optic axis
        delta_n = np.array(volume_file["data/delta_n"])
        optic_axis = np.array(volume_file["data/optic_axis"])

        # Fetch optical info
        volume_shape = np.array(volume_file["optical_info/volume_shape"])
        voxel_size_um = np.array(volume_file["optical_info/voxel_size_um"])

        return delta_n, optic_axis, volume_shape, voxel_size_um

    def save_as_channel_stack_tiff(self, filename, delta_n, optic_axis):
        """
        Saves the provided volume data as a multi-channel TIFF file.

        Args:
        - filename (str): The file path where the TIFF file will be saved.
        - delta_n (np.ndarray): Numpy array containing the birefringence information of the volume.
        - optic_axis (np.ndarray): Numpy array containing the optic axis data of the volume.

        The method combines delta_n and optic_axis data into a single multi-channel array
        and saves it as a TIFF file. Exceptions related to file operations are caught and logged.
        """
        try:
            print(f"Saving volume to file: {filename}")
            combined_data = np.stack(
                [delta_n, optic_axis[0], optic_axis[1], optic_axis[2]], axis=0
            )
            tifffile.imwrite(filename, combined_data)
            print("Volume saved successfully.")
        except Exception as e:
            print(f"Error saving file: {e}")

    def save_as_h5(
        self, h5_file_path, delta_n, optic_axis, optical_info, description, optical_all
    ):
        """
        Saves the volume data, including birefringence information (delta_n) and optic axis data,
        along with optical metadata into an H5 file.

        The method creates an H5 file at the specified path and writes the provided data
        to this file, organizing the data into appropriate groups and datasets within the file.

        Args:
        - h5_file_path (str): The file path where the H5 file will be saved.
        - delta_n (np.ndarray): Numpy array containing the birefringence information of the volume.
        - optic_axis (np.ndarray): Numpy array containing the optic axis data of the volume.
        - optical_info (dict): Dictionary containing optical metadata about the volume. This may
          include properties like volume shape, voxel size, etc.
        - description (str): A brief description or note to be included in the optical information
          of the H5 file. Useful for providing context or additional details about the data.
        - optical_all (bool): A flag indicating whether to save all optical metadata present in
          `optical_info` to the H5 file. If False, only specific predefined metadata (like volume
          shape and voxel size) will be saved.

        Returns:
        None. The result of this method is the creation of an H5 file with the specified data.
        """
        with h5py.File(h5_file_path, "w") as f:
            self._save_optical_info(f, optical_info, description, optical_all)
            self._save_data(f, delta_n, optic_axis)

    def _save_optical_info(self, file_handle, optical_info, description, optical_all):
        """
        Private method to save optical information to an H5 file.

        Args:
        - file_handle (File): An open H5 file handle.
        - optical_info (dict): Dictionary containing optical metadata.
        - description (str): Description to be included in the H5 file.
        - optical_all (bool): Flag indicating whether to save all optical metadata.

        This method creates a group for optical information and adds datasets to it.
        """
        optics_grp = file_handle.create_group("optical_info")
        optics_grp.create_dataset("description", data=np.bytes_(description))
        if not optical_all:
            vol_shape = optical_info.get("volume_shape", None)
            voxel_size_um = optical_info.get("voxel_size_um", None)
            if vol_shape is not None:
                optics_grp.create_dataset("volume_shape", data=np.array(vol_shape))
            if voxel_size_um is not None:
                optics_grp.create_dataset("voxel_size_um", data=np.array(voxel_size_um))
        else:
            for k, v in optical_info.items():
                optics_grp.create_dataset(k, data=np.array(v))

    def _save_data(self, file_handle, delta_n, optic_axis):
        """
        Private method to save delta_n and optic_axis data to an H5 file.

        Args:
        - file_handle (File): An open H5 file handle.
        - delta_n (np.ndarray): Numpy array of delta_n data.
        - optic_axis (np.ndarray): Numpy array of optic_axis data.

        This method creates a group for volume data and adds datasets for delta_n and optic_axis.
        """
        data_grp = file_handle.create_group("data")
        data_grp.create_dataset("delta_n", delta_n.shape, data=delta_n)
        data_grp.create_dataset("optic_axis", optic_axis.shape, data=optic_axis)

    def save_as_npz(self, filename, delta_n, optic_axis):
        """
        Saves the provided volume data as a NumPy NPZ file.

        Args:
        - filename (str): The file path where the NPZ file will be saved.
        - delta_n (np.ndarray): Numpy array containing the birefringence information of the volume.
        - optic_axis (np.ndarray): Numpy array containing the optic axis data of the volume.
        """
        try:
            print(f"Saving volume to file: {filename}")
            np.savez(filename, birefringence=delta_n, optic_axis=optic_axis)
            print("Volume saved successfully as numpy arrays.")
        except Exception as e:
            print(f"Error saving file: {e}")
