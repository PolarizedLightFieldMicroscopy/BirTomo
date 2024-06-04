"""Abstract classes"""

# Third party libraries imports
from enum import Enum
import pickle
from os.path import exists
import copy
import numpy as np
import matplotlib.pyplot as plt
import itertools
from VolumeRaytraceLFM.visualization.plotting_rays import plot_ray_angles

# Packages needed for siddon algorithm calculations
from VolumeRaytraceLFM.my_siddon import (
    siddon_params,
    siddon_midpoints,
    vox_indices,
    siddon_lengths,
)

# Optional imports: as the classes here depend on OpticBlock, we create a dummy
#   that will work with only numpy
try:
    import torch
    import torch.nn as nn
    from VolumeRaytraceLFM.optic_config import *
except:

    class OpticBlock:  # Dummy function for numpy
        def __init__(
            self,
            optic_config=None,
            members_to_learn=None,
        ):
            pass


DEBUG = False


class SimulType(Enum):
    """Defines which types of volumes we can have, as each type has a
    different ray-voxel interaction"""

    NOT_SPECIFIED = 0
    FLUOR_INTENS = (
        1  # Voxels add intensity as the ray goes trough the volume: commutative
    )
    BIREFRINGENT = (
        2  # Voxels modify polarization as the ray goes trough: non commutative
    )
    # FLUOR_POLAR     = 3     #
    # DIPOLES         = 4     # Voxels add light depending on their angle with respect to the dipole
    # etc. attenuators and di-attenuators (polarization dependent)


class BackEnds(Enum):
    """Defines type of backend (numpy,pytorch,etc)"""

    NUMPY = 1  # Use numpy back-end
    PYTORCH = 2  # Use Pytorch, with auto-differentiation and GPU support.


class OpticalElement(OpticBlock):
    """Abstract class defining a elements, with a back-end ans some optical information"""

    default_optical_info = {
        # Volume information
        "volume_shape": 3 * [1],
        "axial_voxel_size_um": 1.0,
        "cube_voxels": True,
        # Microlens array information
        "pixels_per_ml": 17,
        "n_micro_lenses": 1,
        "n_voxels_per_ml": 1,
        # Objective lens information
        "M_obj": 60,
        "na_obj": 1.2,
        "n_medium": 1.35,
        "wavelength": 0.550,
        "camera_pix_pitch": 6.5,
        # Polarization information
        "polarizer": np.array([[1, 0], [0, 1]]),
        "analyzer": np.array([[1, 0], [0, 1]]),
        "polarizer_swing": 0.03,
    }

    def __init__(
        self, backend: BackEnds = BackEnds.NUMPY, torch_args={}, optical_info={}
    ):
        # torch args could be {'optic_config' : None, 'members_to_learn' : []},

        # Optical info is needed
        assert (
            len(optical_info) > 0
        ), f"Optical info (optical_info) dictionary needed: \
                        use OpticalElement.default_optical_info as reference \
                        {OpticalElement.default_optical_info}"
        # Compute voxel size
        if optical_info["cube_voxels"] is False:
            optical_info["voxel_size_um"] = [
                optical_info["axial_voxel_size_um"],
            ] + 2 * [
                optical_info["pixels_per_ml"]
                * optical_info["camera_pix_pitch"]
                / optical_info["M_obj"]
                / optical_info["n_voxels_per_ml"]
            ]
        else:
            # Option to make voxel size uniform
            optical_info["voxel_size_um"] = 3 * [
                optical_info["pixels_per_ml"]
                * optical_info["camera_pix_pitch"]
                / optical_info["M_obj"]
                / optical_info["n_voxels_per_ml"]
            ]
        # Check if back-end is torch and overwrite self with an optic block, for Waveblocks
        # compatibility.
        if backend == BackEnds.PYTORCH:
            # We need to make a copy if we don't want to modify the torch_args default argument,
            # very weird.
            new_torch_args = copy.deepcopy(torch_args)
            # If no optic_config is provided, create one
            if "optic_config" not in torch_args.keys() or (
                "optic_config" not in torch_args.keys()
                and not isinstance(torch_args["optic_config"], OpticConfig)
            ):
                new_torch_args["optic_config"] = OpticConfig()
                new_torch_args["optic_config"].volume_config.volume_shape = (
                    optical_info["volume_shape"]
                )
                new_torch_args["optic_config"].volume_config.voxel_size_um = (
                    optical_info["voxel_size_um"]
                )
                new_torch_args["optic_config"].mla_config.n_pixels_per_mla = (
                    optical_info["pixels_per_ml"]
                )
                new_torch_args["optic_config"].mla_config.n_micro_lenses = optical_info[
                    "n_micro_lenses"
                ]
                new_torch_args["optic_config"].PSF_config.NA = optical_info["na_obj"]
                new_torch_args["optic_config"].PSF_config.ni = optical_info["n_medium"]
                new_torch_args["optic_config"].PSF_config.wvl = optical_info[
                    "wavelength"
                ]
                try:
                    new_torch_args["optic_config"].pol_config.polarizer = optical_info[
                        "polarizer"
                    ]
                    new_torch_args["optic_config"].pol_config.analyzer = optical_info[
                        "analyzer"
                    ]
                except:
                    print("Error: Polarizer and Analyzer not found in optical_info")
            super(OpticalElement, self).__init__(
                optic_config=new_torch_args["optic_config"],
                members_to_learn=(
                    new_torch_args["members_to_learn"]
                    if "members_to_learn" in new_torch_args.keys()
                    else []
                ),
            )
        # Store variables
        self.backend = backend
        self.simul_type = SimulType.NOT_SPECIFIED
        self.optical_info = optical_info

    @staticmethod
    def get_optical_info_template():
        return copy.deepcopy(OpticalElement.default_optical_info)


###########################################################################################
class RayTraceLFM(OpticalElement):
    """This is a base class that takes a volume geometry and LFM geometry and
    calculates which arrive to each of the pixels behind each microlens, and
    discards the rest. This class also pre-computes how each rays traverses the
    volume with the Siddon algorithm. The interaction between the voxels and
    the rays is defined by each instance of this class.
    """

    def __init__(
        self,
        backend: BackEnds = BackEnds.NUMPY,
        torch_args={},
        optical_info={
            "volume_shape": [11, 11, 11],
            "voxel_size_um": 3 * [1.0],
            "pixels_per_ml": 17,
            "na_obj": 1.2,
            "n_medium": 1.52,
            "wavelength": 0.550,
            "n_micro_lenses": 1,
            "n_voxels_per_ml": 1,
        },
    ):
        # Initialize the OpticalElement class
        super(RayTraceLFM, self).__init__(
            backend=backend, torch_args=torch_args, optical_info=optical_info
        )

        # Create dummy variables for pre-computed rays and paths through the volume.
        # Many of these are defined in compute_rays_geometry.
        self.ray_valid_indices = None
        self.ray_vol_colli_indices = None
        self.ray_vol_colli_lengths = None
        self.ray_direction_basis = None
        self.ray_valid_direction = None
        self.ray_valid_indices_by_ray_num = None
        self.vox_ctr_idx = None
        self.volume_ctr_um = None
        self.lateral_ray_length_from_center = 0
        self.voxel_span_per_ml = 0
        self.vol_shape_restricted = None
        self.use_lenslet_based_filtering = True

        self.nonzero_pixels_dict = self._create_default_nonzero_pixels_dict(
            optical_info["n_micro_lenses"], optical_info["pixels_per_ml"]
        )

    def forward(self, volume_in):
        """Perform the forward model calculations."""
        # Check if type of volume is the same as input volume, if one is provided
        # if volume_in is not None:
        #     assert volume_in.simul_type == self.simul_type, f"Error: wrong type of volume \
        #           provided, this ray-tracer works for {self.simul_type} and a volume \
        #           {volume_in.simul_type} was provided"
        return self.ray_trace_through_volume(volume_in)

    def _create_default_nonzero_pixels_dict(self, num_mla, num_pixels):
        """Creates a dictionary that stores the mask for which rays lead to
        nonzero pixels. This dictionary is a placeholder for the actual mask."""
        default_value = lambda: np.ones((num_pixels, num_pixels), dtype=bool)
        # Generate all (i,j) pairs using itertools.product
        iter_product = itertools.product(range(num_mla), repeat=2)
        nonzero_pixels_dict = {(i, j): default_value() for i, j in iter_product}
        return nonzero_pixels_dict

    ###########################################################################################
    # Helper functions
    @staticmethod
    def ravel_index(x, dims):
        """Method used for debugging"""
        if x[0] >= dims[0] and x[1] >= dims[1] and x[2] >= dims[2]:
            print("here")
        c = np.cumprod([1] + dims[::-1])[:-1][::-1]
        return np.dot(c, x)

    @staticmethod
    def safe_ravel_index(vox, microlens_offset, volume_shape):
        x, y, z = vox[0], vox[1] + microlens_offset[0], vox[2] + microlens_offset[1]
        assert x >= 0 and y >= 0 and z >= 0, "Negative index detected"
        return RayTraceLFM.ravel_index((x, y, z), volume_shape)

    @staticmethod
    def rotation_matrix(axis, angle):
        """Generates the rotation matrix that will rotate a 3D vector
        around "axis" by "angle" counterclockwise."""
        ax, ay, az = axis[0], axis[1], axis[2]
        s = np.sin(angle)
        c = np.cos(angle)
        u = 1 - c
        R_tuple = (
            (ax * ax * u + c, ax * ay * u - az * s, ax * az * u + ay * s),
            (ay * ax * u + az * s, ay * ay * u + c, ay * az * u - ax * s),
            (az * ax * u - ay * s, az * ay * u + ax * s, az * az * u + c),
        )
        R = np.asarray(R_tuple)
        return R

    @staticmethod
    def find_orthogonal_vec(v1, v2):
        """v1 and v2 are numpy arrays (3d vectors)
        This function accommodates for a divide by zero error."""
        value = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # Check if vectors are parallel or anti-parallel
        if np.linalg.norm(value) == 1:
            if v1[1] == 0:
                normal_vec = np.array([0, 1, 0])
            elif v1[2] == 0:
                normal_vec = np.array([0, 0, 1])
            elif v1[0] == 0:
                normal_vec = np.array([1, 0, 0])
            else:
                non_par_vec = np.array([1, 0, 0])
                normal_vec = np.cross(v1, non_par_vec) / np.linalg.norm(
                    np.cross(v1, non_par_vec)
                )
        else:
            normal_vec = np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2))
        return normal_vec

    @staticmethod
    def rotation_matrix_torch(axis, angle):
        """Generates the rotation matrix that will rotate a 3D vector
        around "axis" by "angle" counterclockwise."""
        ax, ay, az = axis[:, 0], axis[:, 1], axis[:, 2]
        s = torch.sin(angle)
        c = torch.cos(angle)
        u = 1 - c
        R = torch.zeros([angle.shape[0], 3, 3], device=axis.device)
        R[:, 0, 0] = ax * ax * u + c
        R[:, 0, 1] = ax * ay * u - az * s
        R[:, 0, 2] = ax * az * u + ay * s
        R[:, 1, 0] = ay * ax * u + az * s
        R[:, 1, 1] = ay * ay * u + c
        R[:, 1, 2] = ay * az * u - ax * s
        R[:, 2, 0] = az * ax * u - ay * s
        R[:, 2, 1] = az * ay * u + ax * s
        R[:, 2, 2] = az * az * u + c
        return R

    @staticmethod
    def find_orthogonal_vec_torch(v1, v2):
        """v1 and v2 are numpy arrays (3d vectors)
        This function accomodates for a divide by zero error."""
        value = (
            torch.linalg.multi_dot((v1, v2))
            / (torch.linalg.norm(v1.unsqueeze(2), dim=1) * torch.linalg.norm(v2))[0]
        )
        # Check if vectors are parallel or anti-parallel
        normal_vec = torch.zeros_like(v1)

        # Search for invalid indices
        invalid_indices = torch.isclose(
            value.abs(), torch.ones([1], device=value.device)
        )
        valid_indices = ~invalid_indices
        # Compute the invalid normal_vectors
        if invalid_indices.sum():
            for n_axis in range(3):
                normal_vec[invalid_indices, n_axis] = (
                    v1[invalid_indices, n_axis] == 0
                ) * 1.0
                # Turn off fixed indices
                invalid_indices[v1[:, n_axis] == 0] = False
            if invalid_indices.sum():  # treat remaning ones
                non_par_vec = (
                    torch.tensor([1.0, 0, 0], device=value.device)
                    .unsqueeze(0)
                    .repeat(v1.shape[0], 1)
                )
                C = torch.linalg.cross(
                    v1[invalid_indices, :], non_par_vec[invalid_indices, :]
                )
                normal_vec[invalid_indices, :] = C / torch.linalg.norm(C, dim=1)

        # Compute the valid normal_vectors
        normal_vec[valid_indices] = torch.linalg.cross(
            v1[valid_indices], v2.unsqueeze(0).repeat(v1.shape[0], 1)[valid_indices]
        ) / torch.linalg.norm(
            torch.linalg.cross(
                v1[valid_indices], v2.unsqueeze(0).repeat(v1.shape[0], 1)[valid_indices]
            ).unsqueeze(2),
            dim=1,
        )
        return normal_vec

    @staticmethod
    def calc_ray_direction(ray):
        """
        Allows to the calculations to be done in ray-space coordinates
        as oppossed to laboratory coordinates
        Parameters:
            ray (np.array): normalized 3D vector giving the direction
                            of the light ray
        Returns:
            ray (np.array): same as input
            ray_perp1 (np.array): normalized 3D vector
            ray_perp2 (np.array): normalized 3D vector
        """
        # in case ray is not a unit vector <- does not need to be normalized
        # ray = ray / np.linalg.norm(ray)
        theta = np.arccos(np.dot(ray, np.array([1, 0, 0])))
        # Unit vectors that give the laboratory axes, can be changed
        scope_axis = np.array([1, 0, 0])
        scope_perp1 = np.array([0, 1, 0])
        scope_perp2 = np.array([0, 0, 1])
        theta = np.arccos(np.dot(ray, scope_axis))
        # print(f"Rotating by {np.around(np.rad2deg(theta), decimals=0)} degrees")
        normal_vec = RayTraceLFM.find_orthogonal_vec(ray, scope_axis)
        Rinv = RayTraceLFM.rotation_matrix(normal_vec, -theta)
        # Extracting basis vectors that are orthogonal to the ray and will be parallel
        # to the laboratory axes that are not the optic axis after a rotation.
        # Note: If scope_perp1 if the y-axis, then ray_perp1 if the 2nd column of Rinv.
        ray_perp1 = np.dot(Rinv, scope_perp1)
        ray_perp2 = np.dot(Rinv, scope_perp2)
        return [ray, ray_perp1, ray_perp2]

    @staticmethod
    def calc_ray_direction_torch(ray_in):
        """
        Allows to the calculations to be done in ray-space coordinates
        as oppossed to laboratory coordinates
        Parameters:
            ray_in [n_rays,3] (torch.array): normalized 3D vector giving the direction
                            of the light ray
        Returns:
            ray (torch.array): same as input
            ray_perp1 (torch.array): normalized 3D vector
            ray_perp2 (torch.array): normalized 3D vector
        """
        if not torch.is_tensor(ray_in):
            ray = torch.from_numpy(ray_in)
        else:
            ray = ray_in
        theta = torch.arccos(
            torch.linalg.multi_dot(
                (ray, torch.tensor([1.0, 0, 0], dtype=ray.dtype, device=ray_in.device))
            )
        )
        # Unit vectors that give the laboratory axes, can be changed
        scope_axis = torch.tensor([1.0, 0, 0], dtype=ray.dtype, device=ray_in.device)
        scope_perp1 = torch.tensor([0, 1.0, 0], dtype=ray.dtype, device=ray_in.device)
        scope_perp2 = torch.tensor([0, 0, 1.0], dtype=ray.dtype, device=ray_in.device)
        # print(f"Rotating by {np.around(torch.rad2deg(theta).numpy(), decimals=0)} degrees")
        normal_vec = RayTraceLFM.find_orthogonal_vec_torch(ray, scope_axis)
        Rinv = RayTraceLFM.rotation_matrix_torch(normal_vec, -theta)
        # Extracting basis vectors that are orthogonal to the ray and will be parallel
        # to the laboratory axes that are not the optic axis after a rotation.
        # Note: If scope_perp1 if the y-axis, then ray_perp1 if the 2nd column of Rinv.
        if scope_perp1[0] == 0 and scope_perp1[1] == 1 and scope_perp1[2] == 0:
            ray_perp1 = Rinv[:, :, 1]  # dot product needed
        else:
            # todo: we need to put a for loop to do this operation
            # ray_perp1 = torch.linalg.multi_dot((Rinv, scope_perp1))
            raise NotImplementedError
        if scope_perp2[0] == 0 and scope_perp2[1] == 0 and scope_perp2[2] == 1:
            ray_perp2 = Rinv[:, :, 2]
        else:
            # todo: we need to put a for loop to do this operation
            # ray_perp2 = torch.linalg.multi_dot((Rinv, scope_perp2))
            raise NotImplementedError

        # Returns a list size 3, where each element is a torch tensor shaped [n_rays, 3]
        return torch.cat(
            [ray.unsqueeze(0), ray_perp1.unsqueeze(0), ray_perp2.unsqueeze(0)], 0
        )

    ###########################################################################################
    # Ray-tracing functions
    @staticmethod
    def rays_through_vol(pixels_per_ml, naObj, nMedium, volume_ctr_um):
        """Identifies the rays that pass through the volume and the central lenslet
        Args:
            pixels_per_ml (int): number of pixels per microlens in one direction,
                                    preferrable to be odd integer so there is a central
                                    pixel behind each lenslet
            naObj (float): numerical aperture of the objective lens
            nMedium (float): refractive index of the volume
            volume_ctr_um (np.array): 3D vector containing the coordinates of the center of the
                                volume in volume space units (um)
        Returns:
            ray_enter (np.array): (3, X, X) array where (3, i, j) gives the coordinates
                                    within the volume ray entrance plane for which the
                                    ray that is incident on the (i, j) pixel with intersect
            ray_exit (np.array): (3, X, X) array where (3, i, j) gives the coordinates
                                    within the volume ray exit plane for which the
                                    ray that is incident on the (i, j) pixel with intersect
            ray_diff (np.array): (3, X, X) array giving the direction of the rays through
                                    the volume
        """
        # Units are in pixel indicies, referring to the pixel that is centered up 0.5 units
        #   Ex: if ml_ctr = [8, 8], then the spatial center pixel is at [8.5, 8.5]
        ml_ctr = [(pixels_per_ml - 1) / 2, (pixels_per_ml - 1) / 2]
        ml_radius = 7.5  # pixels_per_ml / 2
        i = np.linspace(0, pixels_per_ml - 1, pixels_per_ml)
        j = np.linspace(0, pixels_per_ml - 1, pixels_per_ml)
        jv, iv = np.meshgrid(i, j)
        dist_from_ctr = np.sqrt((iv - ml_ctr[0]) ** 2 + (jv - ml_ctr[1]) ** 2)

        # Angles that reach the pixels
        cam_pixels_azim = np.arctan2(jv - ml_ctr[1], iv - ml_ctr[0])
        cam_pixels_azim[dist_from_ctr > ml_radius] = np.NaN
        dist_from_ctr[dist_from_ctr > ml_radius] = np.NaN
        cam_pixels_tilt = np.arcsin(dist_from_ctr / ml_radius * naObj / nMedium)

        # Plotting
        if DEBUG:
            plot_ray_angles(dist_from_ctr, cam_pixels_azim, cam_pixels_tilt)

        # Positions of the ray in volume coordinates
        # assuming rays pass through the center voxel
        ray_enter_x = np.zeros([pixels_per_ml, pixels_per_ml])
        ray_enter_y = (
            volume_ctr_um[0] * np.tan(cam_pixels_tilt) * np.sin(cam_pixels_azim)
            + volume_ctr_um[1]
        )
        ray_enter_z = (
            volume_ctr_um[0] * np.tan(cam_pixels_tilt) * np.cos(cam_pixels_azim)
            + volume_ctr_um[2]
        )
        ray_enter_x[np.isnan(ray_enter_y)] = np.NaN
        ray_enter = np.array([ray_enter_x, ray_enter_y, ray_enter_z])
        vol_ctr_grid_tmp = np.array(
            [
                np.full((pixels_per_ml, pixels_per_ml), volume_ctr_um[i])
                for i in range(3)
            ]
        )
        ray_exit = ray_enter + 2 * (vol_ctr_grid_tmp - ray_enter)

        # Direction of the rays at the exit plane
        ray_diff = ray_exit - ray_enter
        ray_diff = ray_diff / np.linalg.norm(ray_diff, axis=0)
        return ray_enter, ray_exit, ray_diff

    def compute_rays_geometry(
        self, filename=None, image=None, apply_filter_to_rays=False
    ):
        """Computes the ray-voxel collision based on the Siddon algorithm.
        Requires:
            calling self.rays_through_volumes to compute ray entry, exit and directions.
        Args:
            filename (str) optional: Saves the geometry to a pickle file,
                and loads the geometry from a file if the file exists.
            image (np.array) optional:
                The image used to create a mask for the rays.
            apply_filter_to_rays (bool) optional: Whether to apply the mask to
                the rays. Only works when using a 1x1 microlens array.
        Returns:
            self: The RayTraceLFM instance with updated geometry.
        Computes:
            self.vox_ctr_idx (np.array [3]):
                3D index of the central voxel.
            self.volume_ctr_um (np.array [3]):
                3D coordinate in um of the central voxel
            self.ray_valid_indices_by_ray_num (list of tuples n_rays*[(i,j),]):
                Store the 2D ray index of a valid ray (without nan in entry/exit)
            self.ray_valid_indices: array of shape (2, n_valid_rays):
                Stores the 2D indices of the valid rays
            self.ray_vol_colli_indices (list of list of tuples n_valid_rays*[(z,y,x),(z,y,x)]):
                Stores the coordinates of the voxels that the ray n collides with.
            self.ray_vol_colli_lengths (list of list of floats n_valid_rays*[ell1,ell2]):
                Stores the length of traversal of ray n through the voxels inside
                ray_vol_colli_indices.
            self.ray_valid_direction  (list [n_valid_rays, 3]):
                Stores the direction of ray n.
            self.lateral_ray_length_from_center (float):
                Maximum lateral reach of a ray from the center voxel.
            self.voxel_span_per_ml (float):
                Maximum lateral reach of a ray from the center voxel, rounded up.
            self.ray_direction_basis (list [3]):
                List size 3, where each element is a torch tensor shaped [n_rays, 3]
            self.nonzero_pixels_dict (dict):
                Mask for which rays lead to nonzero pixels for each lenslet.
                Computed if an image is provided.
        """
        # If a filename is provided, check if it exists and load the whole ray tracer class from it.
        if self._load_geometry_from_file(filename):
            return self

        # Identify the endpoints of the rays
        ray_enter, ray_exit, ray_diff = self._initialize_ray_geometry()

        # The maximum voxel-span is with respect to the middle voxel,
        #   let's shift that to the origin
        lat_length, vox_span = RayTraceLFM.compute_lateral_ray_length_and_voxel_span(
            ray_diff, self.optical_info["volume_shape"][0]
        )
        self.lateral_ray_length_from_center = lat_length
        self.voxel_span_per_ml = vox_span

        # DEBUG: checking indexing
        use_full_volume = False
        if use_full_volume:
            vol_shape_for_raytracing = self.optical_info["volume_shape"]
        else:
            vol_shape_for_raytracing = self.vol_shape_restricted
        (
            ray_valid_indices_by_ray_num,
            ray_vol_colli_indices,
            ray_vol_colli_lengths,
            ray_valid_direction,
        ) = self.compute_ray_collisions(
            ray_enter,
            ray_exit,
            self.optical_info["voxel_size_um"],
            vol_shape_for_raytracing,
        )

        # ray_valid_indices_by_ray_num gives pixel indices of the given ray number
        self.ray_valid_indices_by_ray_num = ray_valid_indices_by_ray_num

        # Maximum number of ray-voxel interactions, to define
        max_ray_voxels_collision = np.max([len(D) for D in ray_vol_colli_indices])

        n_valid_rays = len(ray_valid_indices_by_ray_num)

        # Initialize storage for ray valid indices
        if self.backend == BackEnds.NUMPY:
            self.ray_valid_indices = np.zeros((2, n_valid_rays), dtype=int)
        elif self.backend == BackEnds.PYTORCH:
            self.ray_valid_indices = torch.zeros(2, n_valid_rays, dtype=int)
        else:
            raise ValueError("Backend not recognized.")

        # Populate the ray valid indices array
        for ix, pixel_pos in enumerate(ray_valid_indices_by_ray_num):
            self.ray_valid_indices[0, ix] = pixel_pos[0]
            self.ray_valid_indices[1, ix] = pixel_pos[1]

        self._filter_invalid_rays(
            max_ray_voxels_collision, ray_vol_colli_lengths, ray_valid_direction
        )

        # TODO: check if collision indices should be filtered too
        # Collisions indices does not get filtered
        self.ray_vol_colli_indices = ray_vol_colli_indices

        if image is not None:
            if not np.any(image):
                raise ValueError("The image cannot be all zeros.")
            if apply_filter_to_rays:
                self.filter_rays_based_on_pixels(image)
            self.nonzero_pixels_dict = self.identify_rays_from_pixels_mla(
                image, ray_valid_indices=self.ray_valid_indices
            )

        # Update volume shape information to account for the whole workspace
        self._update_volume_shape_info()

        # Calculate ray directions basis and stores in self.ray_direction_basis
        self._calculate_ray_directions()

        # Save geometry to file if filename is provided
        if filename is not None:
            self.pickle(filename)
            print(f"Saved RayTraceLFM object from {filename}")

        return self

    def filter_rays_based_on_pixels(self, image):
        """
        Filters the rays based on the non-zero pixel values in the given image.
        Args:
            image (numpy.ndarray): The image used to filter the rays.
        Returns:
            self: The updated instance of the class with filtered ray attributes.
        """
        # Reshape self.ray_valid_indices to pair row and column indices
        reshaped_indices = self.ray_valid_indices.T

        # Create a boolean mask based on whether the image value at each index is not zero
        mask = np.array([image[index[0], index[1]] != 0 for index in reshaped_indices])

        # Apply the mask to reshaped_indices
        filtered_reshaped_indices = reshaped_indices[mask]

        # ray_valid_indices_by_ray_num is not adjusted
        #   because it is not used in the forward model
        colli_indices = self.ray_vol_colli_indices
        filtered_ray_vol_colli_indices = [
            idx for idx, mask_val in zip(colli_indices, mask) if mask_val
        ]
        filtered_ray_vol_colli_lengths = self.ray_vol_colli_lengths[mask]
        filtered_ray_valid_direction = self.ray_valid_direction[mask]

        # Re-assign instance attributes
        self.ray_vol_colli_indices = filtered_ray_vol_colli_indices
        self.ray_vol_colli_lengths = filtered_ray_vol_colli_lengths
        self.ray_valid_direction = filtered_ray_valid_direction

        # Transpose back to get the filtered self.ray_valid_indices
        self.ray_valid_indices = filtered_reshaped_indices.T

        return self

    def identify_rays_from_pixels_mla(self, mla_image, ray_valid_indices=None):
        """
        Args:
            mla_image (np.array): Light field image.
        Return:
            nonzero_pixels_dict: Mask for which rays lead to nonzero pixels.
                Keys are tuples of (lenslet_row, lenslet_col).
                Values are boolean arrays of shape (num_pixels, num_pixels).
        Class attributes used:
        - self.ray_valid_indices: 2D indices of the valid rays.
        """
        num_mla = self.optical_info["n_micro_lenses"]
        num_pixels = self.optical_info["pixels_per_ml"]
        mla_pixels = num_mla * num_pixels
        err_message = (
            "mla_image must be a square matrix of shape "
            + f"{mla_pixels, mla_pixels} instead of {mla_image.shape}."
        )
        assert mla_image.shape == (mla_pixels, mla_pixels), err_message

        nonzero_pixels_dict = {}

        if ray_valid_indices is None:
            self.compute_rays_geometry()
            # Reshape self.ray_valid_indices to pair row and column indices
            reshaped_indices = self.ray_valid_indices.T
        else:
            reshaped_indices = ray_valid_indices.T

        # Loop through sections of mla_image, and store a mask.
        for i in range(num_mla):
            for j in range(num_mla):
                lenslet_image = mla_image[
                    i * num_pixels : (i + 1) * num_pixels,
                    j * num_pixels : (j + 1) * num_pixels,
                ]
                # Initialize a mask of the same shape as lenslet_image
                mask = np.full(lenslet_image.shape, False, dtype=bool)
                # Set True in the mask only for indices in reshaped_indices
                # and where lenslet_image is not zero.
                for idx in reshaped_indices:
                    if lenslet_image[idx[0], idx[1]] > 3e-8:
                        mask[idx[0], idx[1]] = True
                nonzero_pixels_dict[(i, j)] = mask

        return nonzero_pixels_dict

    def _load_geometry_from_file(self, filename):
        """Loads ray tracer class from a file if it exists."""
        if filename and exists(filename):
            data = self.unpickle(filename)
            print(f"Loaded RayTraceLFM object from {filename}")
            return data
        return False

    def _initialize_ray_geometry(self):
        """
        Initializes and calculates the ray geometry for the RayTraceLFM
        class. Fetches necessary variables from `optical_info`, computes
        the restricted volume shape, and calculates the central voxel
        indices and their unit measurements. Determines ray entry, exit,
        and direction for the optical setup.

        Adjusts for numpy and torch rays, especially when rays extend
        outside the volume of interest. Stores ray geometry locally
        based on the backend (numpy or pytorch).

        Returns:
            tuple: A tuple with three elements:
                - ray_enter: Entry points of rays (np.array/torch.Tensor)
                - ray_exit: Exit points of rays (np.array/torch.Tensor)
                - ray_diff: Direction of rays (np.array/torch.Tensor)

        Note:
            Updates attributes `vol_shape_restricted`, `ray_entry`,
            `ray_exit`, and `ray_direction`. Converts ray geometry
            to pytorch tensors for pytorch backend, otherwise uses
            numpy arrays.
        """
        # We may need to treat differently numpy and torch rays, as some
        # rays go outside the volume of interest.

        # Fetch necessary variables from optical_info
        pixels_per_ml = self.optical_info["pixels_per_ml"]
        naObj = self.optical_info["na_obj"]
        nMedium = self.optical_info["n_medium"]
        valid_vol_shape = (
            self.optical_info["n_micro_lenses"] * self.optical_info["n_voxels_per_ml"]
        )
        self.vol_shape_restricted = [
            self.optical_info["volume_shape"][0],
        ] + 2 * [valid_vol_shape]
        # vox_ctr_idx is in index units
        vox_ctr_idx_restricted = np.array(
            [
                self.vol_shape_restricted[0] / 2,
                self.vol_shape_restricted[1] / 2,
                self.vol_shape_restricted[2] / 2,
            ]
        )
        voxel_size_um = self.optical_info["voxel_size_um"]
        # volume_ctr_um_restricted is in volume units (um)
        volume_ctr_um_restricted = vox_ctr_idx_restricted * voxel_size_um

        # Calculate the ray geometry
        ray_enter, ray_exit, ray_diff = RayTraceLFM.rays_through_vol(
            pixels_per_ml, naObj, nMedium, volume_ctr_um_restricted
        )

        # Store locally
        if self.backend == BackEnds.PYTORCH:
            self.ray_entry = torch.from_numpy(ray_enter).float()
            self.ray_exit = torch.from_numpy(ray_exit).float()
            self.ray_direction = torch.from_numpy(ray_diff).float()
        else:
            self.ray_entry = ray_enter
            self.ray_exit = ray_exit
            self.ray_direction = ray_diff

        return ray_enter, ray_exit, ray_diff

    @staticmethod
    def compute_lateral_ray_length_and_voxel_span(ray_diff, axial_volume_dim):
        """
        Computes the lateral length of the ray and the maximum voxel span.

        Args:
            ray_diff (np.array): The ray differential, an array containing
                                the direction of the rays through the volume.
        Returns:
            lateral_ray_length_from_center (float): The maximum lateral
                reach of a ray from the center voxel.
            voxel_span_per_ml (float): The maximum voxel span per microlens.
                This the the lateral_ray_length_from_center rounded up.
        """
        # Find the first valid ray from one of the borders
        half_ml_shape = ray_diff.shape[1] // 2
        valid_ray_coord = 0
        while np.isnan(ray_diff[0, valid_ray_coord, half_ml_shape]):
            valid_ray_coord += 1

        # Compute how long is the ray laterally
        lateral_ray_length = (
            axial_volume_dim
            * ray_diff[2, valid_ray_coord, half_ml_shape]
            / ray_diff[0, valid_ray_coord, half_ml_shape]
        )

        # Compensate for different voxel sizes axially vs laterally
        # Uncomment the next line if different voxel sizes need to be considered
        # lateral_ray_length *= (self.optical_info['voxel_size_um'][1] /
        #                        self.optical_info['voxel_size_um'][0])

        # Compute the maximum reach of a ray from the center voxel
        voxel_span_per_ml = np.ceil(lateral_ray_length / 2.0)

        lateral_ray_length_from_center = lateral_ray_length / 2.0

        return lateral_ray_length_from_center, voxel_span_per_ml

    def _filter_invalid_rays(
        self, max_num_collisions, collision_lengths, valid_direction
    ):
        """
        Filters out invalid rays and processes the data of valid rays.

        This method takes the lengths of ray-voxel collisions and the
        valid directions of rays, then filters out invalid rays based on
        the provided indices. It initializes and populates arrays for
        storing the processed ray-voxel collision lengths and valid ray
        directions.

        Args:
            max_num_collisions (int): The maximum number of voxel collisions for any ray.
            collision_lengths (list or np.array or torch.Tensor): A collection that
                contains the lengths of traversal of each ray through voxels.
            valid_direction (list or np.array or torch.Tensor): A collection that
                contains the directions of each valid ray.

        Returns:
            None: This method does not return any value but updates the instance variables
                `ray_vol_colli_lengths` and `ray_valid_direction` with the processed
                information for valid rays.

        Notes:
        - The method distinguishes between handling data in NumPy and PyTorch
            based on the backend specified in the class.
        - The instance variable `ray_valid_indices_by_ray_num` is expected
            to be populated beforehand to indicate valid rays.
        """
        n_valid_rays = len(self.ray_valid_indices_by_ray_num)

        # Initialize storage for ray-voxel collision lengths and ray valid directions
        if self.backend == BackEnds.NUMPY:
            self.ray_vol_colli_lengths = np.zeros([n_valid_rays, max_num_collisions])
            self.ray_valid_direction = np.zeros([n_valid_rays, 3])
        elif self.backend == BackEnds.PYTORCH:
            self.ray_vol_colli_lengths = torch.zeros(n_valid_rays, max_num_collisions)
            self.ray_vol_colli_lengths.requires_grad = False
            self.ray_valid_direction = torch.zeros(n_valid_rays, 3)
            self.ray_valid_direction.requires_grad = False

        # Process each valid ray
        for valid_ray in range(n_valid_rays):
            val_lengths = collision_lengths[valid_ray]
            self.ray_valid_direction[valid_ray, :] = valid_direction[valid_ray]
            if self.backend == BackEnds.NUMPY:
                self.ray_vol_colli_lengths[valid_ray, : len(val_lengths)] = val_lengths
            elif self.backend == BackEnds.PYTORCH:
                self.ray_vol_colli_lengths[valid_ray, : len(val_lengths)] = (
                    torch.tensor(val_lengths)
                )

    def compute_ray_collisions(self, ray_enter, ray_exit, voxel_size_um, vol_shape):
        """
        Computes parameters for collisions of rays with voxels.
        For each ray defined by start (ray_enter) and end (ray_exit) points,
        calculates the intersected voxels and lengths of intersections within
        a volume of given shape (vol_shape) and voxel size (voxel_size_um).

        Args:
        - ray_enter: Array of ray start points.
        - ray_exit: Array of ray end points.
        - voxel_size_um: Size of a single voxel in micrometers.
        - vol_shape: Shape of the volume as a list [x, y, z].

        Returns:
        - ray_valid_indices: List of valid ray indices.
        - ray_vol_colli_indices: List of voxel indices for each ray segment.
        - ray_vol_colli_lengths: List of intersection lengths for each voxel.
        - ray_valid_direction: List of directions for each valid ray.

        Class attributes accessed:
        - self.ray_direction: The direction of each valid ray.
        """
        ray_valid_indices = []
        ray_valid_direction = []
        ray_vol_colli_indices = []
        ray_vol_colli_lengths = []

        i_range, j_range = ray_enter.shape[1:]

        for ii in range(i_range):
            for jj in range(j_range):
                start = ray_enter[:, ii, jj]
                stop = ray_exit[:, ii, jj]

                if np.any(np.isnan(start)) or np.any(np.isnan(stop)):
                    if self.backend == BackEnds.PYTORCH:
                        continue
                    siddon_list = []
                    voxels_of_segs = []
                    seg_mids = []
                    voxel_intersection_lengths = []
                else:
                    siddon_list = siddon_params(start, stop, voxel_size_um, vol_shape)
                    seg_mids = siddon_midpoints(start, stop, siddon_list)
                    voxels_of_segs = vox_indices(seg_mids, voxel_size_um)
                    voxel_intersection_lengths = siddon_lengths(
                        start, stop, siddon_list
                    )

                ray_valid_indices.append((ii, jj))
                ray_vol_colli_indices.append(voxels_of_segs)
                ray_vol_colli_lengths.append(voxel_intersection_lengths)
                ray_valid_direction.append(self.ray_direction[:, ii, jj])

        return (
            ray_valid_indices,
            ray_vol_colli_indices,
            ray_vol_colli_lengths,
            ray_valid_direction,
        )

    def _update_volume_shape_info(self):
        """
        Updates volume shape information, including the 3D index and
        coordinates of the central voxel.
        """
        vol_shape = self.optical_info["volume_shape"]
        # Central voxel index (in index units)
        self.vox_ctr_idx = np.array(
            [vol_shape[0] // 2, vol_shape[1] // 2, vol_shape[2] // 2], dtype=int
        )
        # Central voxel coordinates (in volume units)
        self.volume_ctr_um = self.vox_ctr_idx * self.optical_info["voxel_size_um"]

    def _calculate_ray_directions(self):
        """
        Calculates the direction of each valid ray with two perpendicular
        directions. Updates ray direction basis.
        """
        if self.backend == BackEnds.NUMPY:
            self.ray_direction_basis = [
                RayTraceLFM.calc_ray_direction(ray) for ray in self.ray_valid_direction
            ]
        elif self.backend == BackEnds.PYTORCH:
            self.ray_direction_basis = RayTraceLFM.calc_ray_direction_torch(
                self.ray_valid_direction
            )

    # Helper functions to load/save the whole class to disk
    def pickle(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def unpickle(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)

    def plot_rays(self, colormap="inferno", use_matplotlib=False):
        """Multiple methods to plot the ray geometry"""
        ray_entry = self.ray_entry
        ray_exit = self.ray_exit
        i_shape, j_shape = ray_entry.shape[1:]

        # Grab all rays
        all_entry = np.reshape(ray_entry, [ray_entry.shape[0], i_shape * j_shape])
        all_exit = np.reshape(ray_exit, [ray_entry.shape[0], i_shape * j_shape])
        x_entry, y_entry, z_entry = all_entry[1, :], all_entry[2, :], all_entry[0, :]
        x_exit, y_exit, z_exit = all_exit[1, :], all_exit[2, :], all_exit[0, :]

        # grab the ray index to color them
        ray_index = list(range(len(x_exit)))
        if use_matplotlib:
            import matplotlib

            # And plot them
            plt.clf()
            ax = plt.subplot(1, 3, 1)
            plt.scatter(x_entry, y_entry, c=ray_index, cmap=colormap)
            ax.set_box_aspect(1)
            plt.title("entry rays coords")
            ax = plt.subplot(1, 3, 2)
            plt.scatter(x_exit, y_exit, c=ray_index, cmap=colormap)
            ax.set_box_aspect(1)
            plt.title("exit rays coords")
            ax = plt.subplot(1, 3, 3, projection="3d")
            for ray_ix in range(len(x_entry)):
                cmap = matplotlib.cm.get_cmap(colormap)
                rgba = cmap(ray_ix / len(x_entry))
                plt.plot(
                    [x_entry[ray_ix], x_exit[ray_ix]],
                    [y_entry[ray_ix], y_exit[ray_ix]],
                    [z_entry[ray_ix], z_exit[ray_ix]],
                    color=rgba,
                )

            # # Add area covered by MLAs
            # if optical_config is not None:
            #     mz = optical_config.volume_config.volume_size_um[0]/2
            #     m = optical_config.volume_config.volume_size_um[1]/2
            #     n_mlas = optical_config.mla_config.n_mlas//2
            #     mla_sample_pitch = optical_config.mla_config.pitch / optical_config.PSF_config.M
            #     x = [m-n_mlas*mla_sample_pitch,
            #          m+n_mlas*mla_sample_pitch,
            #          m+n_mlas*mla_sample_pitch,
            #          m-n_mlas*mla_sample_pitch]
            #     y = [m-n_mlas*mla_sample_pitch,
            #          m-n_mlas*mla_sample_pitch,
            #          m+n_mlas*mla_sample_pitch,
            #          m+n_mlas*mla_sample_pitch]
            #     z = [mz,mz,mz,mz]
            #     verts = [list(zip(x,y,z))]
            #     ax.add_collection3d(Poly3DCollection(verts,alpha=.20))

            # ax.set_box_aspect((1,1,5))
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()
        else:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            # method = "entrance and exit coords"
            method = "rays through volume"
            if method == "entrance and exit coords":
                # import plotly.express as px

                # Plot entry and exit?
                # if False:
                fig = make_subplots(
                    rows=1,
                    cols=3,
                    specs=[[{"is_3d": False}, {"is_3d": False}, {"is_3d": True}]],
                    subplot_titles=(
                        "Entry rays coords",
                        "Exit rays coords",
                        "Rays through volume",
                    ),
                    print_grid=True,
                )
                fig.update_layout(
                    autosize=True,
                    scene=dict(aspectratio=dict(x=1, y=1, z=1), aspectmode="manual"),
                )
                fig.append_trace(
                    go.Scatter(
                        x=x_entry,
                        y=y_entry,
                        mode="markers",
                        marker=dict(color=ray_index, colorscale=colormap),
                    ),
                    row=1,
                    col=1,
                )
                fig.append_trace(
                    go.Scatter(
                        x=x_exit,
                        y=y_exit,
                        mode="markers",
                        marker=dict(color=ray_index, colorscale=colormap),
                    ),
                    row=1,
                    col=2,
                )

                # Plot rays
                for ray_ix in range(len(x_entry)):
                    cmap = matplotlib.cm.get_cmap(colormap)
                    rgba = cmap(ray_ix / len(x_entry))
                    if not np.isnan(x_entry[ray_ix]) and not np.isnan(x_exit[ray_ix]):
                        fig.append_trace(
                            go.Scatter3d(
                                x=[x_entry[ray_ix], x_exit[ray_ix]],
                                y=[y_entry[ray_ix], y_exit[ray_ix]],
                                z=[z_entry[ray_ix], z_exit[ray_ix]],
                                marker=dict(color=rgba, size=4),
                                line=dict(color=rgba),
                            ),
                            row=1,
                            col=3,
                        )
            elif method == "rays through volume":
                fig = make_subplots(
                    rows=1,
                    cols=1,
                    specs=[[{"is_3d": True}]],
                    subplot_titles=("Rays through volume",),
                    print_grid=True,
                )
                # Gather all rays in single arrays, to plot them all at once,
                #   and placing NAN in between them.
                # Prepare colormap
                all_x = np.empty((3 * len(x_entry)))
                all_x[::3] = x_entry
                all_x[1::3] = x_exit
                all_x[2::3] = np.NaN

                all_y = np.empty((3 * len(y_entry)))
                all_y[::3] = y_entry
                all_y[1::3] = y_exit
                all_y[2::3] = np.NaN

                all_z = np.empty((3 * len(z_entry)))
                all_z[::3] = z_entry
                all_z[1::3] = z_exit
                all_z[2::3] = np.NaN

                # prepare colors for each line
                rgba = [ray_ix / len(all_x) for ray_ix in range(len(all_x))]
                # Draw the lines and markers
                fig.append_trace(
                    go.Scatter3d(
                        z=all_x,
                        y=all_y,
                        x=all_z,
                        marker=dict(color=rgba, colorscale=colormap, size=4),
                        line=dict(color=rgba, colorscale=colormap),
                        connectgaps=False,
                        mode="lines+markers",
                    ),
                    row=1,
                    col=1,
                )
                fig.update_layout(
                    scene=dict(
                        xaxis_title="Axial dimension",
                    ),
                    # width=700,
                    margin=dict(r=0, l=0, b=0, t=0),
                )

            fig.show()

    ######## Not implemented: These functions need an implementation in derived objects
    def ray_trace_through_volume(self, volume_in):
        """We have a separate function as we have some basic functionality that is shared"""
        raise NotImplementedError

    def init_volume(self, volume_in):
        """This function assigns a volume the correct internal structure for a given simul_type
        For example: a single value per voxel for fluorescence, or two values for birefringence
        """
        raise NotImplementedError
