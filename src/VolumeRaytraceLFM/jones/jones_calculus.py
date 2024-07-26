"""Jones Calculus Matrices and Vector Generators

Constructors for different types of elements.
These methods are constructors only. They don't support torch
optimization of internal variables.
"""

import numpy as np
import torch
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_base import BirefringentElement


class JonesMatrixGenerators(BirefringentElement):
    """2x2 Jones matrices representing various of polariztion elements"""

    def __init__(self, backend: BackEnds = BackEnds.NUMPY):
        super().__init__(backend=backend, torch_args={}, optical_info={})

    @staticmethod
    def rotator(angle, backend=BackEnds.NUMPY):
        """2D rotation matrix
        Args:
            angle: angle to rotate by counterclockwise [radians]
        Return: Jones matrix"""
        if backend == BackEnds.NUMPY:
            s = np.sin(angle)
            c = np.cos(angle)
            R = np.array([[c, -s], [s, c]])
        elif backend == BackEnds.PYTORCH:
            s = torch.sin(angle)
            c = torch.cos(angle)
            R = torch.tensor([[c, -s], [s, c]])
        return R

    @staticmethod
    def linear_retarder(ret, azim, backend=BackEnds.NUMPY):
        """Linear retarder
        Args:
            ret (float): retardance [radians]
            azim (float): azimuth angle of fast axis [radians]
        Return: Jones matrix
        """
        retarder_azim0 = JonesMatrixGenerators.linear_retarder_azim0(
            ret, backend=backend
        )
        R = JonesMatrixGenerators.rotator(azim, backend=backend)
        Rinv = JonesMatrixGenerators.rotator(-azim, backend=backend)
        if backend == BackEnds.PYTORCH:
            dtype = retarder_azim0.dtype
            R = R.to(dtype)
            Rinv = Rinv.to(dtype)
        return R @ retarder_azim0 @ Rinv

    @staticmethod
    def linear_retarder_azim0(ret, backend=BackEnds.NUMPY):
        """todo"""
        if backend == BackEnds.NUMPY:
            return np.array([[np.exp(1j * ret / 2), 0], [0, np.exp(-1j * ret / 2)]])
        else:
            # Handling both single and multiple retardance values
            exp_ret = torch.exp(1j * ret / 2)
            zero = torch.zeros_like(exp_ret)
            jones_matrix = torch.stack(
                [
                    torch.stack([exp_ret, zero], dim=-1),
                    torch.stack([zero, torch.conj(exp_ret)], dim=-1),
                ],
                dim=-2,
            )
            return jones_matrix

    @staticmethod
    def linear_retarter_azim90(ret, backend=BackEnds.NUMPY):
        """Linear retarder, convention not establisted yet"""
        # TODO: using same convention as linear_retarder_azim0
        if backend == BackEnds.NUMPY:
            return np.array([[np.exp(1j * ret / 2), 0], [0, np.exp(-1j * ret / 2)]])
        else:
            return torch.tensor(
                [[torch.exp(1j * ret / 2), 0], [0, torch.exp(-1j * ret / 2)]]
            )

    @staticmethod
    def quarter_waveplate(azim):
        """Quarter Waveplate
        Linear retarder with lambda/4 or equiv pi/2 radians
        Commonly used to convert linear polarized light to circularly polarized light"""
        ret = np.pi / 2
        return JonesMatrixGenerators.linear_retarder(ret, azim)

    @staticmethod
    def half_waveplate(azim):
        """Half Waveplate
        Linear retarder with lambda/2 or equiv pi radians
        Commonly used to rotate the plane of linear polarization"""
        # Faster method
        s = np.sin(2 * azim)
        c = np.cos(2 * azim)
        # # Alternative method
        # ret = np.pi
        # JM = self.LR(ret, azim)
        return np.array([[c, s], [s, -c]])

    @staticmethod
    def linear_polarizer(theta):
        """Linear Polarizer
        Args:
            theta: angle that light can pass through
        Returns: Jones matrix
        """
        c = np.cos(theta)
        s = np.sin(theta)
        J00 = c**2
        J11 = s**2
        J01 = s * c
        J10 = J01
        return np.array([[J00, J01], [J10, J11]])

    @staticmethod
    def right_circular_polarizer():
        """Right Circular Polarizer"""
        return 1 / 2 * np.array([[1, -1j], [1j, 1]])

    @staticmethod
    def left_circular_polarizer():
        """Left Circular Polarizer"""
        return 1 / 2 * np.array([[1, 1j], [-1j, 1]])

    @staticmethod
    def right_circular_retarder(ret):
        """Right Circular Retarder"""
        return JonesMatrixGenerators.rotator(-ret / 2)

    @staticmethod
    def left_circular_retarder(ret):
        """Left Circular Retarder"""
        return JonesMatrixGenerators.rotator(ret / 2)

    @staticmethod
    def polscope_analyzer():
        """Acts as a circular polarizer
        Inhomogeneous elements because eigenvectors are linear (-45 deg) and
        (right) circular polarization states
        Source: 2010 Polarized Light pg. 224"""
        return 1 / (2 * np.sqrt(2)) * np.array([[1 + 1j, 1 - 1j], [1 + 1j, 1 - 1j]])

    @staticmethod
    def universal_compensator(retA, retB):
        """Universal Polarizer
        Used as the polarizer for the LC-PolScope"""
        LP = JonesMatrixGenerators.linear_polarizer(0)
        LCA = JonesMatrixGenerators.linear_retarder(retA, -np.pi / 4)
        LCB = JonesMatrixGenerators.linear_retarder_azim0(retB)
        return LCB @ LCA @ LP

    @staticmethod
    def universal_compensator_modes(setting=0, swing=0):
        """Settings for the LC-PolScope polarizer
        Parameters:
            setting (int): LC-PolScope setting number between 0 and 4
            swing (float): proportion of wavelength, for ex 0.03
        Returns:
            Jones matrix"""
        swing_rad = swing * 2 * np.pi
        if setting == 0:
            retA = np.pi / 2
            retB = np.pi
        elif setting == 1:
            retA = np.pi / 2 + swing_rad
            retB = np.pi
        elif setting == 2:
            retA = np.pi / 2
            retB = np.pi + swing_rad
        elif setting == 3:
            retA = np.pi / 2
            retB = np.pi - swing_rad
        elif setting == 4:
            retA = np.pi / 2 - swing_rad
            retB = np.pi
        return JonesMatrixGenerators.universal_compensator(retA, retB)


class JonesVectorGenerators(BirefringentElement):
    """2x1 Jones vectors representing various states of polarized light"""

    def __init__(self, backend: BackEnds = BackEnds.NUMPY):
        super().__init__(backend=backend, torch_args={}, optical_info={})

    @staticmethod
    def right_circular():
        """Right circularly polarized light"""
        return np.array([1, -1j]) / np.sqrt(2)

    @staticmethod
    def left_circular():
        """Left circularly polarized light"""
        return np.array([1, 1j]) / np.sqrt(2)

    @staticmethod
    def linear(angle):
        """Linearlly polarized light at an angle in radians"""
        return JonesMatrixGenerators.rotator(angle) @ np.array([1, 0])

    @staticmethod
    def horizonal():
        """Horizontally polarized light"""
        return np.array([1, 0])

    @staticmethod
    def vertical():
        """Vertically polarized light"""
        return np.array([0, 1])
