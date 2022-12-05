
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------
# Authors:    Luis Miguel Sanchez Brea and Jesus del Hoyo
# Date:       2019/01/09 (version 1.0)
# License:    GPL
# -------------------------------------
"""
Jones_matrix objects describe optical polarization elements in the Jones formalism.

**Class fields:**
    * **M**: 2x2xN array containing all the Jones matrices.
    * **name**: Name of the object for print purposes.
    * **shape**: Shape desired for the outputs.
    * **size**: Number of stored Jones matrices.
    * **ndim**: Number of dimensions for representation purposes.
    * **no_rotation**: If True, rotation method do not act upon the object. Useful for objects that shouldn't be rotated as mirrors.
    * **type**: Type of the object ('Jones_matrix'). This is used for determining the object class as using isinstance may throw unexpected results in .ipynb files.
    * **parameters**: parameters of the Jones matrices.
    * **checks**: checks of the Jones matrices.
    * **analysis**: analysis of the Jones matrices.


**Parent methods**
    * **clear**:  Removes data and name form Jones vector.
    * **copy**:  Creates a copy of the Jones_vector object.
    * **stretch**:  Stretches a Jones vector of size 1.
    * **shape_like**:  Takes the shape of another object to use as its own.
    * **reshape**: Changes the shape of the object.
    * **flatten**:  Transforms N-D objects into 1-D objects (0-D if only 1 element).
    * **flip**: Flips the object along some dimensions.
    * **get_list**: Creates a list with single elements.
    * **from_list**: Creates the object from a list of single elements.
    * **concatenate**: Canocatenates several objects into a single one.
    * **draw**: Draws the components of the object.
    * **clear**: Clears the information of the object.


**Generation methods**
    * **from_components**: Creates a Jones_matrix object directly from the 4 elements (m00, m01, m10, m11).
    * **from_matrix**: Creates a Jones_matrix object directly from a 2x2xN numpy arrays.
    * **from_list**: Creates a Jones_matrix object directly from a list of 2x2 numpy arrays.
    * **from_Mueller**: Takes a non-depolarizing Mueller Matrix and converts into Jones matrix.
    * **vacuum**:  Creates the matrix for vacuum.
    * **mirror**: Creates the matrix for a mirror. NOTE: Don't rotate this matrix.
    * **filter_amplifier**: Creates the matrix for a neutral filter or amplifier element
    * **diattenuator_perfect**: Creates a perfect linear diattenuator.
    * **diattenuator_linear**: Creates a real polarizer with perpendicular axes.
    * **diattenuator_retarder_linear**: Creates a linear diattenuator retarder with the same axes for diattenuation and retardance.
    * **diattenuator_charac_angles**: Creates the most general homogeneous diattenuator with orthogonal eigenstates from the characteristic angles of the main eigenstate.
    * **diattenuator_azimuth_ellipticity**: Creates the general homogeneous diattenuator with orthogonal eigenstates from the characteristic angles of the main eigenstate.
    * **quarter_waveplate**: Creates a quarter-waveplate.
    * **half_waveplate**: Creates a half-waveplate.
    * **retarder_linear**: Creates a retarder using delay.
    * **retarder_material**: Creates a retarder using physical properties of a anisotropic material.
    * **retarder_charac_angles**: Creates the most general homogenous retarder with orthogonal eigenstates from the characteristic angles of the main eigenstate.
    * **retarder_azimuth_ellipticity**: Creates the general homogeneous retarder with orthogonal eigenstates from the characteristic angles of the main eigenstate.


**Manipulation methods**
    * **rotate**: Rotates the Jones matrix.
    * **sum**: Calculates the summatory of the Jones matrices in the object.
    * **prod**: Calculates the product of the Jones matrices in the object.
    * **remove_global_phase**: Removes the phase introduced by the optical element (respect to J00).
    * **add_global_phase**: Increases the phase introduced by the optical element.
    * **set_global_phase**: Sets the phase introduced by the optical element.
    * **reciprocal**: The optical element is fliped so the light transverses it in the opposite direction.
    * **transpose**: Transposes the Jones matrix of the element.
    * **hermitan**: Calculates the hermitan matrix of the Jones matrix.
    * **inverse**: Calculates the inverse matrix of the Jones matrix.


**Parameters subclass methods**
    * **matrix**:  Gets a numpy array with all the matrices.
    * **components**: Extracts the four components of the Jones matrix.
    * **inhomogeneity**: Calculates the inhomogeneity parameter.
    * **diattenuation / polarizance**:   Calculates the diattenuation of the matrix.
    * **retardance**: Calculates the retardance (or delay) introduced between the fast and slow axes.
    * **global_phase**: Calculates the phase introduced by the optical element (respect to J00).
    * **transmissions**: Calculates the maximum and minimum field and/or intensity transmissions.
    * **mean_transmission**: Calculates the mean intensity transmission.
    * **eig**: Calculates the eigenvalues and eigenstates (eigenvectors) of the Jones matrices.
    * **eigenvalues**: Calculates the eigenvalues and of the Jones matrices.
    * **eigenstates**: Calculates the eigenstates (eigenvectors) of the Jones matrices.
    * **det**: Calculates the determinant and of the Jones matrices.
    * **trace**: Calculates the trace of the Jones matrices.
    * **norm**: Calculates the norm of the Jones matrices.
    * **get_all**: Returns a dictionary with all the parameters of the object.


**Checks subclass methods**
    * **is_phisycall**: Check if the Jones matrices correspond to physically realizable optical elements.
    * **is_homogeneous**: Determines if the matrices correspond to homogeneous optical elements.
    * **is_retarder**: Checks if the Jones matrices correspond to homogeneous retarders.
    * **is_diattenuator / is_polarizer**: Checks if the Jones matrices correspond to homogeneous diattenuators.
    * **is_symmetric**: Checks if the Jones matrices are symmetric.
    * **is_conjugate_symmetric**: Checks if the Jones matrices are conjugate symmetric.
    * **is_eigenstate**: Checks if a given light state is an eigenstate of the objct.
    * **get_all**: Returns a dictionary with all the checks of the object.


**Analysis subclass methods**
    * **decompose_pure**: Decomposes the Jones matrices in two: an homogeneous retarder and diattenuator.
    * **diattenuator / polarizer**: Analyzes the Jones matrices as if they were diattenuators.
    * **retarder**: Analyzes the Jones matrices as if they were retarders.
"""

from .utils import *
from .py_pol import Py_pol
from .jones_vector import Jones_vector, create_Jones_vectors
from . import degrees, eps, np, num_decimals, um, number_types
from copy import deepcopy
from functools import wraps
from cmath import exp as cexp
import copy as copy
import cmath
import warnings

warnings.filterwarnings('ignore')

X_jones = np.diag([1, -1])
N_print_list = 5
print_list_spaces = 3
empty_matrix = np.array(np.zeros((2, 2, 1), dtype=float))
change_names = True
tol_default = eps

################################################################################
# Functions
################################################################################


def create_Jones_matrices(name='J', N=1, out_object=True):
    """Function that creates several Jones_matrix objects at the same time from a list of names or a number.

    Parameters:
        name (string or list): Name of the object for print purposes. Default: 'J'.
        N (int): Number of created elements. Default: 1.
        out_object (bool): If N=1 and out_object is True the output is a Jones_matrix instead of a list. Default: True.


    Attributes:
        self.parameters (class): Class containing the measurable parameters of the Jones matrices.
        self.checks (class): Class containing the methods that check something about the Jones matrices.

    Returns:
        J (list or Jones_matrix): Result.
    """
    J = []
    if isinstance(name, list) or isinstance(name, tuple):
        for n in name:
            J.append(Jones_matrix(n))
    else:
        for _ in range(N):
            J.append(Jones_matrix(name))
    if len(J) == 1 and out_object:
        J = J[0]
    return J


def set_printoptions(N_list=None, list_spaces=None):
    """Function that modifies the global print options parameters.

    Parameters:
        N_list (int): Number of matrices that will be printed as a list if the shape of the object is 1D. Default: None
        list_spaces (int): Number of spaces between matrices if they are printed as a list. Default: None
    """
    global N_print_list, print_list_spaces
    if list_spaces is not None:
        print_list_spaces = list_spaces
    if N_list is not None:
        N_print_list = N_list


################################################################################
# Main class
################################################################################


class Jones_matrix(Py_pol):
    """Class for Jones matrices.

    Parameters:
        M (np.ndarray): 2x2xN array containing all the Jones matrices.
        name (string): Name of the object for print purposes.
        shape (tuple or list): Shape desired for the outputs.
        size (int): Number of stored Jones matrices.
        ndim (int): Number of dimensions for representation purposes.
        no_rotation (bool): If True, rotation method do not act upon the object. Useful for objects that shouldn't be rotated as mirrors.
        type (string): Type of the object ('Jones_matrix'). This is used for determining the object class as using isinstance may throw unexpected results in .ipynb files.

    Attributes:
        self.parameters (class): parameters of the Jones matrices.
        self.checks (class): checks of the Jones matrices.
        self.analysis (class): analysis of the Jones matrices.
    """
    __array_priority__ = 20000

    ############################################################################
    # Operations
    ############################################################################

    def __init__(self, name='J'):
        super().__init__(name=name, _class="Jones_matrix")
        self.no_rotation = False
        self.parameters = Parameters_Jones_Matrix(self)
        self.checks = Checks_Jones_Matrix(self)
        self.analysis = Analysis_Jones_Matrix(self)

    def __add__(self, other):
        """Adds two Jones matrices.

        Parameters:
            other (Jones_matrix): 2nd Jones matrix to add.

        Returns:
            j3 (Jones_matrix): Result.
        """
        try:
            if other.type == 'Jones_matrix':
                j3 = Jones_matrix()
                j3.M = self.M + other.M
                j3.shape = take_shape((self, other))
                if change_names:
                    j3.name = self.name + " + " + other.name
                return j3
            else:
                raise ValueError('other is {} instead of Jones_matrix.'.format(
                    other.type))
        except:
            raise ValueError('other is not a py_pol object')

    def __sub__(self, other):
        """Substracts two Jones matrices.

        Parameters:
            other (Jones_matrix): 2nd Jones matrix to substract.

        Returns:
            j3 (Jones_matrix): Result.
        """
        try:
            if other.type == 'Jones_matrix':
                j3 = Jones_matrix()
                j3.M = self.M - other.M
                j3.shape = take_shape((self, other))
                if change_names:
                    j3.name = self.name + " - " + other.name
                return j3
            else:
                raise ValueError('other is {} instead of Jones_matrix.'.format(
                    other.type))
        except:
            raise ValueError('other is not a py_pol object')

    def __mul__(self, other):
        """
        Multiplies the Jones matrix by a number, an array of numbers, a Jones vector or another Jones matrix.

        Parameters:
            other (float, numpy.ndarray, Jones_vector or Jones_matrix): 2nd object to multiply.

        Returns:
            j3 (Jones_matrix): Result.
        """
        # Easy case, multiply by a number
        if isinstance(other, number_types):
            j3 = Jones_matrix()
            j3.M = self.M * other
            j3.shape = self.shape
            if change_names:
                j3.name = self.name + " * " + str(other)
        # Multiply by an array of numbers
        elif isinstance(other, np.ndarray):
            j3 = Jones_matrix()
            if other.size == self.size or self.size == 1:
                if self.size == 1:
                    j3.M = np.multiply.outer(self.get_list(), other.flatten())
                else:
                    j3.M = np.multiply.outer(np.ones(
                        (2, 2)), other.flatten()) * self.M
                j3.shape = take_shape((self, other))
            else:
                raise ValueError(
                    'The number of elements in other and {} is not the same'.
                    format(self.name))
        # Multiply by py_pol objects
        elif other.type in ('Jones_vector', 'Jones_matrix'):
            # Prepare variables
            new_self, new_other = expand_objects([self, other], copy=True)
            # print(new_self.M.shape, new_other.M.shape)
            if other.type is 'Jones_vector':
                j3 = Jones_vector()
            else:
                j3 = Jones_matrix()
            # Multiply
            Mf = matmul_pypol(new_self.M, new_other.M)
            # if new_self.size == 1:
            #     Mf = new_self.get_list() @ new_other.get_list()
            # else:
            #     # Move axes of the variables to allow multiplication
            #     M1 = np.moveaxis(new_self.M, 2, 0)
            #     if other.type is 'Jones_vector':
            #         M2 = np.moveaxis(new_other.M, 1, 0)
            #         M2 = np.expand_dims(M2, 2)
            #         Mf = M1 @ M2
            #         Mf = np.moveaxis(np.squeeze(Mf), 0, 1)
            #     else:
            #         M2 = np.moveaxis(new_other.M, 2, 0)
            #         Mf = M1 @ M2
            #         Mf = np.moveaxis(Mf, 0, 2)
            j3.from_matrix(Mf)
            j3.shape = take_shape((self, other))
            if change_names:
                j3.name = self.name + " * " + other.name
        else:
            raise ValueError('other thype ({}) is not correct'.format(
                type(other)))
        return j3

    def __rmul__(self, other):
        """Multiplies a Jones matrix object by a number or array.

        Parameters:
            other (int, float, numpy.ndarray): 2nd element to multiply.

        Returns:
            j3 (Jones_matrix): Result.
        """
        j3 = Jones_matrix()
        # Easy case, multiply by a number
        if isinstance(other, number_types):
            j3.M = self.M * other
            j3.shape = self.shape
            if change_names:
                j3.name = self.name + " * " + str(other)
        # Multiply by an array of numbers
        elif isinstance(other, np.ndarray):
            if other.size == self.size or self.size == 1:
                if self.size == 1:
                    j3.M = np.multiply.outer(self.get_list(), other.flatten())
                else:
                    j3.M = np.multiply.outer(np.ones(
                        (2, 2)), other.flatten()) * self.M
                j3.shape = take_shape((self, other.shape))
            else:
                raise ValueError(
                    'The number of elements in other and {} is not the same'.
                    format(self.name))
        else:
            raise ValueError('other thype ({}) is not correct'.format(
                type(other)))
        return j3

    def __truediv__(self, other):
        """Divides a Jones matrix by a number or array.

        Parameters:
            other (int, float, numpy.ndarray): 2nd element to divide.

        Returns:
            j3 (Jones_matrix): Result.
        """
        j3 = Jones_matrix()
        # Easy case, multiply by a number
        if isinstance(other, number_types):
            j3.M = self.M / other
            j3.shape = self.shape
            if change_names:
                j3.name = self.name + " / " + str(other)
        # Multiply by an array of numbers
        elif isinstance(other, np.ndarray):
            if other.size == self.size or self.size == 1:
                if self.size == 1:
                    j3.M = np.multiply.outer(self.get_list(),
                                             1 / other.flatten())
                else:
                    j3.M = np.multiply.outer(np.ones(
                        (2, 2)), 1 / other.flatten()) * self.M
                j3.shape = take_shape((self, other))
            else:
                raise ValueError(
                    'The number of elements in other and {} is not the same'.
                    format(self.name))
        else:
            raise ValueError('other thype ({}) is not correct'.format(
                type(other)))
        return j3

    def __repr__(self):
        """
        Represents the Jones matrix with print().
        """
        # Extract the components
        J00, J01, J10, J11 = self.parameters.components()
        if np.sum(self.M.imag**2)/self.size < tol_default:
            J00, J01, J10, J11 = J00.real, J01.real, J10.real, J11.real
        # If the object is empty, say it
        if self.size == 0 or self.shape is None or np.all(self.M == 0):
            return '{} is empty\n'.format(self.name)
        # If the object is 0D or 1D, print it like a list or inline
        # elif self.size == 1 or self.shape is None or len(self.shape) < 2:
        elif self.ndim <= 1:
            if self.size <= N_print_list:
                list = self.get_list(out_number=False)
                l0_name = "{} = \n".format(self.name)
                l1_name = PrintMatrices(list, print_list_spaces)
                return l0_name + l1_name
            else:
                l0_name = "{} J00 = {}".format(self.name, J00)
                l1_name = " " * len(self.name) + " J01 = {}".format(J01)
                l2_name = " " * len(self.name) + " J10 = {}".format(J10)
                l3_name = " " * len(self.name) + " J11 = {}".format(J11)
        # Print higher dimensionality as pure arrays
        else:
            l0_name = "{} J00 = \n{}".format(self.name, J00)
            l1_name = "{} J01 = \n{}".format(self.name, J01)
            l2_name = "{} J10 = \n{}".format(self.name, J10)
            l3_name = "{} J11 = \n{}".format(self.name, J11)
        return l0_name + '\n' + l1_name + '\n' + l2_name + '\n' + l3_name + '\n'

    def __getitem__(self, index):
        """
        Implements object extraction from indices.
        """
        if change_names:
            E = Jones_matrix(self.name + '_picked')
        else:
            E = Jones_matrix(self.name)
        # If the indices are 1D, act upon the matrix directly
        if isinstance(index, (int, slice)) and self.ndim > 1:
            E.from_matrix(self.M[:, :, index])
        elif isinstance(index,
                        np.ndarray) and index.ndim == 1 and self.ndim > 1:
            E.from_matrix(self.M[:, :, index])
        # If not, act upon the components
        else:
            J00, J01, J10, J11 = self.parameters.components(out_number=False)
            M = np.array([[J00[index], J01[index]], [J10[index], J11[index]]])
            E.from_matrix(M)

        return E

    def __setitem__(self, index, data):
        """
        Implements object inclusion from indices.
        """
        # Check that data is a correct pypol object
        if self.type != data.type:
            raise ValueError('data is type {} instead of {}.'.format(
                data.type, self.type))
        # Change to complex if necessary
        if np.iscomplexobj(data.M):
            self.M = np.array(self.M, dtype=complex)
        # If the indices are 1D, act upon the matrix directly
        if isinstance(index, int) and self.ndim > 1:
            self.M[:, :, index] = np.squeeze(data.M)
        elif isinstance(index, slice) and self.ndim > 1:
            if data.size == 1:
                if index.step is None:
                    step = 1
                else:
                    step = index.step
                N = int((index.stop - index.start) / step)
                data2 = data.stretch(length=N, keep=True)
            else:
                data2 = data
            self.M[:, :, index] = np.squeeze(data2.M)
        elif isinstance(index,
                        np.ndarray) and index.ndim == 1 and self.ndim > 1:
            self.M[:, :, index] = data.M
        # If not, act upon the components
        else:
            J00, J01, J10, J11 = self.parameters.components(out_number=False)
            J00_new, J01_new, J10_new, J11_new = data.parameters.components(
                out_number=False)
            J00[index] = np.squeeze(J00_new)
            J01[index] = np.squeeze(J01_new)
            J10[index] = np.squeeze(J10_new)
            J11[index] = np.squeeze(J11_new)
            self.from_components((J00, J01, J10, J11))

    def __eq__(self, other):
        """
        Implements equality operation.
        """
        try:
            if other.type == 'Jones_matrix':
                j3 = self - other
                norm = j3.parameters.norm()
                cond = norm < tol_default
                return cond
            else:
                raise ValueError('other is {} instead of Jones_matrix.'.format(
                    other.type))
        except:
            raise ValueError('other is not a py_pol object')

    ##################################################################
    # Manipulation
    ##################################################################

    # @_actualize_
    def rotate(self, angle=0, keep=False, change_name=change_names):
        """Rotates a jones_matrix a certain angle:

        M_rotated = R(-angle) * self.M * R(angle)

        Parameters:
            angle (float): Rotation angle in radians. Default: 0
            keep (bool): If True, the original element is not updated. Default: False.
            change_name (bool): If True and angle is of size 1, changes the object name adding @ XX deg, being XX the total rotation angle. Default: True.

        Returns:
            (Jones_matrix): Rotated Jones matrix.
        """
        # Don't rotate objects that shouldn't
        if self.no_rotation:
            print('Warning: Tried to rotate {}, which must not be rotated.'.
                  format(self.name))
            return self
        else:
            # Act differently if we want to keep self intact
            if keep:
                new_obj = self.copy()
            else:
                new_obj = self
            # Prepare variables
            angle, new_obj, new_shape = prepare_variables([angle],
                                                          expand=[True],
                                                          obj=new_obj,
                                                          give_shape=True)
            # Calculate the rotation objects
            Jneg, Jpos = create_Jones_matrices(('', ''))
            Jneg.from_matrix(rotation_matrix_Jones(-angle))
            Jpos.from_matrix(rotation_matrix_Jones(angle))
            # Rotate
            other = Jneg * (new_obj * Jpos)
            new_obj.from_matrix(other.M)
            # Update name
            if change_name and angle.size == 1:
                if np.abs(angle) > tol_default:
                    new_obj.name = new_obj.name + \
                        " @ {:1.2f} deg".format(angle[0] / degrees)
            # Return
            new_obj.shape, _ = select_shape(new_obj, new_shape)
            return new_obj

    # @_actualize_
    def reciprocal(self,
                   keep=False,
                   global_phase=0,
                   length=1,
                   shape_like=None,
                   shape=None,
                   change_name=change_names):
        """Calculates the reciprocal of the optical element, so the light tranverses it in the opposite direction. It is calculated as:

        .. math:: J^{r}=\left[\begin{array}{cc}
                    1 & 0\\
                    0 & -1
                    \end{array}\right]J^{T}\left[\begin{array}{cc}
                    1 & 0\\
                    0 & -1
                    \end{array}\right]

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), pp 106.

        Parameters:
            keep (bool): If True, the original element is not updated. Default: False.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            change_name (bool): If True, changes the object name adding Recip. of at the beggining of the name. Default: True.

        Returns:
            (Jones_matrix): Result.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Extract the components
        J00, J01, J10, J11 = new_obj.transpose(
            keep=True).parameters.components(shape=False)
        new_obj.from_components((J00, -J01, -J10, J11))
        # Add the global phase
        new_obj = new_obj.add_global_phase(global_phase,
                                           length=length,
                                           shape_like=shape_like,
                                           shape=shape)
        # Fix the name if required
        if change_name:
            new_obj.name = 'Reciprocal of ' + new_obj.name
        return new_obj

    def transpose(self,
                  keep=False,
                  global_phase=0,
                  length=1,
                  shape_like=None,
                  shape=None,
                  change_name=change_names):
        """Calculates the transposed matrices of the Jones matrices.

        Parameters:
            keep (bool): if True, the original element is not updated. Default: False.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            change_name (bool): If True, changes the object name adding Recip. of at the beggining of the name. Default: True.

        Returns:
            (Jones_matrix): Modified object.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Caluclate inverse
        new_obj.from_matrix(np.transpose(new_obj.M, axes=(1, 0, 2)))
        # Add the global phase
        new_obj = new_obj.add_global_phase(global_phase,
                                           length=length,
                                           shape_like=shape_like,
                                           shape=shape)
        # Fix the name if required
        if change_name:
            new_obj.name = 'Transpose of ' + new_obj.name
        return new_obj

    def hermitian(self,
                  keep=False,
                  global_phase=0,
                  length=1,
                  shape_like=None,
                  shape=None,
                  change_name=change_names):
        """Calculates the hermitian conjugate matrix of the Mueller matrix.

        Parameters:
            keep (bool): if True, the original element is not updated. Default: False.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            change_name (bool): If True, changes the object name adding Recip. of at the beggining of the name. Default: True.

        Returns:
            (Jones_matrix): Modified object.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Caluclate inverse
        new_obj.from_matrix(np.conj(np.transpose(new_obj.M, axes=(1, 0, 2))))
        # Add the global phase
        new_obj = new_obj.add_global_phase(global_phase,
                                           length=length,
                                           shape_like=shape_like,
                                           shape=shape)
        # Fix the name if required
        if change_name:
            new_obj.name = 'Hermitian of ' + new_obj.name
        return new_obj

    def inverse(self,
                keep=False,
                global_phase=0,
                length=1,
                shape_like=None,
                shape=None,
                change_name=change_names):
        """Calculates the inverse matrix of the Jones matrix.

        Parameters:
            keep (bool): if True, the original element is not updated. Default: False.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            change_name (bool): If True, changes the object name adding Recip. of at the beggining of the name. Default: True.

        Returns:
            (Jones_matrix): Modified object.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Caluclate inverse
        new_obj.from_matrix(inv_pypol(self.M))
        # Add the global phase
        new_obj = new_obj.add_global_phase(global_phase,
                                           length=length,
                                           shape_like=shape_like,
                                           shape=shape)
        new_obj.shape, _ = select_shape(shape_var=self.shape,
                                                   shape_like=shape_like,
                                                   shape_fun=shape)
        # Fix the name if required
        if change_name:
            new_obj.name = 'Inverse of ' + new_obj.name
        return new_obj

    def sum(self, axis=None, keep=False, change_name=change_names):
        """Calculates the sum of Jones matrices stored in the object.

        Parameters:
            axis (int, list or tuple): Axes along which the summatory is performed. If None, all matrices are summed.
            keep (bool): If True, the original element is not updated. Default: False.
            change_name (bool): If True, changes the object name adding Recip. of at the beggining of the name. Default: True.

        Returns:
            (Jones_matrix): Modified object.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Simple case
        if axis is None or new_obj.ndim <= 1:
            M = np.sum(new_obj.M, axis=2)
        # Complicated case
        else:
            # Calculate maximum axis
            if isinstance(axis, int):
                axis = axis + 2
                m = axis
            else:
                axis = np.array(axis) + 2
                m = np.max(axis)
            # Check that the axes are correct
            if m >= new_obj.ndim + 2:
                raise ValueError(
                    'Axis {} greater than the number of dimensions of {}, which is {}'
                    .format(m, new_obj.name, new_obj.ndim))
            # Reshape M to fit the current shape
            shape = [2, 2] + new_obj.shape
            M = np.reshape(new_obj.M, shape)
            # check if the axis is int or not
            if isinstance(axis, int):
                M = np.sum(M, axis=axis)
            else:
                M = np.sum(M, axis=tuple(axis))
        # Create the object and return it
        new_obj.from_matrix(M)
        if change_names:
            new_obj.name = 'Sum of ' + new_obj.name
        return new_obj

    def prod(self, axis=None, keep=False, change_name=change_names):
        """Calculates the product of Jones matrices stored in the object.

        Parameters:
            axis (int, list or tuple): Axes along which the product is performed. If None, all matrices are multiplied.
            keep (bool): if True, the original element is not updated. Default: False.
            change_name (bool): If True, changes the object name adding Recip. of at the beggining of the name. Default: True.

        Returns:
            (Jones_matrix): Modified object.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Simple case
        if axis is not None:
            N_axis = np.array(axis).size
        if axis is None or new_obj.ndim <= 1 or new_obj.ndim == N_axis:
            M = new_obj.M[:, :, 0]
            for ind in range(1, new_obj.size):
                M = M @ new_obj.M[:, :, ind]
        # Complicated case
        else:
            # Calculate maximum axis
            if isinstance(axis, int):
                m = axis + 2
            else:
                axis = np.array(axis)
                m = np.max(axis) + 2
            # Check that the axes are correct
            if m >= new_obj.ndim + 2:
                raise ValueError(
                    'Axis {} greater than the number of dimensions of {}, which is {}'
                    .format(m, new_obj.name, new_obj.ndim))
            # Calculate shapes, sizes and indices
            if isinstance(axis, int):
                shape_removed = new_obj.shape[axis]
            else:
                shape_removed = np.array(new_obj.shape)[axis]
            N_removed = np.prod(shape_removed)
            ind_removed = combine_indices(
                np.unravel_index(np.array(range(N_removed)), shape_removed))
            shape_matrix = np.delete(new_obj.shape, axis)
            N_matrix = np.prod(shape_matrix)
            ind_matrix = combine_indices(
                np.unravel_index(np.array(range(N_matrix)), shape_matrix))
            shape_final = [2, 2] + list(shape_matrix)
            axes_aux = np.array(range(2, new_obj.ndim + 2))
            shape_orig = [2, 2] + list(new_obj.shape)
            # Make the for loop of the matrix to be calculated
            M_orig = np.reshape(new_obj.M, shape_orig)
            M = np.zeros(shape_final)
            for indM in range(N_matrix):
                # Make the multiplication loop
                indices = merge_indices(ind_matrix[indM], ind_removed[0], axis)
                aux = multitake(M_orig, indices, axes_aux)
                for indR in range(1, N_removed):
                    indices = merge_indices(ind_matrix[indM],
                                            ind_removed[indR], axis)
                    aux = aux @ multitake(M_orig, indices, axes_aux)
                # Store the result
                ind_aux = tuple([0, 0] + list(ind_matrix[indM]))
                M[ind_aux] = aux[0, 0]
                ind_aux = tuple([0, 1] + list(ind_matrix[indM]))
                M[ind_aux] = aux[0, 1]
                ind_aux = tuple([1, 0] + list(ind_matrix[indM]))
                M[ind_aux] = aux[1, 0]
                ind_aux = tuple([1, 1] + list(ind_matrix[indM]))
                M[ind_aux] = aux[1, 1]
        # Create the object and return it
        new_obj.from_matrix(M)
        if change_names:
            new_obj.name = 'Prod of ' + new_obj.name
        return new_obj


    # @_actualize_
    def remove_global_phase(self,
                            keep=False,
                            length=1,
                            shape_like=None,
                            shape=None):
        """Function that transforms the Jones vector removing the global phase, so J00 is real and positive.

        Parameters:
            keep (bool): If True, self is not updated. Default: False.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_matrix): Simplified Jones matrix.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Calculate the phase and the components
        old_shape = new_obj.shape
        J00, J01, J10, J11 = self.parameters.components(shape=False)
        phase = self.parameters.global_phase(shape=False)
        # Remove the phase
        phase = np.exp(1j * phase)
        new_obj.from_components(
            (J00 / phase, J01 / phase, J10 / phase, J11 / phase))
        new_obj.shape = old_shape
        # Return
        return new_obj

    def add_global_phase(self,
                         global_phase=0,
                         keep=False,
                         length=1,
                         shape_like=None,
                         shape=None):
        """Function that adds a phase to the Jones matrix.

        Parameters:
            global_phase (float or np.ndarray): Phase to be added to the Jones matrix. Default: 0.
            keep (bool): If True, self is not updated. Default: False.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_matrix): Recalculated Jones matrix.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Prepare variables
        global_phase, new_obj, new_shape = prepare_variables([global_phase],
                                                             expand=[True],
                                                             obj=new_obj,
                                                             give_shape=True)
        # Add the phase
        J00, J01, J10, J11 = new_obj.parameters.components(shape=False)
        global_phase = np.exp(1j * global_phase)
        new_obj.from_components((J00 * global_phase, J01 * global_phase,
                                 J10 * global_phase, J11 * global_phase))
        new_obj.shape, _ = select_shape(new_obj, new_shape)
        # Return
        return new_obj

    def set_global_phase(self,
                         global_phase=0,
                         keep=False,
                         length=1,
                         shape_like=None,
                         shape=None):
        """Function that sets the phase of the Jones matrix.

        Parameters:
            global_phase (float or np.ndarray): Phase to be added to the Jones matrix. Default: 0.
            keep (bool): If True, self is not updated. Default: False.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_matrix): Recalculated Jones matrix.
        """
        # Act differently if we want to keep self intact
        if keep:
            new_obj = self.copy()
        else:
            new_obj = self
        # Prepare variables
        global_phase, new_obj, new_shape = prepare_variables([global_phase],
                                                             expand=[True],
                                                             obj=new_obj,
                                                             give_shape=True)
        # Remove the current phase
        new_obj.remove_global_phase(shape_like=shape_like, shape=shape)
        # Add the phase
        new_obj.add_global_phase(global_phase,
                                 length=length,
                                 shape_like=shape_like,
                                 shape=shape)
        new_obj.shape, _ = select_shape(new_obj, new_shape)
        # Return
        return new_obj



    ####################################################################
    # Creation
    ####################################################################

    def from_components(self,
                        components,
                        global_phase=0,
                        length=1,
                        shape_like=None,
                        shape=None):
        """Creates the Jones matrix object form the arrays of electric field components.

        Parameters:
            components (tuple or list): A 4 element tuple containing the 4 components of the Jones matrices (J00, J01, J10, J11).
            global_phase (float or numpy.ndarray): Adds a global phase to the Jones matrix. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_matrix): Created object.
        """
        # Prepare variables
        J00, J01, J10, J11 = components
        (J00, J01, J10, J11, global_phase), new_shape = prepare_variables(
            vars=[J00, J01, J10, J11, global_phase],
            expand=[True, True, True, True, False],
            length=length,
            give_shape=True)
        # Add global Phase
        if global_phase is not 0:
            J00 = J00 * np.exp(1j * global_phase)
            J01 = J01 * np.exp(1j * global_phase)
            J10 = J10 * np.exp(1j * global_phase)
            J11 = J11 * np.exp(1j * global_phase)
        # Store
        self.M = np.array([[J00, J01], [J10, J11]])
        self.no_rotation = False
        self.shape, _ = select_shape(self,
                                             shape_var=new_shape,
                                             shape_fun=shape,
                                             shape_like=shape_like)
        return self

    def from_matrix(self,
                    M,
                    global_phase=0,
                    length=1,
                    shape_like=None,
                    shape=None):
        """Create a Jones_matrix object from an external array.

        Parameters:
            M (numpy.ndarray): New matrix. At least two dimensions must be of size 2.
            global_phase (float or numpy.ndarray): Adds a global phase to the Jones matrix. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_matrix): Created object.
        """
        # Check if the matrix is of the correct Size
        M = np.array(M)
        s = M.size
        # 1D and 2D
        if M.ndim == 1 or M.ndim == 2:
            if M.size % 4 == 0:
                M = np.reshape(M, (2, 2, int(M.size / 4)))
            else:
                raise ValueError(
                    'M must have a number of elements multiple of 4.')
            if M.size == 4:
                sh = None
            else:
                sh = [int(M.size / 4)]
        # 3D or more
        elif M.ndim > 2:
            sh = np.array(M.shape)
            N = np.sum(sh == 2)
            if N > 1:
                # Find the matrix indices and the final shape
                ind1 = np.argmin(~(sh == 2))
                sh = np.delete(sh, ind1)
                ind2 = np.argmin(~(sh == 2))
                sh = np.delete(sh, ind2)
                ind2 = ind2 + 1
                # Calculate the components and construct the matrix from them
                M = np.array([[
                    multitake(M, [0, 0], [ind1, ind2]).flatten(),
                    multitake(M, [0, 1], [ind1, ind2]).flatten()
                ],
                    [
                    multitake(M, [1, 0], [ind1, ind2]).flatten(),
                    multitake(M, [1, 1], [ind1, ind2]).flatten()
                ]])

            else:
                raise ValueError(
                    'M must have two elements in at least two dimensions.')
        else:
            raise ValueError('M can not be empty')

        # Increase length if required
        if M.size == 4 and length > 1:
            M = np.multiply.outer(np.squeeze(M), np.ones(length))
        # End operations
        self.M = M
        # self.size = M.size / 4
        self.no_rotation = False
        self.add_global_phase(global_phase)
        self.shape, _ = select_shape(self,
                                             shape_var=sh,
                                             shape_fun=shape,
                                             shape_like=shape_like)
        return self


    # @_actualize_
    def from_Mueller(self, M, length=1, shape_like=None, shape=None):
        """Converts a pure Mueller matrix into Jones matrix object. Elements of Mueller object which are not pure are converted into NaN values. The values are found inverting the equation:

        .. math:: M(J)=\left[\begin{array}{cccc}
                    1 & 0 & 0 & 1\\
                    1 & 0 & 0 & -1\\
                    0 & 1 & 1 & 0\\
                    0 & i & -i & 0
                    \end{array}\right]\left(J\otimes J^{*}\right)\left[\begin{array}{cccc}
                    1 & 0 & 0 & 1\\
                    1 & 0 & 0 & -1\\
                    0 & 1 & 1 & 0\\
                    0 & i & -i & 0
                    \end{array}\right]^{-1}

        References:
            Handbook of Optics vol 2. 22.36 (52-54)

        Parameters:
            M (Mueller): Mueller object.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (float or numpy.ndarray): Use the shape of this array. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_matrix): Created object.
        """
        # Extract components from the Jones object and derivatives
        M00, M01, M02, M03, M10, M11, M12, M13, M20, M21, M22, M23, M30, M31, M32, M33 = M.parameters.components(
            shape=False)
        # Calculate absolute value of the Jones components. Use abs to avoid -0
        J00 = np.sqrt(np.abs(M00 + M01 + M10 + M11) / 2)
        J01 = np.sqrt(np.abs(M00 - M01 + M10 - M11) / 2)
        J10 = np.sqrt(np.abs(M00 + M01 - M10 - M11) / 2)
        J11 = np.sqrt(np.abs(M00 - M01 - M10 + M11) / 2)
        # Calculate the complex phases
        phase_00 = M.parameters.global_phase(shape=False, give_nan=False)
        phase_01 = np.arctan2(-M03 - M13, M02 + M12) + phase_00
        phase_10 = np.arctan2(M30 + M31, M20 + M21) + phase_00
        phase_11 = np.arctan2(M32 + M23, M22 + M33) + phase_00
        self.from_components(
            (J00 * np.exp(1j * phase_00), J01 * np.exp(1j * phase_01),
             J10 * np.exp(1j * phase_10), J11 * np.exp(1j * phase_11)),
            length=length,
            shape=shape,
            shape_like=shape_like)

        return self

    # @_actualize_
    def vacuum(self, global_phase=0, length=1, shape_like=None, shape=None):
        """Creates the matrix for vacuum i.e., an optically neutral element.

        Parameters:
            global_phase (float or numpy.ndarray): Adds a global phase to the Jones matrix. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_matrix): Created object.
        """
        # Prepare variables
        global_phase, new_shape = prepare_variables(vars=[global_phase],
                                                    expand=[True],
                                                    length=length,
                                                    give_shape=True)
        # Calculate
        global_phase = np.exp(1j * global_phase)
        z = np.zeros_like(global_phase)
        self.from_components((global_phase, z, z, global_phase))
        self.no_rotation = False
        self.shape, _ = select_shape(self,
                                             shape_var=new_shape,
                                             shape_fun=shape,
                                             shape_like=shape_like)
        return self

    # @_actualize_
    def filter_amplifier(self,
                         D=1,
                         global_phase=0,
                         length=1,
                         shape_like=None,
                         shape=None):
        """Creates the Jones matrix of neutral filters or amplifiers.

        Parameters:
            D (float or numpy.ndarray): Attenuation (gain if > 1). Default: 1.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_matrix): Created object.
        """
        # Prepare variables
        (D,
         global_phase), new_shape = prepare_variables(vars=[D, global_phase],
                                                      expand=[True, False],
                                                      length=length,
                                                      give_shape=True)
        # Add global Phase
        if global_phase is not 0:
            D = D * np.exp(1j * global_phase)
        # Calculate
        z = np.zeros_like(D)
        self.from_components((D, z, z, D))
        self.no_rotation = False
        self.shape, _ = select_shape(self,
                                             shape_var=new_shape,
                                             shape_fun=shape,
                                             shape_like=shape_like)
        return self

    # @_actualize_
    def mirror(self,
               ref=1,
               ref_field=None,
               global_phase=0,
               length=1,
               shape_like=None,
               shape=None):
        """Jones matrix of a mirror.

        Parameters:
            ref (float or numpy.ndarray): Intensity reflectivity of the mirror. Default: 1.
            ref_field (float or numpy.ndarray): Electric field reflectivity coefficient. If not None, it overrides REF. Default: None.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_matrix): Created object.
        """
        # Use the field reflectivity
        if ref_field is None:
            ref_field = np.sqrt(ref)
        # Prepare variables
        (ref_field, global_phase), new_shape = prepare_variables(
            vars=[ref_field, global_phase],
            expand=[True, False],
            length=length,
            give_shape=True)
        # Calculate
        global_phase = np.exp(1j * global_phase)
        z = np.zeros_like(global_phase)
        self.from_components(
            (ref_field * global_phase, z, z, -ref_field * global_phase))
        self.no_rotation = True
        self.shape, _ = select_shape(self,
                                             shape_var=new_shape,
                                             shape_fun=shape,
                                             shape_like=shape_like)
        return self

    # @_actualize_
    def diattenuator_perfect(self,
                             azimuth=0,
                             global_phase=0,
                             length=1,
                             shape_like=None,
                             shape=None):
        """Creates a perfect diattenuator (polarizer).

        Parameters:
            azimuth (float or numpy.ndarray): rotation angle of the high transmission polarizer axis. Default: 0.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_matrix): Created object.
        """
        # Prepare variables
        (azimuth, global_phase), new_shape = prepare_variables(
            vars=[azimuth, global_phase],
            expand=[False, True],
            length=length,
            give_shape=True)
        z = np.zeros_like(global_phase)
        # Calculate
        global_phase = np.exp(1j * global_phase)
        self.M = np.array([[global_phase, z], [z, z]])
        self.no_rotation = False
        self.rotate(azimuth)
        self.shape, _ = select_shape(self,
                                             shape_var=new_shape,
                                             shape_fun=shape,
                                             shape_like=shape_like)
        return self

    # @_actualize_
    def diattenuator_linear(self,
                            p1=1,
                            p2=0,
                            Tmax=None,
                            Tmin=None,
                            azimuth=0,
                            global_phase=0,
                            length=1,
                            shape_like=None,
                            shape=None):
        """Creates a real polarizer with perpendicular axes:

        .. math:: J\left(\theta=0\right)=\left[\begin{array}{cc}
                    p_{1} & 0\\
                    0 & p_{2}
                    \end{array}\right]'.

        Parameters:
            p1 (float or numpy.ndarray): Electric field transmission coefficient of the transmission eigenstate. Default: 1.
            p2 (float or numpy.ndarray): Electric field transmission coefficient of the extinction eigenstate. Default: 0.
            Tmax (float or numpy.ndarray): Maximum transmission. If not None, overrides p1. Default: None.
            Tmin (float or numpy.ndarray): Minimum transmission. If not None, overrides p2. Default: None.
            azimuth (float or numpy.ndarray): rotation angle of the high transmission polarizer axis. Default: 0.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_matrix): Created object.
        """
        # Use field transmission coefficients
        if Tmax is not None:
            p1 = np.sqrt(Tmax)
        if Tmin is not None:
            p2 = np.sqrt(Tmin)
        # Prepare variables
        (p1, p2, azimuth, global_phase), new_shape = prepare_variables(
            vars=[p1, p2, azimuth, global_phase],
            expand=[True, True, False, False],
            length=length,
            give_shape=True)
        z = np.zeros_like(p1)
        # Add global Phase
        if global_phase is not 0:
            p1 = p1 * np.exp(1j * global_phase)
            p2 = p2 * np.exp(1j * global_phase)
        # Create the object
        self.M = np.array([[p1, z], [z, p2]])
        self.no_rotation = False
        self.rotate(azimuth)
        self.shape, _ = select_shape(self,
                                             shape_var=new_shape,
                                             shape_fun=shape,
                                             shape_like=shape_like)
        return self

    # @_actualize_
    def diattenuator_charac_angles(self,
                                   p1=1,
                                   p2=0,
                                   Tmax=None,
                                   Tmin=None,
                                   alpha=0,
                                   delay=0,
                                   global_phase=0,
                                   length=1,
                                   shape_like=None,
                                   shape=None):
        """Creates the most general homogenous diattenuator with orthogonal
        eigenstates from the characteristic angles of the main eigenstate.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016) pp 137.

        Parameters:
            p1 (float or numpy.ndarray): Electric field transmission coefficient of the transmission eigenstate. Default: 1.
            p2 (float or numpy.ndarray): [0, 1] Square root of the lower transmission for the other eigenstate. Default: 0.
            Tmax (float or numpy.ndarray): Maximum transmission. If not None, overrides p1. Default: None.
            Tmin (float or numpy.ndarray): Minimum transmission. If not None, overrides p2. Default: None.
            alpha (float or numpy.ndarray): [0, pi/2]: tan(alpha) is the ratio between field amplitudes of X and Y components. Default: 0.
            delay (float or numpy.ndarray): [0, 2*pi]: phase difference between X and Y field components. Default: 0.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_matrix): Created object.
        """
        # Use field transmission coefficients
        if Tmax is not None:
            p1 = np.sqrt(Tmax)
        if Tmin is not None:
            p2 = np.sqrt(Tmin)
        # Prepare variables
        (p1, p2, alpha, delay, global_phase), new_shape = prepare_variables(
            vars=[p1, p2, alpha, delay, global_phase],
            expand=[True, True, False, False, False],
            length=length,
            give_shape=True)
        alpha = put_in_limits(alpha, "alpha")
        delay = put_in_limits(delay, "delay")
        # Compute the common operations
        sa, ca = (np.sin(alpha), np.cos(alpha))
        ed, edm = (np.exp(1j * delay), np.exp(-1j * delay))
        # Calculate the Jones matrix
        self.M = np.array(
            [[p1 * ca**2 + p2 * sa**2, sa * ca * (p1 - p2) * edm],
             [sa * ca * (p1 - p2) * ed, p2 * ca**2 + p1 * sa**2]])
        self.no_rotation = False
        self.add_global_phase(global_phase)
        self.shape, _ = select_shape(self,
                                             shape_var=new_shape,
                                             shape_fun=shape,
                                             shape_like=shape_like)

        return self

    # @_actualize_
    def diattenuator_azimuth_ellipticity(self,
                                         p1=1,
                                         p2=0,
                                         Tmax=None,
                                         Tmin=None,
                                         azimuth=0,
                                         ellipticity=0,
                                         global_phase=0,
                                         length=1,
                                         shape_like=None,
                                         shape=None):
        """Creates the general diattenuator with orthogonal eigenstates from the characteristic angles of the main eigenstate.

        References:
            J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016) pp 137.

        Parameters:
            p1 (float or numpy.ndarray): [0, 1] Square root of the higher transmission for one eigenstate.  Default: 1.
            p2 (float or numpy.ndarray): [0, 1] Square root of the lower transmission for the other eigenstate.  Default: 0.
            Tmax (float or numpy.ndarray): Maximum transmission. If not None, overrides p1. Default: None.
            Tmin (float or numpy.ndarray): Minimum transmission. If not None, overrides p2. Default: None.
            azimuth (float): [0, pi]: Azimuth.  Default: 0.
            ellipticity (float): [-pi/4, pi/]: Ellipticity angle.  Default: 0.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_matrix): Created object.
        """
        # Prepare variables
        alpha, delay = azimuth_elipt_2_charac_angles(azimuth=azimuth,
                                                     ellipticity=ellipticity)
        # Calculate
        self.diattenuator_charac_angles(p1=p1,
                                        p2=p2,
                                        Tmax=Tmax,
                                        Tmin=Tmin,
                                        alpha=alpha,
                                        delay=delay,
                                        global_phase=global_phase,
                                        length=length,
                                        shape_like=shape_like,
                                        shape=shape)
        return self

    # @_actualize_
    def quarter_waveplate(self,
                          azimuth=0,
                          global_phase=0,
                          length=1,
                          shape_like=None,
                          shape=None):
        """Jones matrix of an ideal quarter-waveplate :math:`\lambda/4`.

        Parameters:
            azimuth (float or numpy.ndarray): rotation angle of the fast state axis. Default: 0.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_matrix): Created object.
        """
        self.retarder_linear(R=90 * degrees,
                             azimuth=azimuth,
                             global_phase=global_phase,
                             length=length,
                             shape_like=shape_like,
                             shape=shape)
        return self

    # @_actualize_
    def half_waveplate(self,
                       azimuth=0,
                       global_phase=0,
                       length=1,
                       shape_like=None,
                       shape=None):
        """Jones matrix of an ideal half-waveplate :math:`\lambda/2`.

        Parameters:
            azimuth (float or numpy.ndarray): Rotation angle of the fast state axis. Default: 0.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_matrix): Created object.
        """
        self.retarder_linear(R=180 * degrees,
                             azimuth=azimuth,
                             global_phase=global_phase,
                             length=length,
                             shape_like=shape_like,
                             shape=shape)
        return self

    # @_actualize_
    def retarder_linear(self,
                        R=90 * degrees,
                        azimuth=0,
                        global_phase=0,
                        length=1,
                        shape_like=None,
                        shape=None):
        """Creates a linear retarder.

        Parameters:
            R (float): [0, pi] Retardance (delay between components). Default: 90 degrees.
            azimuth (float or numpy.ndarray): Rotation angle of the fast state axis. Default: 0.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_matrix): Created object.
        """
        # Prepare variables
        R, new_shape = prepare_variables(vars=[R], expand=[True], length=length, give_shape=True)
        # Calculate
        M = np.array([[np.ones_like(R), np.zeros_like(R)],
                    [np.zeros_like(R), np.exp(-1j * R)]], dtype=complex)
        self.M = M
        self.no_rotation = False
        self.rotate(azimuth)
        self.set_global_phase(global_phase)
        self.shape, _ = select_shape(self,
                                     shape_var=new_shape,
                                     shape_fun=shape,
                                     shape_like=shape_like)

        return self

    # @_actualize_
    def retarder_material(self,
                          ne=1,
                          no=1,
                          d=1 * um,
                          wavelength=0.6328 * um,
                          azimuth=0,
                          global_phase=0,
                          length=1,
                          shape_like=None,
                          shape=None):
        """Creates a retarder using the physical properties of an anisotropic material.

        .. math::  \phi  = 2 \pi (n_e-n_o) d / \lambda.

        Parameters:
            ne (float or numpy.ndarray): Extraordinary index. Default: 1.
            n0 (float or numpy.ndarray): Ordinary index. Default: 1.
            d (float or numpy.ndarray): Thickness of the sheet in microns. Default: 1 um.
            wavelength (float or numpy.ndarray): Wavelength of the illumination. Default: 0.6328 um.
            azimuth (float or numpy.ndarray): Rotation angle of the fast state axis. Default: 0.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_matrix): Created object.
        """
        # Prepare variables
        (ne, no, d, wavelength, azimuth), new_shape = prepare_variables(
             vars=[ne, no, d, wavelength, azimuth],
             expand=[True, False, False, False, False],
             length=length,
             give_shape=True)
        phase = 2 * np.pi * (ne - no) * d / wavelength
        z = np.zeros_like(ne)
        # Calculate
        self.M = np.array([[np.ones_like(phase), np.zeros_like(phase)],
                    [np.zeros_like(phase), np.exp(-1j * phase)]], dtype=complex)
        self.no_rotation = False
        self.rotate(azimuth)
        self.set_global_phase(global_phase)
        self.shape, _ = select_shape(self,
                                     shape_var=new_shape,
                                     shape_fun=shape,
                                     shape_like=shape_like)
        return self

    # @_actualize_
    def retarder_charac_angles(self,
                               R=90 * degrees,
                               alpha=0,
                               delay=0,
                               global_phase=0,
                               length=1,
                               shape_like=None,
                               shape=None):
        """Function that calculates the most general homogeneous diattenuator from the characteristic angles of the fast eigenstate.

        References:
            "Polarized light and the Mueller Matrix approach", J. J. Gil, pp 125.

        Parameters:
            R (float): [0, pi] Retardance (delay between components). Default: 90 degrees.
            alpha (float): [0, pi/2]: tan(alpha) is the ratio between amplitudes of the eigenstates in Jones formalism. Default: 0.
            delay (float): [0, 2*pi]: phase difference between both components of the eigenstates in Jones formalism. Default: 0.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_matrix): Created object.
        """
        # Prepare variables
        (R, alpha, delay, global_phase), new_shape = prepare_variables(
            vars=[R, alpha, delay, global_phase],
            expand=[True, True, True, False],
            length=length,
            give_shape=True)
        alpha = put_in_limits(alpha, "alpha")
        delay = put_in_limits(delay, "delay")
        # Compute the common operations
        sa, ca = (np.sin(alpha), np.cos(alpha))
        s2a, sD = (np.sin(2 * alpha), np.sin(R / 2))
        ed, edm = (np.exp(1j * delay), np.exp(-1j * delay))
        eD, eDm = (np.exp(1j * R / 2), np.exp(-1j * R / 2))
        # Calculate the Jones matrix
        self.M = np.array([[ca**2 * eD + sa**2 * eDm, 1j * s2a * sD * edm],
                           [1j * s2a * sD * ed, ca**2 * eDm + sa**2 * eD]])
        self.remove_global_phase()
        # Rest of operations
        self.no_rotation = False
        self.add_global_phase(global_phase)
        self.shape, _ = select_shape(self,
                                             shape_var=new_shape,
                                             shape_fun=shape,
                                             shape_like=shape_like)
        return self

    # @_actualize_
    def retarder_azimuth_ellipticity(self,
                                     R=90 * degrees,
                                     azimuth=0,
                                     ellipticity=0,
                                     global_phase=0,
                                     length=1,
                                     shape_like=None,
                                     shape=None):
        """Function that calculates the most general homogeneous diattenuator from the characteristic angles of the fast eigenstate.

        References:
            "Polarized light and the Mueller Matrix approach", J. J. Gil, pp 125.

        Parameters:
            R (float): [0, pi] Retardance (delay between components). Default: 90 degrees.
            azimuth (float): [0, pi]: Azimuth. Default: 0.
            ellipticity (float): [-pi/4, pi/4]: Ellipticity. Default: 0.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_matrix): Created object.
        """
        # Prepare variables
        alpha, delay = azimuth_elipt_2_charac_angles(azimuth=azimuth,
                                                     ellipticity=ellipticity)
        # Calculate
        self.retarder_charac_angles(R,
                                    alpha,
                                    delay,
                                    global_phase=global_phase,
                                    length=length,
                                    shape_like=shape_like,
                                    shape=shape)
        return self

    # @_actualize_
    def diattenuator_retarder_linear(self,
                                     p1=1,
                                     p2=0,
                                     Tmax=None,
                                     Tmin=None,
                                     R=0,
                                     azimuth=0,
                                     global_phase=0,
                                     length=1,
                                     shape_like=None,
                                     shape=None):
        """Creates a linear diattenuator retarder with the same
        axes for diattenuation and retardance. At 0 degrees, the matrix is of
        the form:

        .. math:: J\left(\theta=0\right)=\left[\begin{array}{cc}
                    p_{1} & 0\\
                    0 & p_{2}e^{i R}
                    \end{array}\right]'.

        Parameters:
            p1 (float or numpy.ndarray): Field transmission of the fast axis. Default: 1.
            p2 (float or numpy.ndarray): Electric field transmission coefficient of the extinction eigenstate. Default: 0.
            Tmax (float or numpy.ndarray): Maximum transmission. If not None, overrides p1. Default: None.
            Tin (float or numpy.ndarray): Minimum transmission. If not None, overrides p2. Default: None.
            R (float or numpy.ndarray): Retardance. Default: 0.
            azimuth (float or numpy.ndarray): Rotation angle of the high transmission polarizer axis. Default: 0.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_matrix): Created object.
        """
        # Use field transmission coefficients
        if Tmax is not None:
            p1 = np.sqrt(Tmax)
        if Tmin is not None:
            p2 = np.sqrt(Tmin)
        # Prepare variables
        (p1, p2, R, azimuth, global_phase), new_shape = prepare_variables(
            vars=[p1, p2, R, azimuth, global_phase],
            expand=[True, True, False, False, False],
            length=length,
            give_shape=True)
        z = np.zeros_like(p1)
        # Add global Phase
        if global_phase is not 0:
            p1 = p1 * np.exp(1j * global_phase)
            p2 = p2 * np.exp(1j * global_phase)
        # Create the object
        self.M = np.array([[p1, z], [z, p2 * np.exp(1j * R)]])
        self.no_rotation = False
        self.rotate(azimuth)
        self.shape, _ = select_shape(self,
                                             shape_var=new_shape,
                                             shape_fun=shape,
                                             shape_like=shape_like)
        return self

    def diattenuator_retarder_azimuth_ellipticity(self,
                                                  p1=1,
                                                  p2=0,
                                                  Tmax=None,
                                                  Tmin=None,
                                                  R=0,
                                                  azimuth=0,
                                                  ellipticity=0,
                                                  global_phase=0,
                                                  length=1,
                                                  shape_like=None,
                                                  shape=None):
        """Creates the most general homogenous diattenuator retarder from the azimuth and ellipticity of the fast eigenstate.

        Parameters:
            p1 (float or numpy.ndarray): Field transmission of the fast axis. Default: 1.
            p2 (float or numpy.ndarray): Electric field transmission coefficient of the extinction eigenstate. Default: 0.
            Tmax (float or numpy.ndarray): Maximum transmission. If not None, overrides p1. Default: None.
            Tmin (float or numpy.ndarray): Minimum transmission. If not None, overrides p2. Default: None.
            R (float or numpy.ndarray): Retardance. Default: 0.
            azimuth (float or numpy.ndarray): rotation angle of the high transmission polarizer axis. Default: 0.
            ellipticity (float): [-pi/4, pi/]: Ellipticity angle.  Default: 0.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_matrix): Created object.
        """
        # Use field transmission coefficients
        if Tmax is not None:
            p1 = np.sqrt(Tmax)
        if Tmin is not None:
            p2 = np.sqrt(Tmin)
        # Create the two objects
        E1 = Jones_matrix()
        E1.diattenuator_azimuth_ellipticity(p1=p1,
                                            p2=p2,
                                            azimuth=azimuth,
                                            ellipticity=ellipticity,
                                            shape=shape,
                                            shape_like=shape_like,
                                            length=length)
        E2 = Jones_matrix()
        E2.retarder_azimuth_ellipticity(R=R,
                                        azimuth=azimuth,
                                        ellipticity=ellipticity,
                                        shape=shape,
                                        shape_like=shape_like,
                                        length=length)
        # Multiply and extract
        new_obj = E1 * E2
        self.from_matrix(new_obj.M)
        self.shape, _ = new_obj.shape, new_obj.ndim
        # return self

    def diattenuator_retarder_charac_angles(self,
                                            p1=1,
                                            p2=0,
                                            Tmax=None,
                                            Tmin=None,
                                            R=0,
                                            alpha=0,
                                            delay=0,
                                            global_phase=0,
                                            length=1,
                                            shape_like=None,
                                            shape=None):
        """Creates the most general homogenous diattenuator retarder from the characteristic angles of the fast eigenstate.

        Parameters:
            p1 (float or numpy.ndarray): Field transmission of the fast axis. Default: 1.
            p2 (float or numpy.ndarray): Electric field transmission coefficient of the extinction eigenstate. Default: 0.
            Tmax (float or numpy.ndarray): Maximum transmission. If not None, overrides p1. Default: None.
            Tmin (float or numpy.ndarray): Minimum transmission. If not None, overrides p2. Default: None.
            R (float or numpy.ndarray): Retardance. Default: 0.
            alpha (float or numpy.ndarray): [0, pi/2]: tan(alpha) is the ratio between field amplitudes of X and Y components. Default: 0.
            delay (float or numpy.ndarray): [0, 2*pi]: Phase difference between X and Y field components. Default: 0.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_matrix): Created object.
        """
        # Use field transmission coefficients
        if Tmax is not None:
            p1 = np.sqrt(Tmax)
        if Tmin is not None:
            p2 = np.sqrt(Tmin)
        # Create the two objects
        E1, E2 = create_Jones_matrices(N=2)
        E1.diattenuator_charac_angles(p1=p1,
                                      p2=p2,
                                      alpha=alpha,
                                      delay=delay,
                                      shape=shape,
                                      shape_like=shape_like,
                                      length=length)
        E2.retarder_charac_angles(R=R,
                                  alpha=alpha,
                                  delay=delay,
                                  shape=shape,
                                  shape_like=shape_like,
                                  length=length)
        # Multiply and extract
        new_obj = E1 * E2
        self.from_matrix(new_obj.M)
        self.shape, _ = new_obj.shape, new_obj.ndim
        return self

    def general_eigenstates(self,
                            E1,
                            E2=None,
                            p1=1,
                            p2=0,
                            Tmax=None,
                            Tmin=None,
                            R=0,
                            global_phase=0,
                            length=1,
                            shape_like=None,
                            shape=None):
        """Creates the most general optical element from its eigenstates.

        Parameters:
            E1 (Jones_vector): First eigenstate.
            E2 (Jones_vector): Second eigenstate. If None, E2 is taken as the perpendicular state to E1, so the optical object is homogenous. Default: None
            p1 (float or numpy.ndarray): Field transmission of the fast axis. Default: 1.
            p2 (float or numpy.ndarray): Electric field transmission coefficient of the extinction eigenstate. Default: 0.
            Tmax (float or numpy.ndarray): Maximum transmission. If not None, overrides p1. Default: None.
            Tmin (float or numpy.ndarray): Minimum transmission. If not None, overrides p2. Default: None.
            R (float or numpy.ndarray): Retardance. Default: 0.
            global_phase (float or numpy.ndarray): Global phase introduced by the optical element. Default: 0.
            length (int): If final object is of size 1, it is stretched to match this size. Default: 1.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.

        Returns:
            (Jones_matrix): Created object.
        """
        # Use field transmission coefficients
        if Tmax is not None:
            p1 = np.sqrt(Tmax)
        if Tmin is not None:
            p2 = np.sqrt(Tmin)
        # Main calculation
        if E2 is None:
            # Simple case: homogenous case
            az, el = E1.parameters.azimuth_ellipticity()
            self.diattenuator_retarder_azimuth_ellipticity(
                p1=p1,
                p2=p2,
                R=R,
                azimuth=az,
                ellipticity=el,
                shape=shape,
                shape_like=shape_like,
                length=length,
                global_phase=global_phase)
        else:
            # Prepare variables
            length = np.max([length, E1.size, E2.size])
            (p1, p2, R, global_phase), new_shape = prepare_variables(
                vars=[p1, p2, R, global_phase],
                expand=[True, True, False, False],
                length=length,
                give_shape=True)
            # Complicated case: inhomogeneous case
            R = put_in_limits(R, 'Retardance')
            # Create the diagonal matrix with the eigenvalues
            d00 = p1 * np.exp(1j * global_phase)
            d11 = p2 * np.exp(1j * (global_phase + R))
            z = np.zeros_like(d00)
            D = np.array([[d00, z], [z, d11]])
            # Create the matrix with the eigenvectors as columns
            V = np.array([E1.M, E2.M])
            V = np.transpose(V, axes=(1, 0, 2))
            V_inv = inv_pypol(V)
            # Multiply the matrices
            M = matmul_pypol(D, V_inv)
            M = matmul_pypol(V, M)
            # Create the object
            self.from_matrix(M,
                             shape=shape,
                             shape_like=shape_like,
                             length=length,
                             global_phase=global_phase)
        return self




################################################################################
# Parameters
################################################################################


class Parameters_Jones_Matrix(object):
    """Class for Jones Matrix Parameters.

    Parameters:
        self.parent (Jones_matrix): Parent object.
    """

    def __init__(self, Jones_matrix):
        self.parent = Jones_matrix

    def __repr__(self):
        """Print all parameters."""
        self.get_all(verbose=True, draw=True)
        return ''

    def get_all(self, verbose=False, draw=False):
        """Creates a dictionary with all the parameters of Jones Matrix.

        Parameters:
            verbose (bool): If True, print all parameters. Default: False.
            draw (bool): If True, draw all plots/images of the parameters. Default: False.

        Returns:
            (dict): Dictionary with parameters of Jones Matrix.
        """
        dict_params = {}
        dict_params['J00'], dict_params['J01'], dict_params[
            'J10'], dict_params['J11'] = self.components(verbose=verbose,
                                                              draw=draw)
        dict_params['diattenuation'] = self.diattenuation(verbose=verbose,
                                                               draw=draw)
        dict_params['retardance'] = self.retardance(verbose=verbose,
                                                         draw=draw)
        dict_params['global_phase'] = self.global_phase(verbose=verbose,
                                                             draw=draw)
        dict_params['inhomogeneity'] = self.inhomogeneity(verbose=verbose,
                                                               draw=draw)
        dict_params['T_max'], dict_params[
            'T_min'] = self.transmissions(verbose=verbose, draw=draw)
        dict_params['p1'], dict_params[
            'p2'] = self.transmissions(kind='field', verbose=verbose, draw=draw)
        dict_params['mean_transmission'] = self.mean_transmission(
            verbose=verbose, draw=draw)
        dict_params['det'] = self.det(verbose=verbose, draw=draw)
        dict_params['trace'] = self.trace(verbose=verbose, draw=draw)
        dict_params['norm'] = self.norm(verbose=verbose, draw=draw)
        dict_params['v1'], dict_params['v2'], dict_params[
            'e1'], dict_params['e2'] = self.eig(verbose=verbose,
                                                     draw=draw)

        return dict_params

    def matrix(self, shape=None, shape_like=None):
        """Returns the numpy array of Jones matrices.

        Parameters:
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.

        Returns:
            (numpy.ndarray) 2x2xN numpy array.
        """
        shape, _ = select_shape(obj=self.parent,
                                shape_fun=shape,
                                shape_like=shape_like)
        if shape is not None and len(shape) > 1:
            shape = tuple([2, 2] + list(shape))
            M = np.reshape(self.parent.M, shape)
        else:
            M = self.parent.M
        return M

    def components(self,
                   out_number=True,
                   shape_like=None,
                   shape=None,
                   verbose=False,
                   draw=False):
        """Extracts the matrix components of the Jones matrix.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): if True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            J00 (numpy.ndarray): array of the 0, 0 element of the matrix.
            J01 (numpy.ndarray): array of the 0, 1 element of the matrix.
            J10 (numpy.ndarray): array of the 1, 0 element of the matrix.
            J11 (numpy.ndarray): array of the 1, 1 element of the matrix.
        """
        # Calculate the components
        J00 = self.parent.M[0, 0, :]
        J01 = self.parent.M[0, 1, :]
        J10 = self.parent.M[1, 0, :]
        J11 = self.parent.M[1, 1, :]
        # If the result is a number and the user asks for it, return a float
        if out_number and J00.size == 1:
            J00, J01, J10, J11 = (J00[0], J01[0], J10[0], J11[0])
        # Reshape if required
        J00, J01, J10, J11 = reshape([J00, J01, J10, J11],
                                     shape_like=shape_like,
                                     shape_fun=shape,
                                     obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The matrix components of {} are:'.format(
                self.parent.name)
            PrintParam(param=(J00, J01, J10, J11),
                       shape=self.parent.shape,
                       title=('J00', 'J01', 'J10', 'J11'),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        # Return
        return J00, J01, J10, J11

    def inhomogeneity(self,
                      method='val',
                      out_number=True,
                      shape_like=None,
                      shape=None,
                      verbose=False,
                      draw=False):
        """Calculates the inhomogeneity parameter.

        References:
            Method EIG: J.J. Gil, R. Ossikovsky "Polarized light and the Mueller Matrix approach", CRC Press (2016), pp 119.
            Method VAL: "Homogeneous and inhomogeneous Jones matrices", S.Y. Lu and R.A. Chipman, J. Opt. Soc. Am. A/Vol. 11, No. 2 pp. 766 (1994)

        Parameters:
            method (string): Method used for the calculation of the inhomogeneity parameter: EIG uses the eigenstates and VAL uses determinant, norm and trace.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            eta (numpy.ndarray or float): Inhomogeneity parameter.
        """
        if method in ('eig', 'Eig', 'EIG'):
            # Calculate the egenstates
            e1, e2 = self.eigenstates(shape=False)
            # Calculate the parameter
            eta = np.abs(np.sum(np.conj(e1) * e2, axis=0))
        elif method in ('val', 'Val', 'VAL'):
            # Calculate the values
            det = self.det(out_number=False, shape=False)
            trace = self.trace(out_number=False, shape=False)
            norm2 = self.norm(out_number=False, shape=False)**2
            # Calculate the parameter
            a = norm2 - 0.5 * np.abs(trace)**2
            b = 0.5 * np.abs(trace**2 - 4 * det)
            eta = (a - b) / (a + b)
        else:
            raise ValueError('Method {} is not defined'.format(method))
        # If the result is a number and the user asks for it, return a float
        if out_number and eta.size == 1:
            eta = eta[0]
        # Reshape if neccessary
        eta = reshape([eta],
                      shape_like=shape_like,
                      shape_fun=shape,
                      obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The inhomogeneity parameter of {} is:'.format(
                self.parent.name)
            PrintParam(param=eta,
                       shape=self.parent.shape,
                       title='Inhomogeneity',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)

        return eta

    def diattenuation(self,
                      remove_nan=False,
                      out_number=True,
                      shape_like=None,
                      shape=None,
                      verbose=False,
                      draw=False):
        """Calculation of the diattenuation of a Jones Matrix.

        References:
            "Homogeneous and inhomogeneous Jones matrices", S.Y. Lu and R.A. Chipman, J. Opt. Soc. Am. A/Vol. 11, No. 2 pp. 766 (1994)

        Parameters:
            remove_nan (bool): If True, np.nan values are substitued by 0. Default: False.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            D (numpy.ndarray or float): Diattenuation.
        """
        # Calculate eigenvalues, eigenstates and homogeneity
        v1, v2 = self.eigenvalues(out_number=False, shape=False)
        a1, a2 = (np.abs(v1), np.abs(v2))
        eta = self.inhomogeneity(out_number=False, shape=False)
        cond = eta < tol_default**2
        D = np.zeros_like(v1)
        # Case homogenous
        if np.any(cond):
            D[cond] = np.abs(a1[cond]**2 - a2[cond]**2) / (a1[cond]**2 +
                                                           a2[cond]**2)
        # Case inhomogeneous
        cond = ~cond
        if np.any(cond):
            num = 2 * (1 - eta[cond]**2) * a1[cond] * a2[cond]
            den = a1[cond]**2 + a2[cond]**2 - eta[cond]**2 * \
                (v1[cond] * np.conj(v2[cond]) + v2[cond] * np.conj(v1[cond]))
            D[cond] = np.sqrt(1 - (num / den)**2)
        # D must be real, but complex numbers are used during calculation
        D = np.array(D, dtype=float)
        # If there are nans and the user asks for it, change them for 0
        cond = np.isnan(D)
        if remove_nan and np.any(cond):
            D[cond] = 0
        # If the result is a number and the user asks for it, return a float
        if out_number and D.size == 1:
            D = D[0]
        # Reshape if neccessary
        D = reshape([D],
                    shape_like=shape_like,
                    shape_fun=shape,
                    obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The diattenuation of {} is:'.format(self.parent.name)
            PrintParam(param=D,
                       shape=self.parent.shape,
                       title='Diattenuation',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return D

    def polarizance(self,
                    remove_nan=False,
                    out_number=True,
                    shape_like=None,
                    shape=None,
                    verbose=False,
                    draw=False):
        """Calculation of the polarizance of the Jones matrices. In Jones formalism, this is the same as diattenuation.

        Parameters:
            remove_nan (bool): If True, np.nan values are substitued by 0. Default: False.
            out_number (bool): # IDEA: f True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): if True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            P (numpy.ndarray or float): Polarizance.
        """
        # Calculate the diattenuation
        D = self.diattenuation(remove_nan=remove_nan,
                               out_number=out_number,
                               shape=shape,
                               shape_like=shape_like)
        P = D
        # Print the result if required
        if verbose or draw:
            heading = 'The polarizance of {} is:'.format(self.parent.name)
            PrintParam(param=P,
                       shape=self.parent.shape,
                       title='Polarizance',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return P

    def retardance(self,
                   remove_nan=False,
                   out_number=True,
                   shape_like=None,
                   shape=None,
                   verbose=False,
                   draw=False):
        """Calculation of the retardance (delay between eigenstates) of a Jones optical element.

        References:
            "Homogeneous and inhomogeneous Jones matrices", Shih-Yau Lu and  Russell A. Chipman, J. Opt. Soc. Am. A/Vol. 11, No. 2 pp. 766 (1994)

        Parameters:
            remove_nan (bool): If True, np.nan values are substitued by 0. Default: False.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            R (numpy.ndarray or float): Retardance.
        """
        # Calculate the needed values
        v1, v2 = self.eigenvalues(out_number=False, shape=False)
        a1, a2 = (np.abs(v1), np.abs(v2))
        M = np.moveaxis(self.parent.M, -1, 0)
        det = self.det(out_number=False, shape=False)
        trace = self.trace(out_number=False, shape=False)
        norm2 = self.norm(out_number=False, shape=False)**2
        eta = self.inhomogeneity(out_number=False, shape=False)
        # Act differently if the object is homogeneous
        cond1 = eta < tol_default**2
        R = np.zeros_like(eta)
        # Homogeneous case
        if np.any(cond1):
            cond2 = np.abs(det) < tol_default**2
            R[cond1 * cond2] = 2 * np.arccos(
                np.abs(trace[cond1 * cond2]) / np.sqrt(norm2[cond1 * cond2]))
            cond2 = ~cond2
            num = np.abs(trace[cond1 * cond2] +
                         det[cond1 * cond2] * np.conj(trace[cond1 * cond2]) /
                         np.abs(det[cond1 * cond2]))
            den = 2 * np.sqrt(norm2[cond1 * cond2] +
                              2 * np.abs(det[cond1 * cond2]))
            R[cond1 * cond2] = 2 * np.arccos(num / den)
        # Inhomogeneous case
        cond1 = ~cond1
        if np.any(cond1):
            num = (1 - eta[cond1]**2) * (a1[cond1] + a2[cond1])**2
            den = (a1[cond1] + a2[cond1])**2 - eta[cond1]**2 * (
                2 * v1[cond1] * a1[cond1] * a2[cond1] +
                np.conj(v2[cond1] + v2[cond1] * np.conj(v1[cond1])))
            co = np.cos((np.angle(v1[cond1]) - np.angle(v2[cond1])) / 2)
            R[cond1] = 2 * np.arccos(np.sqrt(num / den) * co)
        # D must be real, but complex numbers are used during calculation
        R = np.array(R, dtype=float)
        # If there are nans and the user asks for it, change them for 0
        cond = np.isnan(R)
        if remove_nan and np.any(cond):
            R[cond] = 0
        # If the result is a number and the user asks for it, return a float
        if out_number and R.size == 1:
            R = R[0]
        # Reshape if neccessary
        R = reshape([R],
                    shape_like=shape_like,
                    shape_fun=shape,
                    obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The retardance of {} is (deg.):'.format(
                self.parent.name)
            PrintParam(param=R / degrees,
                       shape=self.parent.shape,
                       title='Retardance (deg.)',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return R

    def global_phase(self,
                     out_number=True,
                     shape_like=None,
                     shape=None,
                     verbose=False,
                     draw=False):
        """Calculates the phase of J00 (which is the reference for global phase in py_pol model).

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray) [0, 2*pi]: Global phase.
        """
        # Calculate phase
        J00, _, _, _ = self.components(out_number=out_number, shape=False)
        phase = np.angle(J00) % (2 * np.pi)
        # Reshape if neccessary
        phase = reshape([phase],
                        shape_like=shape_like,
                        shape_fun=shape,
                        obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The global phase of {} is (deg.):'.format(
                self.parent.name)
            PrintParam(param=phase / degrees,
                       shape=self.parent.shape,
                       title='Global phase (deg.)',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return phase

    def transmissions(self,
                      kind='INTENSITY',
                      out_number=True,
                      shape_like=None,
                      shape=None,
                      verbose=False,
                      draw=False):
        """Calculate the maximum and minimum transmitance of an optical element.

        References:
            Handbook of Optics vol 2. 22.32 (eq.38)

        Parameters:
            kind (str): There are three options, FIELD, INTENSITY or ALL. Defaut: 'INTENSITY'
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            T_max (numpy.ndarray): Maximum intensity transmission.
            T_min (numpy.ndarray): Minimum intensity transmission.
            p1 (numpy.ndarray): Maximum field transmission.
            p2 (numpy.ndarray): Minimum field transmission.
        """
        # Calculate the needed values
        norm2 = self.norm(out_number=out_number,
                          shape=shape,
                          shape_like=shape_like)**2
        det = self.det(out_number=out_number,
                       shape=shape,
                       shape_like=shape_like)
        # Calculate transmissions
        T_max = (norm2 + np.sqrt(norm2**2 - 4 * np.abs(det)**2)) / 2
        T_min = (norm2 - np.sqrt(norm2**2 - 4 * np.abs(det)**2)) / 2
        if kind.upper() in ('FIELD', 'ALL'):
            p1 = np.sqrt(T_max)
            p2 = np.sqrt(T_min)
        # Print the result if required
        if verbose or draw:
            # Intensity
            if kind.upper() in ('INTENSITY', 'ALL'):
                heading = 'The intensity transmissions of {} are:'.format(
                    self.parent.name)
                PrintParam(param=(T_max, T_min),
                           shape=self.parent.shape,
                           title=('Maximum (int.)', 'Minimum (int.)'),
                           heading=heading,
                           verbose=verbose,
                           draw=draw)
            # Field
            if kind.upper() in ('FIELD', 'ALL'):
                heading = 'The field transmissions of {} are:'.format(
                    self.parent.name)
                PrintParam(param=(p1, p2),
                           shape=self.parent.shape,
                           title=('Maximum (int.)', 'Minimum (int.)'),
                           heading=heading,
                           verbose=verbose,
                           draw=draw)
        # Return
        ret = []
        if kind.upper() in ('INTENSITY', 'ALL'):
            ret += [T_max, T_min]
        if kind.upper() in ('FIELD', 'ALL'):
            ret += [p1, p2]
        return ret

    def mean_transmission(self,
                          out_number=True,
                          shape_like=None,
                          shape=None,
                          verbose=False,
                          draw=False):
        """Calculate the mean intensity transmitance of an optical element.

        References:
            Handbook of Optics vol 2. 22.32 (eq.38)

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray) [0, 1]: Result
        """
        # Calculate the maximum and minimum transmissions
        T_max, T_min = self.transmissions(out_number=out_number,
                                          shape_like=shape_like,
                                          shape=shape)
        T = (T_max - T_min) / 2
        # Print the result if required
        if verbose or draw:
            # Eigenvalues
            heading = 'The mean transmission of {} is:'.format(
                self.parent.name)
            PrintParam(param=T,
                       shape=self.parent.shape,
                       title=('Mean trans.'),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)

        return T

    def eig(self,
            as_objects=False,
            out_number=True,
            shape_like=None,
            shape=None,
            verbose=False,
            draw=False):
        """
        Calculates the eigenvalues and eigenstates of the Jones object.

        Parameters:
            as_objects (bool): If True, the eigenvectors are extracted as py_pol objects. Default: False.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): if True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            v1 (numpy.ndarray or float): First eigenvalue.
            v2 (numpy.ndarray or float): Second eigenvalue.
            e1 (numpy.ndarray or Jones_vector): First eigenstate.
            e2 (numpy.ndarray or Jones_vector): Second eigenstate.

        TODO: Maybe give values and states together as a matrix
        """
        # Differenciate between conjugate symmetric matrices to assure real eigenvalues and orthogonal eigenvectors.
        cond = self.parent.checks.is_conjugate_symmetric(out_number=False,
                                                         shape=False)
        # Calculate the eigenstates
        M = np.moveaxis(self.parent.M, -1, 0)
        val, vect = np.linalg.eig(M)
        if np.any(cond):
            val2, vect2 = np.linalg.eigh(M)
        # Order the values in the py_pol way
        v1, v2 = (val[:, 0], val[:, 1])
        e1 = np.array([vect[:, 0, 0], vect[:, 1, 0]])
        e2 = np.array([vect[:, 0, 1], vect[:, 1, 1]])
        if np.any(cond):
            v1[cond], v2[cond] = (val2[cond, 0], val2[cond, 1])
            e1[0, cond] = vect2[cond, 0, 0]
            e1[1, cond] = vect2[cond, 1, 0]
            e2[0, cond] = vect2[cond, 0, 1]
            e2[1, cond] = vect2[cond, 1, 1]
        if v1.size == 1 and v1.ndim > 1:
            v1, v2 = (v1[0], v2[0])
        # Size 1 objects need some care
        if out_number and v1.size == 1:
            v1, v2 = (v1[0], v2[0])
        if ~out_number and v1.size == 1:
            v1, v2 = (np.array([v1]), np.array([v2]))
        # Reshape if neccessary
        v1, v2 = reshape([v1, v2],
                         shape_like=shape_like,
                         shape_fun=shape,
                         obj=self.parent)
        e1x, e1y = (e1[0, :], e1[1, :])
        e2x, e2y = (e2[0, :], e2[1, :])
        e1x, e1y, e2x, e2y = reshape([e1x, e1y, e2x, e2y],
                                     shape_like=shape_like,
                                     shape_fun=shape,
                                     obj=self.parent)
        new_shape = [2] + list(e1x.shape)
        if len(new_shape) > 2:
            e1 = np.reshape(e1, new_shape)
            e2 = np.reshape(e2, new_shape)
        # Print the result if required
        if verbose or draw:
            # Eigenvalues
            heading = 'The eigenvalues of {} are:'.format(self.parent.name)
            PrintParam(param=(v1, v2),
                       shape=self.parent.shape,
                       title=('v1', 'v2'),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
            # Eigenvectors
            heading = 'The eigenvectors of {} are:'.format(self.parent.name)
            PrintParam(param=(e1x, e1y, e2x, e2y),
                       shape=self.parent.shape,
                       title=('e1x', 'e1y', 'e2x', 'e2y'),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)

        # Return
        if as_objects:
            E1 = Jones_vector(self.parent.name + ' e1')
            E1.from_matrix(e1, shape=shape, shape_like=shape_like)
            E2 = Jones_vector(self.parent.name + ' e2')
            E2.from_matrix(e2, shape=shape, shape_like=shape_like)
        else:
            E1, E2 = (e1, e2)

        return v1, v2, E1, E2

    def eigenvectors(self,
                     shape_like=None,
                     shape=None,
                     verbose=False,
                     draw=False):
        """
        Calculates the eigenvectors of the Jones object.

        Parameters:
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            e1 (numpy.ndarray): 2xN first eigenvectors matrix.
            e2 (numpy.ndarray): 2xN second eigenvectors matrix.
        """
        # Calculate
        _, _, e1, e2 = self.eig(out_objects=True,
                                shape=shape,
                                shape_like=shape_like)
        # Print the result if required
        if verbose or draw:
            # Eigenvectors
            heading = 'The eigenvectors of {} are:'.format(self.parent.name)
            e1x, e1y = (e1[0, :], e1[1, :])
            e2x, e2y = (e2[0, :], e2[1, :])
            e1x, e1y, e2x, e2y = reshape([e1x, e1y, e2x, e2y],
                                         shape_like=shape_like,
                                         shape_fun=shape,
                                         obj=self.parent)
            PrintParam(param=(e1x, e1y, e2x, e2y),
                       shape=self.parent.shape,
                       title=('e1x', 'e1y', 'e2x', 'e2y'),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)

        return e1, e2

    def eigenstates(self,
                    shape_like=None,
                    shape=None,
                    verbose=False,
                    draw=False):
        """
        Calculates the eigenstates of the Jones object. Very similar to eigenvectors, but the output are Jones_vector objects.

        Parameters:
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): if True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            e1 (Jones_vector): First eigenstate.
            e2 (Jones_vector): Second eigenstate.
        """
        # Calculate
        _, _, E1, E2 = self.eig(as_objects=True)
        # Print the result if required
        if verbose or draw:
            # Eigenvectors
            heading = 'The eigenvectors of {} are:'.format(self.parent.name)
            e1x, e1y = (E1.M[0, :], E1.M[1, :])
            e2x, e2y = (E2.M[0, :], E2.M[1, :])
            e1x, e1y, e2x, e2y = reshape([e1x, e1y, e2x, e2y],
                                         shape_like=shape_like,
                                         shape_fun=shape,
                                         obj=self.parent)
            PrintParam(param=(e1x, e1y, e2x, e2y),
                       shape=self.parent.shape,
                       title=('e1x', 'e1y', 'e2x', 'e2y'),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)

        return E1, E2

    def eigenvalues(self,
                    out_number=True,
                    shape_like=None,
                    shape=None,
                    verbose=False,
                    draw=False):
        """
        Calculates the eigenvalues and eigenstates of the Jones object.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            v1 (numpy.ndarray or float): First eigenvalue.
            v2 (numpy.ndarray or float): Second eigenvalue.
        """
        # Differenciate between conjugate symmetric matrices to assure real eigenvalues and orthogonal eigenvectors.
        cond = self.parent.checks.is_conjugate_symmetric(out_number=False,
                                                         shape=False)
        # Calculate the eigenstates
        M = np.moveaxis(self.parent.M, -1, 0)
        val = np.linalg.eigvals(M)
        if np.any(cond):
            val2 = np.linalg.eigvalsh(M)
        # Order the values in the py_pol way
        v1, v2 = (val[:, 0], val[:, 1])
        if np.any(cond):
            v1[cond], v2[cond] = (val2[cond, 0], val2[cond, 1])
        if v1.size == 1 and v1.ndim > 1:
            v1, v2 = (v1[0], v2[0])
        # Size 1 objects need some care
        if out_number and v1.size == 1:
            v1, v2 = (v1[0], v2[0])
        if ~out_number and v1.size == 1:
            v1, v2 = (np.array([v1]), np.array([v2]))
        # Reshape if neccessary
        v1, v2 = reshape([v1, v2],
                         shape_like=shape_like,
                         shape_fun=shape,
                         obj=self.parent)
        # Print the result if required
        if verbose or draw:
            # Eigenvalues
            heading = 'The eigenvalues of {} are:'.format(self.parent.name)
            PrintParam(param=(v1, v2),
                       shape=self.parent.shape,
                       title=('v1', 'v2'),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)

        return v1, v2

    def det(self,
            out_number=True,
            shape_like=None,
            shape=None,
            verbose=False,
            draw=False):
        """
        Calculates the determinants of the Jones matrices.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray, float or complex): Result.
        """
        # Calculate the eigenstates
        M = np.moveaxis(self.parent.M, -1, 0)
        det = np.linalg.det(M)
        # If the result is a number and the user asks for it, return a float
        if out_number and det.size == 1:
            det = det[0]
        # Reshape if neccessary
        det = reshape([det],
                      shape_like=shape_like,
                      shape_fun=shape,
                      obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The determinant of {} is:'.format(self.parent.name)
            PrintParam(param=det,
                       shape=self.parent.shape,
                       title='Determinant',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return det

    def trace(self,
              out_number=True,
              shape_like=None,
              shape=None,
              verbose=False,
              draw=False):
        """
        Calculates the trace of the Jones matrices.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray, float or complex): Result.
        """
        # Calculate the eigenstates
        trace = np.trace(self.parent.M)
        # If the result is a number and the user asks for it, return a float
        if out_number and trace.size == 1:
            trace = trace[0]
        # Reshape if neccessary
        trace = reshape([trace],
                        shape_like=shape_like,
                        shape_fun=shape,
                        obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The trace of {} is:'.format(self.parent.name)
            PrintParam(param=trace,
                       shape=self.parent.shape,
                       title='Trace',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return trace

    def norm(self,
             out_number=True,
             shape_like=None,
             shape=None,
             verbose=False,
             draw=False):
        """
        Calculates the Frobenius norm of the Jones matrices.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray, float or complex): Result.
        """
        # Calculate the eigenstates
        norm = np.linalg.norm(self.parent.M, axis=(0, 1))
        # If the result is a number and the user asks for it, return a float
        if out_number and norm.size == 1:
            norm = norm[0]
        # Reshape if neccessary
        norm = reshape([norm],
                       shape_like=shape_like,
                       shape_fun=shape,
                       obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = 'The norm of {} is:'.format(self.parent.name)
            PrintParam(param=norm,
                       shape=self.parent.shape,
                       title='Norm',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return norm


################################################################################
# Checks
################################################################################


class Checks_Jones_Matrix(object):
    """Class for Jones matrix checks.

    Parameters:
        self.parent (Jones_matrix): Parent object.
    """

    def __init__(self, Jones_matrix):
        self.parent = Jones_matrix

    def __repr__(self):
        """Print all parameters."""
        self.get_all(verbose=True, draw=True)
        return ''

    def get_all(self, verbose=False, draw=False):
        """Creates a dictionary with all the parameters of Jones Matrix.

        Parameters:
            verbose (bool): If True, print all parameters. Default: False.
            draw (bool): If True, draw all plots/images of the parameters. Default: False.

        Returns:
            (dict): Dictionary with parameters of Jones Matrix.
        """
        dict_params = {}
        dict_params['is_physical'] = self.is_physical(verbose=verbose,
                                                           draw=draw)
        dict_params['is_homogeneous'] = self.is_homogeneous(
            verbose=verbose, draw=draw)
        dict_params['is_retarder'] = self.is_retarder(verbose=verbose,
                                                           draw=draw)
        dict_params['is_diattenuator'] = self.is_diattenuator(
            verbose=verbose, draw=draw)
        return dict_params

    def is_physical(self,
                    all_info=False,
                    out_number=True,
                    shape_like=None,
                    shape=None,
                    verbose=False,
                    draw=False):
        """
        Verifies that a Jones matrix is physically realizable. Take into account that amplifiers are not included in this category.

        References:
            R. Martinez-Herrero, P.M. Mejias, G.Piquero "Characterization of partially polarized light fields" Springer series in Optical sciences (2009) ISBN 978-3-642-01326-3, page 3, eqs. 1.4a and 1.4b.

        Parameters:
            all_info (bool): If True, the method returns the information regarding each condition separately. Default: False.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Result.
        """
        # Calculate
        M = np.moveaxis(self.parent.M, -1, 0)
        # H = np.transpose(self.parent.M).conjugate()
        det1 = np.linalg.det(M)
        # trace1 = np.trace(M * H, axis1=1, axis2=2)
        J00, J01, J10, J11 = self.parent.parameters.components(shape=False)
        trace1 = np.abs(J00)**2 + np.abs(J01)**2 + np.abs(J10)**2 + np.abs(
            J11)**2
        condition1 = np.abs(det1) <= 1  # eq. 1.4a
        condition2 = (trace1 >= 0) * (trace1 <= 2)
        cond = condition1 * condition2
        # Reshape if required
        if all_info:
            condition1, condition2 = reshape([condition1, condition2],
                                             shape_like=shape_like,
                                             shape_fun=shape,
                                             obj=self.parent)
        else:
            cond = reshape([cond],
                           shape_like=shape_like,
                           shape_fun=shape,
                           obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = '{} is physically realizable:'.format(self.parent.name)
            if all_info:
                PrintParam(param=(condition1, condition2),
                           shape=self.parent.shape,
                           title=('Determinant cond.', 'Trace cond.'),
                           heading=heading,
                           verbose=verbose,
                           draw=draw)
            else:
                PrintParam(param=(cond),
                           shape=self.parent.shape,
                           title=('Physical'),
                           heading=heading,
                           verbose=verbose,
                           draw=draw)
        # Return
        if all_info:
            return condition1, condition2
        else:
            return cond

    def is_homogeneous(self,
                       out_number=True,
                       shape_like=None,
                       shape=None,
                       verbose=False,
                       draw=False):
        """
        Determines if matrix is homogeneous (the two eigenstates are orthogonal) or not.

        References:
            "Homogeneous and inhomogeneous Jones matrices", S.Y. Lu and R.A. Chipman, J. Opt. Soc. Am. A/Vol. 11, No. 2 pp. 766 (1994)

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Result.
        """
        # Calculate the inhomogeneity parameter
        eta = self.parent.parameters.inhomogeneity(out_number=out_number)
        cond = eta < tol_default**2
        # Reshape if neccessary
        cond = reshape([cond],
                       shape_like=shape_like,
                       shape_fun=shape,
                       obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = '{} is homogeneous:'.format(self.parent.name)
            PrintParam(param=cond,
                       shape=self.parent.shape,
                       title='Homogeneous',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)

        return cond

    def is_retarder(self,
                    out_number=True,
                    shape_like=None,
                    shape=None,
                    verbose=False,
                    draw=False):
        """
        Determines if matrix is an homogeneous retarder. The condition is that the Jones matrix must be unitary ($$J^{\dagger}=J^{-1}$$).

        References:
            "Polarized light and the Mueller Matrix approach", J. J. Gil, pp 123.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Result.
        """
        # Calculate the hermitian conjugate
        H = self.parent.hermitian(keep=True)
        # Check if it is the inverse matrix
        H = H * self.parent
        I = Jones_matrix()
        I.vacuum(length=H.size)
        dif = H - I
        dif = dif.parameters.norm(out_number=out_number, shape=False)
        cond = dif < tol_default
        # Reshape if neccessary
        cond = reshape([cond],
                       shape_like=shape_like,
                       shape_fun=shape,
                       obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = '{} is an homogeneous retarder:'.format(self.parent.name)
            PrintParam(param=cond,
                       shape=self.parent.shape,
                       title='Retarder',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)

        return cond

    def is_diattenuator(self,
                        out_number=True,
                        shape_like=None,
                        shape=None,
                        verbose=False,
                        draw=False):
        """
        Determines if matrix is an homogeneous diattenuator. The condition is that the Jones matrix must be hermitian ($$J^{\dagger}=J$$).

        References:
            "Polarized light and the Mueller Matrix approach", J. J. Gil, pp 123.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Result.
        """
        # Calculate the hermitian conjugate
        H = self.parent.hermitian(keep=True)
        # Check if it is the inverse matrix
        dif = H - self.parent
        dif = dif.parameters.norm(out_number=out_number, shape=False)
        cond = dif < tol_default
        # Reshape if neccessary
        cond = reshape([cond],
                       shape_like=shape_like,
                       shape_fun=shape,
                       obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = '{} is an homogeneous diattenuator:'.format(
                self.parent.name)
            PrintParam(param=cond,
                       shape=self.parent.shape,
                       title='Diattenuator',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)

        return cond

    def is_polarizer(self,
                     out_number=True,
                     shape_like=None,
                     shape=None,
                     verbose=False,
                     draw=False):
        """
        Determines if matrix is an homogeneous polarizer. In Jones formalism, there is no difference between homogeneous diattenuators and polarizers.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Result.
        """
        # Calculate if it is a diattenuator
        cond = self.is_diattenuator(out_number=True,
                                    shape_like=None,
                                    shape=None)
        # Reshape if neccessary
        cond = reshape([cond],
                       shape_like=shape_like,
                       shape_fun=shape,
                       obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = '{} is an homogeneous polarizer:'.format(
                self.parent.name)
            PrintParam(param=cond,
                       shape=self.parent.shape,
                       title='Polarizer',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)

        return cond

    def is_symmetric(self,
                     out_number=True,
                     shape_like=None,
                     shape=None,
                     verbose=False,
                     draw=False):
        """
        Determines if the object matrix is symmetric (i.e. $$J = J^T$$).

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Result.
        """
        # Calculate the matrix components
        _, J01, J10, _ = self.parent.parameters.components(out_number=False,
                                                           shape=False)
        # See if J01 is equal to J10
        cond = np.abs(J01 - J10) < tol_default**2
        # Reshape if neccessary
        cond = reshape([cond],
                       shape_like=shape_like,
                       shape_fun=shape,
                       obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = '{} is symmetric:'.format(self.parent.name)
            PrintParam(param=cond,
                       shape=self.parent.shape,
                       title='Symmetric',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)

        return cond

    def is_conjugate_symmetric(self,
                               out_number=True,
                               shape_like=None,
                               shape=None,
                               verbose=False,
                               draw=False):
        """
        Determines if the object matrix is conjugate symmetric (i.e. $$J = J^{\dagger}$$).

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Result.
        """
        # Calculate the matrix components
        J00, J01, J10, J11 = self.parent.parameters.components(
            out_number=False, shape=False)
        # See if J01 is equal to J10
        cond1 = np.abs(J01 - np.conj(J10)) < tol_default**2
        cond2 = np.abs(J00 - np.conj(J00)) < tol_default**2
        cond3 = np.abs(J11 - np.conj(J11)) < tol_default**2
        cond = cond1 * cond2 * cond3
        # Reshape if neccessary
        cond = reshape([cond],
                       shape_like=shape_like,
                       shape_fun=shape,
                       obj=self.parent)
        # Print the result if required
        if verbose or draw:
            heading = '{} is symmetric:'.format(self.parent.name)
            PrintParam(param=cond,
                       shape=self.parent.shape,
                       title='Symmetric',
                       heading=heading,
                       verbose=verbose,
                       draw=draw)

        return cond

    def is_eigenstate(self, S, out_number=True, shape_like=None, shape=None, verbose=False, draw=False):
        """
        Determines if the vector S is an eigenstate of the object.

        Parameters:
            S (Stokes or Jones_vector): State to test.
            out_number (bool): if True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): if True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            (numpy.ndarray or bool): Result.
        """
        # Multiply
        Sout = self.parent * S
        # Check if S and Sout are proportional
        prop = Sout.M[0,:] / S.M[0,:]
        cond1 = np.abs(Sout.M[1,:] / S.M[1,:] - prop) < tol_default
        if Sout.type == "Stokes":
            cond2 = np.abs(Sout.M[2,:] / S.M[2,:] - prop) < tol_default
            cond3 = np.abs(Sout.M[3,:] / S.M[3,:] - prop) < tol_default
            cond = cond1 * cond2 * cond3
        else:
            cond = cond1
        # Reshape if neccessary
        cond = reshape([cond],
              shape_like=shape_like,
              shape_fun=shape,
              obj=self.parent)
        # Print the result if required
        if verbose or draw:
           heading = '{} is an eigenstate of {}:'.format(S.name, self.parent.name)
           PrintParam(param=cond,
                      shape=self.parent.shape,
                      title='Eigenstate',
                      heading=heading,
                      verbose=verbose,
                      draw=draw)

        return cond


######################################################################
# Checks
######################################################################


class Analysis_Jones_Matrix(object):
    """Class for Jones matrix analysis.

    Parameters:
        self.parent (Jones_matrix): Parent object.
    """

    def __init__(self, Jones_matrix):
        self.parent = Jones_matrix


    def decompose_pure(self,
                       decomposition='RP',
                       all_info=False,
                       out_number=True,
                       shape_like=None,
                       shape=None,
                       verbose=False,
                       draw=False,
                       transmissions='ALL',
                       angles="ALL"):
        """Polar decomposition of a pure Mueller matrix in a retarder and a diattenuator.

        References:
            "Homogeneous and inhomogeneous Jones matrices", S.Y. Lu and R.A. Chipman, J. Opt. Soc. Am. A/Vol. 11, No. 2 pp. 766 (1994)

        Parameters:
            decomposition (string): string with the order of the elements: retarder (R) or diattenuator/polarizer (D or P).
            all_info (bool): If True, the method returns the information regarding each condition separately. Default: False.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.
            transmissions (string): Determines the type of transmission output, FIELD, INTENSITY or ALL. Default: ALL.
            angles (string): Determines the type of angles output, CHARAC (characteristic angles), AZIMUTH (azimuth and ellipticity) or ALL. Default: ALL.

        Returns:
            Jr (Jones_matrix): Jones matrix object of the retarder.
            Jd (Jones_matrix): Jones matrix object of the diattenuator.
            dict_param (dictionary): Dictionary with the 9 parameters (7 independent) of both the retarder and the diattenuator (only if all_info = True).
        """
        # Calculate the matrix Jh * J
        Jh = self.parent.hermitian(keep=True)
        J = Jh * self.parent
        # Extract eigenvalues and eigenvectors
        v1, v2, e1, e2 = J.parameters.eig(shape=False, out_number=False)
        # Calculate the auxiliar eigenvectors
        J00, J01, J10, J11 = self.parent.parameters.components(
            shape=False, out_number=False)
        e3 = np.array(
            [J00 * e1[0, :] + J01 * e1[1, :], J10 * e1[0, :] + J11 * e1[1, :]])
        e4 = np.array(
            [J00 * e2[0, :] + J01 * e2[1, :], J10 * e2[0, :] + J11 * e2[1, :]])
        n3 = np.linalg.norm(e3, axis=0)
        n4 = np.linalg.norm(e4, axis=0)
        # Normalize eigenvectors
        e3 = e3 / n3
        e4 = e4 / n4
        # Calculate the auxiliar matrices
        J1, J2, J3, J4 = create_Jones_matrices(('J1', 'J2', 'J3', 'J4'))
        if decomposition[0] in ('r', 'R'):
            J1.from_components(
                (np.abs(e1[0, :])**2, e1[0, :] * np.conj(e1[1, :]),
                 e1[1, :] * np.conj(e1[0, :]), np.abs(e1[1, :])**2))
            J2.from_components(
                (np.abs(e2[0, :])**2, e2[0, :] * np.conj(e2[1, :]),
                 e2[1, :] * np.conj(e2[0, :]), np.abs(e2[1, :])**2))
        else:
            J1.from_components(
                (np.abs(e3[0, :])**2, e3[0, :] * np.conj(e3[1, :]),
                 e3[1, :] * np.conj(e3[0, :]), np.abs(e3[1, :])**2))
            J2.from_components(
                (np.abs(e4[0, :])**2, e4[0, :] * np.conj(e4[1, :]),
                 e4[1, :] * np.conj(e4[0, :]), np.abs(e4[1, :])**2))
        J3.from_components(
            (e3[0, :] * np.conj(e1[0, :]), e3[0, :] * np.conj(e1[1, :]),
             e3[1, :] * np.conj(e1[0, :]), e3[1, :] * np.conj(e1[1, :])))
        J4.from_components(
            (e4[0, :] * np.conj(e2[0, :]), e4[0, :] * np.conj(e2[1, :]),
             e4[1, :] * np.conj(e2[0, :]), e4[1, :] * np.conj(e2[1, :])))
        # Calculate the matrices of the retarder and the diattenuator
        Jd = np.sqrt(v1) * J1 + np.sqrt(v2) * J2
        Jr = J3 + J4
        # Update shapes
        Jr.shape, _ = select_shape(self.parent,
                                         shape_fun=shape,
                                         shape_like=shape_like)
        Jd.shape, _ = (Jr.shape, Jr.ndim)
        # Fix names
        if change_names:
            Jd.name = self.parent.name + ' Diattenuator'
            Jr.name = self.parent.name + ' Retarder'
        else:
            Jd.name = self.parent.name
            Jr.name = self.parent.name
        # Calculate error
        if decomposition[0] in ('r', 'R'):
            J = Jr * Jd
        else:
            J = Jd * Jr
        J = J - self.parent
        error = J.parameters.norm(shape=shape, shape_like=shape_like)

        # Print the result if required
        if all_info or verbose or draw:
            if verbose or draw:
                print("\n------------------------------------------------------")
                print('Polar decomposition of {} as M = {}.'.format(
                    self.parent.name, decomposition))
            # Diattenuator
            trans, ang = Jd.analysis.diattenuator(transmissions=transmissions,
                                                  angles=angles,
                                                  out_number=out_number,
                                                  verbose=verbose,
                                                  draw=draw)

            # Retarder
            R, ang2 = Jr.analysis.retarder(angles=angles,
                                           out_number=out_number,
                                           shape=shape,
                                           shape_like=shape_like,
                                           verbose=verbose,
                                           draw=draw)

            # Error
            if verbose or draw:
                heading = '{} decomposition mean square error:'.format(
                    self.parent.name)
                PrintParam(param=[error],
                           shape=self.parent.shape,
                           title=['MSE'],
                           heading=heading,
                           verbose=verbose,
                           draw=draw)

        # Extract info from matrices
        if all_info:
            parameters = {}
            parameters['error'] = error
            # Diattenuator
            if transmissions.upper() == 'FIELD':
                parameters['p1'], parameters['p2'] = trans
            elif transmissions.upper() == 'INTENSITY':
                parameters['Tmax'], parameters['Tmin'] = trans
            else:
                parameters['Tmax'], parameters['Tmin'], parameters[
                    'p1'], parameters['p2'] = trans
            if angles.upper() == 'CHARAC':
                parameters['alpha D'], parameters['delay D'] = ang
            elif angles.upper() == 'ALL':
                parameters['alpha D'], parameters['delay D'], parameters[
                    'azimuth D'], parameters['ellipticity D'] = ang
            else:
                parameters['azimuth D'], parameters['ellipticity D'] = ang
            # Retarder
            parameters['R'] = R
            if angles.upper() == 'CHARAC':
                parameters['alpha R'], parameters['delay R'] = ang2
            elif angles.upper() == 'ALL':
                parameters['alpha R'], parameters['delay R'], parameters[
                    'azimuth R'], parameters['ellipticity R'] = ang2
            else:
                parameters['azimuth R'], parameters['ellipticity R'] = ang2

        # Return
        if all_info:
            return Jr, Jd, parameters
        else:
            return Jr, Jd

    def diattenuator(self,
                     transmissions='ALL',
                     angles="ALL",
                     out_number=True,
                     shape_like=None,
                     shape=None,
                     verbose=False,
                     draw=False):
        """Analyzes the properties of the optical objects as a diattenuator.

        Parameters:
            transmissions (string): Determines the type of transmission output, FIELD, INTENSITY or ALL. Default: ALL.
            angles (string): Determines the type of angles output, CHARAC (characteristic angles), AZIMUTH (azimuth and ellipticity) or ALL. Default: ALL.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            dict_param (dictionary): Dictionary of parameters.
        """
        # Calculate transmissions
        trans = self.parent.parameters.transmissions(kind=transmissions,
                                                     out_number=out_number,
                                                     shape=shape,
                                                     shape_like=shape_like)
        # Calculate the eigenstates
        _, E1 = self.parent.parameters.eigenstates(shape=shape,
                                                   shape_like=shape_like)
        E1.name = self.parent.name
        # Calculate the parameters of the eigenstates
        ang, title_ang = ([], [])
        if angles in ('ALL', 'All', 'all', 'CHARAC', 'Charac', 'charac'):
            alpha, delay = E1.parameters.charac_angles(out_number=out_number,
                                                       shape=shape,
                                                       shape_like=shape_like)
            ang += [alpha, delay]
            title_ang += ['Alpha', 'Delay']
        if angles in ('ALL', 'All', 'all', 'AZIMUTH', 'Azimuth', 'azimuth'):
            az, el = E1.parameters.azimuth_ellipticity(out_number=out_number,
                                                       shape=shape,
                                                       shape_like=shape_like)
            ang += [az, el]
            title_ang += ['Azimuth', 'Ellipticity angle']

        # Print the result if required
        if verbose or draw:
            # Transform angles to degrees for representation
            angles_rep = []
            for a in ang:
                angles_rep.append(a / degrees)
            if transmissions in ('INTENSITY', 'Intensity', 'intensity'):
                title_trans = ['Max. transmission', 'Min. transmission']
            elif transmissions in ('FIELD', 'Field', 'field'):
                title_trans = ['p1', 'p2']
            else:
                title_trans = [
                    'Max. transmission', 'Min. transmission', 'p1', 'p2'
                ]

            print('\nAnalysis of {} as polarizer:\n'.format(self.parent.name))
            heading = '- Transmissions of {} are:'.format(self.parent.name)
            PrintParam(param=trans,
                       shape=self.parent.shape,
                       title=title_trans,
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
            heading = '- Angles of {} are:'.format(self.parent.name)

            PrintParam(param=angles_rep,
                       shape=self.parent.shape,
                       title=title_ang,
                       heading=heading,
                       verbose=verbose,
                       draw=draw)
        return trans, ang

    def polarizer(self,
                  out_number=True,
                  shape_like=None,
                  shape=None,
                  verbose=False,
                  draw=False):
        """Analyzes the properties of the optical objects as a polarizer. In Jones formalism, this is the same as analyzing the element as a diattenuator.

        Parameters:
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            dict_param (dictionary): Dictionary of parameters.
        """
        return self.diattenuator(out_number=out_number,
                                 shape=shape,
                                 shape_like=shape_like,
                                 verbose=verbose,
                                 draw=draw)

    def retarder(self,
                 angles="ALL",
                 out_number=True,
                 shape_like=None,
                 shape=None,
                 verbose=False,
                 draw=False):
        """Analyzes the properties of the optical objects as a retarder.

        Parameters:
            angles (string): Determines the type of angles output, CHARAC (characteristic angles), AZIMUTH (azimuth and ellipticity) or ALL. Default: ALL.
            out_number (bool): If True and the result is a 1x1 array, return a number instead. Default: True.
            shape_like (numpy.ndarray or py_pol object): Use the shape of this object. Default: None.
            shape (tuple or list): If no shape_like array is given, use this shape instead. Default: None.
            verbose (bool): If True prints the parameter. Default: False.
            draw (bool): If True and the object is a 1D or 2D, plot it. Default: False.

        Returns:
            dict_param (dictionary): Dictionary of parameters.
        """
        # Calculate retardance
        R = self.parent.parameters.retardance(out_number=out_number,
                                              shape=shape,
                                              shape_like=shape_like)
        # Calculate the components
        comp = self.parent.parameters.components(shape=shape,
                                                 shape_like=shape_like)
        phase_01 = np.angle(comp[1])
        phase_10 = np.angle(comp[2])
        # Calculate the angles
        alpha = 0.5 * np.arcsin(np.abs(comp[1]) / np.sin(R / 2))
        gp = (phase_10 + phase_01 - np.pi) / 2
        delay = (phase_10 - phase_01) / 2
        # Correct delayP
        cond = (gp < -np.pi / 2) + (gp > np.pi / 2)
        if np.any(cond):
            delay[cond] = delay[cond] - np.pi
        delay = put_in_limits(delay, 'delay')
        # Correct alpha
        # aux1 = comp[0] * np.exp(-1j * gp)
        # aux2 = np.cos(alpha)**2 * np.exp(
        #     1j * R / 2) + np.sin(alpha)**2 * np.exp(-1j * R / 2)
        # aux1 = np.abs(aux1 - aux2)
        # cond = (aux1 > tol_default) * (aux1 < np.pi / 2)
        # if np.any(cond):
        #     alpha[cond] = np.pi / 2 - alpha[cond]
        Jaux = Jones_matrix()
        Jaux.retarder_charac_angles(R=R, alpha=np.pi / 2 - alpha, delay=delay)
        dif = self.parent - Jaux
        cond = dif.parameters.norm() < tol_default
        if np.any(cond):
            alpha[cond] = np.pi / 2 - alpha[cond]

        # Calculate the parameters of the eigenstates
        title_ang, ang = ([], [])
        if angles in ('ALL', 'All', 'all', 'CHARAC', 'Charac', 'charac'):
            ang += [alpha, delay]
            title_ang += ['Alpha', 'Delay']
        if angles in ('ALL', 'All', 'all', 'AZIMUTH', 'Azimuth', 'azimuth'):
            az, el = charac_angles_2_azimuth_elipt(alpha, delay)
            ang += [az, el]
            title_ang += ['Azimuth', 'Ellipticity']

        # Print the result if required
        if verbose or draw:
            # Transform angles to degrees for representation
            angles_rep = []
            for a in ang:
                angles_rep.append(a / degrees)

            print('\nAnalysis of {} as retarder:\n'.format(self.parent.name))
            heading = '- Retardance of {} is:'.format(self.parent.name)
            PrintParam(param=(R / degrees),
                       shape=self.parent.shape,
                       title=('Retardance'),
                       heading=heading,
                       verbose=verbose,
                       draw=draw)

            heading = '- Angles of {} are:'.format(self.parent.name)
            PrintParam(param=angles_rep,
                       shape=self.parent.shape,
                       title=title_ang,
                       heading=heading,
                       verbose=verbose,
                       draw=draw)

        # Return
        return R, ang