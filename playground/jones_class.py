'''This jones_class.py script is now mostly incorporated into the birefringence_implementations.py script.
'''
import numpy as np

class JonesMatrix():
    '''Create a Jones matrix represented various optical materials'''
    wavelength = 0.550

    def __init__(self, Delta_n=0, optic_axis=[1, 0, 0], ray_dir=[1, 0, 0], thickness=1):
        self.Delta_n = Delta_n
        self.optic_axis = np.array(optic_axis) / np.linalg.norm(optic_axis)
        self.ray_dir = np.array(ray_dir)
        self.ray_dir_basis = self.calc_rayDir(self.ray_dir)
        self.thickness = thickness
    
    def rotator(self, angle):
        '''2D rotation matrix
        Args:
            angle: angle to rotate by counterclockwise [radians]
        Return: Jones matrix'''
        s = np.sin(angle)
        c = np.cos(angle)
        R = np.array([[c, -s], [s, c]])
        return R

    def LR(self, ret, azim):
        '''Linear retarder
        Args:
            ret (float): retardance [radians]
            azim (float): azimuth angle of fast axis [radians]
        Return: Jones matrix    
        '''
        retardor_azim0 = self.LR_azim0(ret)
        R = self.rotator(azim)
        Rinv = self.rotator(-azim)
        return R @ retardor_azim0 @ Rinv

    def LR_azim0(self, ret):
        return np.array([[np.exp(1j * ret / 2), 0], [0, np.exp(-1j * ret / 2)]])

    def LR_azim90(self, ret):
        return np.array([[np.exp(-1j * ret / 2), 0], [0, np.exp(1j * ret / 2)]])

    def QWP(self, azim):
        '''Quarter Waveplate
        Linear retarder with lambda/4 or equiv pi/2 radians
        Commonly used to convert linear polarized light to circularly polarized light'''
        ret = np.pi / 2
        return self.LR(ret, azim)

    def HWP(self, azim):
        '''Half Waveplate
        Linear retarder with lambda/2 or equiv pi radians
        Commonly used to rotate the plane of linear polarization'''
        # Faster method
        s = np.sin(2 * azim)
        c = np.cos(2 * azim)
        JM = np.array([[c, s], [s, -c]])
        # # Alternative method
        # ret = np.pi
        # JM = self.LR(ret, azim)
        return JM

    def LP(self, theta):
        '''Linear Polarizer
        Args:
            theta: angle that light can pass through
        Returns: Jones matrix
        '''
        c = np.cos(theta)
        s = np.sin(theta)
        J00 = c ** 2
        J11 = s ** 2
        J01 = s * c
        J10 = J01
        return np.array([[J00, J01], [J10, J11]])
    
    # static method
    def RCP(self):
        '''Right Circular Polarizer'''
        return 1 / 2 * np.array([[1, -1j], [1j, 1]])

    # static method
    def LCP(self):
        '''Left Circular Polarizer'''
        return 1 / 2 * np.array([[1, 1j], [-1j, 1]])

    def RCR(self, ret):
        '''Right Circular Retarder'''
        return self.rotator(-ret / 2)

    def LCR(self, ret):
        '''Left Circular Retarder'''
        return self.rotator(ret / 2)

    ###########################################################################################
    # Methods necessary for determining the Jones matrix of a birefringent material
    # maybe this section should be a subclass of JonesMatrix

    def calc_retardance(self):
        ret = abs(self.Delta_n) * (1 - np.dot(self.optic_axis, self.ray_dir) ** 2) * 2 * np.pi * self.thickness / JonesMatrix.wavelength
        # print(f"Accumulated retardance from index ellipsoid is {np.around(np.rad2deg(ret), decimals=0)} ~ {int(np.rad2deg(ret)) % 360} degrees.")
        return ret

    def calc_azimuth(self):
        azim = np.arctan2(np.dot(self.optic_axis, self.ray_dir_basis[1]), np.dot(self.optic_axis, self.ray_dir_basis[2]))
        if self.Delta_n == 0:
            azim = 0
        elif self.Delta_n < 0:
            azim = azim + np.pi / 2
        # print(f"Azimuth angle of index ellipsoid is {np.around(np.rad2deg(azim), decimals=0)} degrees.")
        return azim


    def rotation_matrix(self, axis, angle):
        '''Generates the rotation matrix that will rotate a 3D vector
        around "axis" by "angle" counterclockwise.'''
        ax, ay, az = axis[0], axis[1], axis[2]
        s = np.sin(angle)
        c = np.cos(angle)
        u = 1 - c
        R_tuple = ( ( ax*ax*u + c,    ax*ay*u - az*s, ax*az*u + ay*s ),
            ( ay*ax*u + az*s, ay*ay*u + c,    ay*az*u - ax*s ),
            ( az*ax*u - ay*s, az*ay*u + ax*s, az*az*u + c    ) )
        R = np.asarray(R_tuple)
        return R

    def find_orthogonal_vec(self, v1, v2):
        '''v1 and v2 are numpy arrays (3d vectors)
        This function accomodates for a divide by zero error.'''
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
                normal_vec = np.cross(v1, non_par_vec) / np.linalg.norm(np.cross(v1, non_par_vec))
        else:
            normal_vec = np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2))
        return normal_vec
    
    def calc_rayDir(self, ray):
        '''
        Allows to the calculations to be done in ray-space coordinates
        as oppossed to laboratory coordinates
        Parameters:
            ray (np.array): normalized 3D vector giving the direction 
                            of the light ray
        Returns:
            ray (np.array): same as input
            ray_perp1 (np.array): normalized 3D vector
            ray_perp2 (np.array): normalized 3D vector
        '''
        # ray = ray / np.linalg.norm(ray)    # in case ray is not a unit vector <- does not need to be normalized
        theta = np.arccos(np.dot(ray, np.array([1,0,0])))
        # Unit vectors that give the laboratory axes, can be changed
        scope_axis = np.array([1,0,0])
        scope_perp1 = np.array([0,1,0])
        scope_perp2 = np.array([0,0,1])
        theta = np.arccos(np.dot(ray, scope_axis))
        # print(f"Rotating by {np.around(np.rad2deg(theta), decimals=0)} degrees")
        normal_vec = self.find_orthogonal_vec(ray, scope_axis)
        Rinv = self.rotation_matrix(normal_vec, -theta)
        # Extracting basis vectors that are orthogonal to the ray and will be parallel
        # to the laboratory axes that are not the optic axis after a rotation.
        # Note: If scope_perp1 if the y-axis, then ray_perp1 if the 2nd column of Rinv.
        ray_perp1 = np.dot(Rinv, scope_perp1)
        ray_perp2 = np.dot(Rinv, scope_perp2)
        return [ray, ray_perp1, ray_perp2]

    def LR_material(self):
        ret = self.calc_retardance()
        azim = self.calc_azimuth()
        return self.LR(ret, azim)
    ###########################################################################################

class JonesMatrixProperties():
    '''Input is a Jones matrix. Methods give properties about it.'''
    def __init__(self, JM):
        self.JM = JM

    def list_elements(self):
        J00 = round(self.JM[0, 0], 3)
        J10 = round(self.JM[1, 0], 3)
        J01 = round(self.JM[0, 1], 3)
        J11 = round(self.JM[1, 0], 3)
        print("Jones matrix elements are the following:")
        print(f"\t J00: {J00} \t J01: {J01}")
        print(f"\t J10: {J10} \t J11: {J11}")

    def rad2deg(self, rad):
        deg = np.round(np.rad2deg(rad), 1)
        return deg

    def eig(self):
        values, vectors = np.linalg.eig(self.JM)
        return values, vectors

    def retardance(self):
        '''Phase delay introduced between the fast and slow axis'''
        values, vectors = self.eig()
        e1 = values[0]
        e2 = values[1]
        phase_diff = np.angle(e1) - np.angle(e2)
        return np.abs(phase_diff)

    def azimuth(self):
        '''Rotation angle of the fast axis (neg phase)'''
        # This azimuth calculation does not account for all quadrants appropriately.
        values, vectors = self.eig()
        real_vecs = np.real(vectors)
        if np.imag(values[0]) < 0:
            fast_vector = real_vecs[0]
            # Adjust for the case when 135 deg and is calculated as 45 deg
            if fast_vector[0] == fast_vector[1] and real_vecs[1][1] < 0:
                azim = 3 * np.pi / 4
            else:
                azim = np.arctan(fast_vector[0] / fast_vector[1])
        else:
            fast_vector = real_vecs[1]
            azim = np.arctan(fast_vector[0] / fast_vector[1])
        if azim < 0:
            azim = azim + np.pi
        return azim

    def retarder_type(self):
        '''Check if retarder
        - intensity of light and deg of polarization remain unchanged
        - based on eigenvectors, determine if linear, circular, or the general elliptical case'''
        pass
    
    def characteristics(self):
        # print the most unique properties, such as the eigenpolarizations
        pass

class JonesVector():

    def horizonal(self):
        return np.array([1, 0])

    def vertical(self):
        return np.array([0, 1])

    def linear(self, angle):
        '''Angle is w.r.t. to horizonal plane'''
        return np.array([np.cos(angle), np.sin(angle)])

    def circular_right(self):
        '''Angle is w.r.t. to horizonal plane'''
        return np.array([1, -1j])

    def circular_left(self):
        '''Angle is w.r.t. to horizonal plane'''
        return np.array([1, 1j])

    def elliptical(self):
        pass

class JonesVectorProperties():

    def phase(self, J):
        """Phase of the light"""
        gamma = np.angle(J[..., 1]) - np.angle(J[..., 0])
        return gamma

    def azimuth(self, J):
        '''Rotation angle'''
        Ex0, Ey0 = np.abs(J[0])
        delta = self.phase(J)
        numerator = 2 * Ex0 * Ey0 * np.cos(delta)
        denom = Ex0 ** 2 - Ey0 ** 2
        azim = 0.5 * np.arctan2(numerator, denom)
        return azim

    
    def intensity(self):
        """Property which returns the intensity of the Jones vector
        :return: The intensity of the Jones vector
        :rtype: float
        """
        return np.norm(self.J)

    def Ex(self):
        """Property which returns the x component of the electric field"""
        return self.J[0]

    def Ey(self):
        """Property which returns the y component of the electric field"""
        return self.J[1]

    def Stokes(self):
        """Property which returns the Stokes parameter representation of the Polarization
        Returns (tuple): Stokes parameters
        """
        S0 = self.intensity
        S1 = np.abs(self.Ex) ** 2 - np.abs(self.Ey) ** 2
        S2 = 2 * np.real(self.Ex * np.conjugate(self.Ey))
        S3 = -2 * np.imag(self.Ex * np.conjugate(self.Ey))
        return S0, S1, S2, S3


def main():
    JM = JonesMatrix(Delta_n=1, optic_axis=[1, 2, 5], ray_dir=[1, 0, 0], thickness=2)
    JM = JonesMatrix(Delta_n=1, optic_axis=[1, 1, 1], ray_dir=[1, 0, 0], thickness=2)
    JM.LR(np.pi, 0)
    print(JM.LR_material())
    JM.QWP(0)

    print("Azimuth calculations from input material properties")
    print(np.rad2deg(JM.calc_azimuth()))
    my_JM = JM.LR_material()
    JM_prop = JonesMatrixProperties(my_JM)
    print("Azimuth calculations from input Jones matrix")
    JM_prop.azimuth()


if __name__ == '__main__':
    main()