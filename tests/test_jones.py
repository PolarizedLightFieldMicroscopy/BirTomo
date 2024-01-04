'''Test that Jones matrix conventions are consistent.'''
import numpy as np
from VolumeRaytraceLFM.jones_calculus import (
    JonesMatrixGenerators,
    JonesVectorGenerators
)


def test_polarizer_generators():
    '''Tests if right circular polarizer is opposite of left'''
    left = JonesMatrixGenerators.left_circular_polarizer()
    right = JonesMatrixGenerators.right_circular_polarizer()
    product = left @ right
    zero_matrix = np.array([[0, 0], [0, 0]])
    assert np.all(product == zero_matrix), "Left circular polarizer is not opposite of \
                                            right circular polarizer"


def test_linear_polarizer():
    '''Tests if a linear polarizer can be derived from the product with two rotators'''
    theta = np.pi / 4
    linear = JonesMatrixGenerators.linear_polarizer(theta)
    linear0 = np.array([[1, 0], [0, 0]])
    rotator_pos = JonesMatrixGenerators.rotator(theta)
    rotator_neg = JonesMatrixGenerators.rotator(-theta)
    linear_through_rotation = rotator_pos @ linear0 @ rotator_neg
    assert np.all(linear == linear_through_rotation), "Linear polarizer is off"


def test_polscope():
    '''Tests definitions of LC-PolScope polarizers'''
    analyzer = JonesMatrixGenerators.polscope_analyzer()
    linear = JonesMatrixGenerators.linear_polarizer(np.pi / 4)
    quarter = JonesMatrixGenerators.quarter_waveplate(0)
    assert np.all(np.isclose(analyzer, linear @ quarter)), \
        "Polscope analyzer is not a quater waveplate followed by a linear polarizer."
    universal = JonesMatrixGenerators.universal_compensator(np.pi / 2, np.pi)
    left_circ_vector = JonesVectorGenerators.left_circular()
    output_vector = universal @ left_circ_vector
    assert np.isclose(np.dot(output_vector, output_vector), 0), \
        "Universal compensator does not extinguish right circularly polarizerd light in the \
        extinction setting"
    # Note: Universal compensator is not a right circular polarizer in the extinction setting
    # TODO: test the polscope settings


def main():
    '''Place for debugging test functions'''
    test_polarizer_generators()
    test_linear_polarizer()
    test_polscope()


if __name__ == '__main__':
    main()
