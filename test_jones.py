import pytest
from VolumeRaytraceLFM.birefringence_implementations import *

def test_polarizer_generators():
    LCP = JonesMatrixGenerators.left_circular_polarizer()
    RCP = JonesMatrixGenerators.right_circular_polarizer()
    product = LCP @ RCP
    zero_matrix = np.array([[0, 0], [0, 0]])
    assert np.all(product == zero_matrix), "Left circular polarizer is not opposite of right circular polarizer"
    # assert False
    retA = np.pi / 4
    retB = np.pi / 2
    UC = JonesMatrixGenerators.universal_compensator(retA, retB)
    assert np.all(RCP == UC), "Universal compensator is not a right circular polarizer in the extinction setting"
    

def test_linear_polarizer():
    theta = np.pi / 4
    LP = JonesMatrixGenerators.linear_polarizer(theta)
    LP0 = np.array([[1, 0], [0, 0]])
    rotator_pos = JonesMatrixGenerators.rotator(theta)
    rotator_neg = JonesMatrixGenerators.rotator(-theta)
    LP_through_rotation = rotator_pos @ LP0 @ rotator_neg
    assert np.all(LP == LP_through_rotation), "Linear polarizer is off"

def test_polscope():
    analyzer = JonesMatrixGenerators.polscope_analyzer()
    LP = JonesMatrixGenerators.linear_polarizer(np.pi / 4)
    QWP = JonesMatrixGenerators.quarter_waveplate(0)
    assert np.all(np.isclose(analyzer, LP @ QWP)), "Polscope analyzer is not a quater waveplate followed by a linear polarizer."
    UC = JonesMatrixGenerators.universal_compensator(np.pi / 2, np.pi)
    left_circ_vector = JonesVectorGenerators.left_circular()
    output_vector = UC @ left_circ_vector
    assert np.isclose(np.dot(output_vector, output_vector), 0), "Universal compensator does not extinguish right circularly polarizerd light in the extinction setting"
    # todo: test the polscope settings



def main():
    test_linear_polarizer()
    test_polscope()

if __name__ == '__main__':
    main()