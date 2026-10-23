import sys
import numpy as np

sys.path.append(".")
import csim.math.CIP.parse as P
from csim.constants import RAD_TO_ARCSEC, RAD_TO_DEG


def test_fundamental_arguments_no_time():

    expected = np.array(
        [
            485868.249036,
            1287104.79305,
            335779.526232,
            1072260.70369,
            450160.398036,
            4.402608842 * RAD_TO_ARCSEC,
            3.176146697 * RAD_TO_ARCSEC,
            1.753470314 * RAD_TO_ARCSEC,
            6.203480913 * RAD_TO_ARCSEC,
            0.599546497 * RAD_TO_ARCSEC,
            0.874016757 * RAD_TO_ARCSEC,
            5.481293872 * RAD_TO_ARCSEC,
            5.311886287 * RAD_TO_ARCSEC,
            0,
        ]
    )
    result = P.fundamental_arguments(0)

    assert np.allclose(result * RAD_TO_ARCSEC, expected)


def test_fundamental_arguments_example():
    """From Vallado 4e Example 3.14 p. 220"""

    # Book values given in degrees
    expected_deg = np.array(
        [
            314.9122873,
            91.9393769,
            169.0970043,
            196.7516428,
            42.6046467,
            143.319167,
            156.221635,
            194.890465,
            91.262347,
            163.710186,
            102.168400,
            332.317825,
            313.661341,
            0.059545,
        ]
    )
    t_tt = 0.0426236319
    result = P.fundamental_arguments(t_tt) * RAD_TO_DEG

    assert np.allclose(result, expected_deg)


def test_import():
    coeffs_uas, the_data = P.import_table("tab5.2a.txt")
    coeffs_as = coeffs_uas / 1e6

    assert len(the_data) == 5
    assert np.array_equal([len(x) for x in the_data], [1306, 253, 36, 4, 1])
    assert np.array_equal(
        coeffs_as,
        [-0.016617, 2004.191898, -0.4297829, -0.19861834, 0.000007578, 0.0000059285],
    )
