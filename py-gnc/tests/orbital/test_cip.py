import numpy as np
import pytest

import cesium.orbital.CIP.parse as P
from cesium.constants import RAD_TO_ARCSEC, RAD_TO_DEG


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


@pytest.mark.parametrize(
    "filename, terms, coeffs_uas",
    [
        # IERS Conventions (2010) Table 5.2b: CIP Y
        (
            "tab5.2b.txt",
            [962, 277, 30, 5, 1],
            [-6951, -25896, -22407274.7, 1900.59, 1112.526, 0.1358],
        ),
        # Table 5.2d: s + XY/2
        (
            "tab5.2d.txt",
            [33, 3, 25, 4, 1],
            [94.0, 3808.65, -122.68, -72574.11, 27.98, 15.62],
        ),
    ],
)
def test_import_y_and_s_tables(
    filename: str, terms: list[int], coeffs_uas: list[float]
):
    coeffs, the_data = P.import_table(filename)
    assert [len(x) for x in the_data] == terms
    assert coeffs == pytest.approx(coeffs_uas)
    assert all(
        rows.shape[1] == 16 for rows in the_data
    )  # 2 amplitudes + 14 multipliers


@pytest.mark.parametrize("text", ["", "Not an IERS table\n"])
def test_import_rejects_non_cip_file(tmp_path: str, text: list[str]):
    path = tmp_path / "bad.txt"
    path.write_text(text)
    with pytest.raises(ImportError):
        P.import_table(str(path))  # absolute path overrides the CIP directory


class Test_approx_5th_deg_spline:
    def test_hand_calc(self):
        """1 + 2(2) + 3(4) + 4(8) + 5(16) + 6(32) = 1 + 4 + 12 + 32 + 80 + 192 = 321."""
        assert P.approx_5th_deg_spline(2.0, [1, 2, 3, 4, 5, 6]) == 321

    def test_matches_numpy_polyval_on_x_table(self):
        coeffs, _ = P.import_table("tab5.2a.txt")
        t_tt = 0.0426236319  # Vallado Example 3-14 epoch
        expected = np.polynomial.polynomial.polyval(t_tt, coeffs)
        assert P.approx_5th_deg_spline(t_tt, coeffs) == pytest.approx(
            expected, rel=1e-14
        )


class Test_get_summation:
    def test_empty_rows_sum_to_zero(self):
        assert P.get_summation(np.empty((0, 16)), 0.3) == 0.0

    @pytest.mark.parametrize("shape", [(3, 15), (16,)])
    def test_rejects_wrong_shape(self, shape):
        with pytest.raises(ImportError):
            P.get_summation(np.zeros(shape), 0.0)

    def test_zero_multipliers_give_cosine_amplitude(self):
        """All multipliers 0 -> phase 0 -> S sin(0) + C cos(0) = C."""
        rows = np.zeros((1, 16))
        rows[0, :2] = [2.0, 3.0]
        assert P.get_summation(rows, 0.7) == pytest.approx(3.0)

    def test_single_term_on_lunar_anomaly(self):
        """Hand calc at J2000: phase = l = 485868.249036 arcsec = 134.963403 deg.

        2 sin(134.963403 deg) + 3 cos(134.963403 deg) = 1.415117 - 2.119965 = -0.704848
        """
        rows = np.zeros((1, 16))
        rows[0, :3] = [2.0, 3.0, 1.0]  # S, C, multiplier on l (Moon mean anomaly)
        assert P.get_summation(rows, 0.0) == pytest.approx(-0.704848, abs=1e-6)

    def test_terms_add_linearly(self):
        _, the_data = P.import_table("tab5.2a.txt")
        rows, t_tt = the_data[1], 0.1
        split = P.get_summation(rows[:100], t_tt) + P.get_summation(rows[100:], t_tt)
        assert P.get_summation(rows, t_tt) == pytest.approx(split, rel=1e-12)
