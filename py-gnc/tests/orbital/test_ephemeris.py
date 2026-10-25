import numpy as np
import pytest

from cesium.constants import ARCSEC_TO_DEG, KM_TO_M, RAD_TO_DEG
from cesium.math.time import greg_to_jd
from cesium.orbital.ephemeris import planet_elements_and_rv


def test_curtis_example_8_7():
    """Curtis, Example 8-7, p. 389

    JPL is more updated, so checking within small bounds"""

    # 27 August 2003, 12:00
    jd = greg_to_jd(2003, 8, 27, 12, 0, 0)
    assert jd == 2452879.0

    # coes: [a, e, i, raan, aop, ta]
    coes_earth, r_earth_HCI, _ = planet_elements_and_rv(3, jd)
    coes_mars, r_mars_HCI, _ = planet_elements_and_rv(4, jd)

    # 0.2% error
    d = np.abs(r_earth_HCI - r_mars_HCI)
    assert np.linalg.norm(d) == pytest.approx(55.79e6 * KM_TO_M, rel=2e-4)

    # important coes

    # semi-major axis (0.1%)
    assert coes_earth[0] == pytest.approx(1.4960e8 * KM_TO_M, 1e-4)
    assert coes_mars[0] == pytest.approx(2.2794e8 * KM_TO_M, 1e-4)

    # eccentricity (1%)
    assert coes_earth[1] == pytest.approx(0.016709, 1e-3)
    assert coes_mars[1] == pytest.approx(0.093417, 1e-3)

    # inclination (4 arcseconds)
    assert coes_earth[2] * RAD_TO_DEG == pytest.approx(
        0.00042622, abs=4 * ARCSEC_TO_DEG
    )
    assert coes_mars[2] * RAD_TO_DEG == pytest.approx(1.8504, abs=4 * ARCSEC_TO_DEG)

    # Longitude of periapsis (100 arcseconds)
    # Earth is in the ecliptic, so it's hard to do this so using true anomaly is easier
    assert (coes_earth[3] + coes_earth[4]) * RAD_TO_DEG == pytest.approx(
        102.96, abs=100 * ARCSEC_TO_DEG
    )

    # raan (100 arcseconds)
    # Mars argument of periapsis is easier because it's not in the ecliptic
    assert coes_mars[3] * RAD_TO_DEG == pytest.approx(49.568, abs=100 * ARCSEC_TO_DEG)

    # True anomaly (50-100 arcseconds)
    assert coes_earth[5] * RAD_TO_DEG == pytest.approx(230.81, abs=50 * ARCSEC_TO_DEG)
    assert coes_mars[5] * RAD_TO_DEG == pytest.approx(358.13, abs=100 * ARCSEC_TO_DEG)


# TODO
# def test_curtis_problem_8_7():
#     # Find the maximum Earth-Sun distance during 2005

#     assert aphelion_date == (2005, 7, 4)


# @pytest.mark.slow
# def test_curtis_8_8():
#     # Earth -> Venus, 121-day Lambert transfer
#     ...
#     assert dv_total / 1000 == pytest.approx(6.75, abs=0.1)


# @pytest.mark.slow
# def test_curtis_8_9():
#     # Earth -> Mars Lambert transfer
#     ...
#     assert dv_total / 1000 == pytest.approx(4.86, abs=0.1)

def test_vallado_example_12_8():
    """Vallado 4e, Example 12-8, p. 960"""
    # Using a Curtis solver with Vallado oh god
    
    # September 8, 1977 09:08:17 UTC (Vallado has this wrong but oh well)
    jd_earth = greg_to_jd(1977, 9, 8, 9, 8, 17)
    _, earth_r_HCI, earth_v_HCI = planet_elements_and_rv(3, jd_earth)
    assert earth_r_HCI[:-1] / KM_TO_M == pytest.approx( [146169549, -36666991], rel=1e-2) # Z axis is special because of being in the ecliptic
    assert earth_v_HCI[:-1] / KM_TO_M == pytest.approx( [6.774093, 28.786741], rel=1e-2) # Z axis is special because of being in the ecliptic
    
    # March 5, 1979 12:05:26 UTC
    jd_jupiter = greg_to_jd(1979, 3, 5, 12, 5, 26)
    _, jupiter_r_HCI, jupiter_v_HCI = planet_elements_and_rv(5, jd_jupiter)
    assert jupiter_r_HCI / KM_TO_M == pytest.approx( [-4.82178605e8, 6.27481695e8, 8.221250e6], rel=1e-2)
    assert jupiter_v_HCI / KM_TO_M == pytest.approx( [-10.524225, -7.363294, .266050], rel=1e-2)

    # November 12, 1980 23:46:30 UTC
    jd_saturn = greg_to_jd(1980, 11, 12, 23, 46, 30)
    _, saturn_r_HCI, saturn_v_HCI = planet_elements_and_rv(6, jd_saturn)
    assert saturn_r_HCI / KM_TO_M == pytest.approx([-1.421327810e9, -5.4598217e7, 5.7422412e7], rel=1e-2)
    assert saturn_v_HCI / KM_TO_M == pytest.approx([-.148850, -9.684595, .175045], rel=1e-1) # Uh.
    
    