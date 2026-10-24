import sys
import numpy as np
import pytest


from cesium.orbital.ephemeris import planet_elements_and_rv
from cesium.math.time import greg_to_jd
from cesium.world import MU_EARTH, R_EARTH
from cesium.constants import KM_TO_M, MIN_TO_SEC, DEG_TO_RAD, RAD_TO_DEG, ARCSEC_TO_DEG


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
    assert coes_earth[2]*RAD_TO_DEG == pytest.approx(0.00042622, abs=4*ARCSEC_TO_DEG)
    assert coes_mars[2]*RAD_TO_DEG == pytest.approx(1.8504, abs=4*ARCSEC_TO_DEG)
   
    # Longitude of periapsis (100 arcseconds)
    # Earth is in the ecliptic, so it's hard to do this so using true anomaly is easier
    assert (coes_earth[3] + coes_earth[4])*RAD_TO_DEG == pytest.approx(102.96, abs=100*ARCSEC_TO_DEG)
    
    # raan (100 arcseconds)
    # Mars argument of periapsis is easier because it's not in the ecliptic
    assert coes_mars[3]*RAD_TO_DEG == pytest.approx(49.568, abs=100*ARCSEC_TO_DEG)
    
    # True anomaly (50-100 arcseconds)
    assert coes_earth[5]*RAD_TO_DEG == pytest.approx(230.81, abs=50*ARCSEC_TO_DEG)
    assert coes_mars[5]*RAD_TO_DEG == pytest.approx(358.13, abs=100*ARCSEC_TO_DEG)
    
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