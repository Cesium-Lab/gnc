import sys
import numpy as np
import pytest
sys.path.append(".")

import csim.math.time as T

def test_vallado_greg_to_jd_examples():
    """Vallado 4e pg. 183"""

    # Non-Meeus (Algorithm 14)
    assert T.greg_to_jd(1980, 1, 6, 0, 0, 0, False) == pytest.approx(2444244.5)
    assert T.greg_to_jd(2000, 1, 1, 12, 0, 0, False) == pytest.approx(2451545.0)
    assert T.greg_to_jd(1900, 1, 1, 12, 0, 0, False) == pytest.approx(2415021.0)
    assert T.greg_to_jd(1899, 12, 31, 19, 31, 28.128, False) == pytest.approx(2415020.31352)
    assert T.greg_to_jd(1949, 12, 31, 22, 9, 46.862, False) == pytest.approx(2433282.42345905)
    
    # Meeus
    assert T.greg_to_jd(1980, 1, 6, 0, 0, 0) == pytest.approx(2444244.5)
    assert T.greg_to_jd(2000, 1, 1, 12, 0, 0) == pytest.approx(2451545.0)
    assert T.greg_to_jd(1900, 1, 1, 12, 0, 0) == pytest.approx(2415021.0)
    assert T.greg_to_jd(1899, 12, 31, 19, 31, 28.128) == pytest.approx(2415020.31352)
    assert T.greg_to_jd(1949, 12, 31, 22, 9, 46.862) == pytest.approx(2433282.42345905)

    # Example 3.4 (p. 184)
    assert T.greg_to_jd(1996, 10, 26, 14, 20, 0) == pytest.approx(2450383.09722222)

def test_vallado_jd_to_greg():
    jd = 2449877.3458762
    yr,mo,da,hr,mi,se = T.jd_to_greg(jd)

    assert yr == 1995
    assert mo == 6
    assert da == 8
    assert hr == 20
    assert mi == 18
    assert se == pytest.approx(3.70368, rel=1e-5)

def test_jd_to_greg_leap_year_does_not_corrupt_later_calls():
    """makes sure that the list of month days used in jd_to_greg is not corrupted by a leap year call"""
    T.jd_to_greg(2451545.0)  # 2000-01-01, a leap year

    jd = 2449877.3458762
    yr, mo, da, hr, mi, sec = T.jd_to_greg(jd)

    assert yr == 1995
    assert mo == 6
    assert da == 8
    assert hr == 20
    assert mi == 18
    assert sec == pytest.approx(3.70368, rel=1e-5)

def test_vallado_jd_to_jc():
    # In example Vallado 4e Example 3-7 p. 196
    assert T.jd_to_julian_centuries(2453140.197) == pytest.approx(0.043674121031)

def test_dms():
    assert T.dms_to_rad(-35, -15, -53.63) == pytest.approx(-0.6154886)
    assert np.allclose(T.rad_to_dms(-0.6154886), (-35, -15, -53.6299), atol=1e-2)