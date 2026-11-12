import numpy as np
import pytest

import cesium.math.time as T


def test_vallado_greg_to_jd_examples():
    """Vallado 4e pg. 183"""

    # Non-Meeus (Algorithm 14)
    assert T.greg_to_jd(1980, 1, 6, 0, 0, 0, False) == pytest.approx(2444244.5)
    assert T.greg_to_jd(2000, 1, 1, 12, 0, 0, False) == pytest.approx(2451545.0)
    assert T.greg_to_jd(1900, 1, 1, 12, 0, 0, False) == pytest.approx(2415021.0)
    assert T.greg_to_jd(1899, 12, 31, 19, 31, 28.128, False) == pytest.approx(
        2415020.31352
    )
    assert T.greg_to_jd(1949, 12, 31, 22, 9, 46.862, False) == pytest.approx(
        2433282.42345905
    )

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
    yr, mo, da, hr, mi, se = T.jd_to_greg(jd)

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


def test_vallado_many():
    # Vallado 4e Example 3-7 p. 196
    jd_tt = 2453140.197
    T_tt = 0.043674121031
    jd_tdb = 2453140.197270667

    assert T.jd_to_julian_centuries(jd_tt) == pytest.approx(T_tt)
    assert T.jd_tt_to_tdb(jd_tt) == pytest.approx(jd_tdb)
    assert T.jd_to_julian_centuries(jd_tdb) == pytest.approx(
        T_tt
    )  # Shows that tdb barely changes from tt


def test_dms():
    assert T.dms_to_rad(-35, -15, -53.63) == pytest.approx(-0.6154886)
    assert np.allclose(T.rad_to_dms(-0.6154886), (-35, -15, -53.6299), atol=1e-2)


def test_tdb_minus_tt_bounds():
    """TDB - TT stays within the 1.657 ms amplitude over a year."""
    jds = np.linspace(2451545.0, 2451545.0 + 365.25, 1000)
    offsets = np.array([T.tdb_minus_tt_s(jd) for jd in jds])
    assert np.abs(offsets).max() == pytest.approx(0.001657, rel=1e-3)


def test_tt_tdb_round_trip():
    """TT -> TDB -> TT recovers the input to JD float resolution."""
    jd_tt = T.greg_to_jd(2035, 12, 25, 0, 0, 0)
    jd_back = T.jd_tdb_to_tt(T.jd_tt_to_tdb(jd_tt))
    assert (jd_back - jd_tt) * 86400 == pytest.approx(0.0, abs=1e-6)


def test_sec_j2000_round_trip():
    assert T.jd_to_sec_j2000(2451545.0) == 0.0
    assert T.jd_to_sec_j2000(2451546.0) == pytest.approx(86400.0)
    assert T.sec_j2000_to_jd(T.jd_to_sec_j2000(2461000.25)) == pytest.approx(2461000.25, abs=1e-9)
