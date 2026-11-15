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
    assert T.sec_j2000_to_jd(T.jd_to_sec_j2000(2461000.25)) == pytest.approx(
        2461000.25, abs=1e-9
    )


def test_jd_to_greg_str_formats_and_rounds():
    assert T.jd_to_greg_str(2451545.0) == "2000-01-01 12:00:00"
    assert T.jd_to_greg_str(2464686.5, suffix="UTC") == "2035-12-25 00:00:00 UTC"
    assert (
        T.jd_to_greg_str(2451545.0 + 1.5 / 86400, decimals=3)
        == "2000-01-01 12:00:01.500"
    )
    # 59.9996 s rounds up and carries through minutes, hours, and the date
    just_before_midnight = 2451545.5 - 0.0004 / 86400
    assert T.jd_to_greg_str(just_before_midnight) == "2000-01-02 00:00:00"
    assert (
        T.jd_to_greg_str(just_before_midnight, decimals=4) == "2000-01-01 23:59:59.9996"
    )
    with pytest.raises(ValueError):
        T.jd_to_greg_str(2451545.0, decimals=7)


def test_jd_to_greg_str_matches_jd_to_greg():
    jd = T.greg_to_jd(2035, 6, 14, 22, 4, 48)
    y, mo, d, h, mi, s = T.jd_to_greg(jd)
    assert (
        T.jd_to_greg_str(jd)
        == f"{y:04.0f}-{mo:02.0f}-{d:02.0f} {h:02.0f}:{mi:02.0f}:{round(s):02d}"
    )


def test_jd_tt_to_utc_offset():
    # TT - UTC = 32.184 s + 37 leap seconds = 69.184 s
    jd_tt = 2464686.5
    assert (jd_tt - T.jd_tt_to_utc(jd_tt)) * 86400 == pytest.approx(69.184, abs=1e-4)
    assert (
        jd_tt - T.jd_tt_to_utc(jd_tt, tai_minus_utc_s=36.0)
    ) * 86400 == pytest.approx(68.184, abs=1e-4)


class Test_mjd:
    def test_mjd_epoch_is_1858_nov_17(self) -> None:
        """MJD 0 is 1858-11-17 00:00 by definition (MJD = JD - 2400000.5)."""
        assert T.jd_to_mjd(T.greg_to_jd(1858, 11, 17, 0, 0, 0)) == pytest.approx(0.0)

    def test_j2000(self) -> None:
        """J2000 = JD 2451545.0 = 2000-01-01 12:00, so MJD 51544.5."""
        assert T.jd_to_mjd(2451545.0) == 51544.5
        assert T.mjd_to_jd(51544.5) == 2451545.0

    def test_round_trip(self) -> None:
        jd = 2463000.123
        assert T.mjd_to_jd(T.jd_to_mjd(jd)) == pytest.approx(jd, abs=1e-9)


def test_julian_centuries_to_jd() -> None:
    """One Julian century is 36525 days after J2000; inverse of jd_to_julian_centuries."""
    assert T.julian_centuries_to_jd(0.0) == 2451545.0
    assert T.julian_centuries_to_jd(1.0) == 2451545.0 + 36525.0

    # roundtrip
    jd = 2463000.123
    assert T.julian_centuries_to_jd(T.jd_to_julian_centuries(jd)) == pytest.approx(
        jd, abs=1e-6
    )
