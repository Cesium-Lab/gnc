from datetime import datetime, timedelta

import numpy as np

from ..constants import DAY_TO_SEC, J2000_JD, JD_CENTURY_TO_DAY, SEC_TO_DAY

LMonth = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
"""List of month lengths as discussed in Vallado 3.6.4"""

################################################################################
#               Julian Date
################################################################################


def jd_to_mjd(jd: float):
    """Convert a Julian date to a Modified Julian Date.

    Parameters
    ----------
    jd : float
        Julian date (any time standard).

    Returns
    -------
    mjd : float
        Modified Julian Date, in the same time standard as `jd`.

    References
    ----------
    Vallado 4e, p. 183.
    """
    return jd - 2400000.5


def mjd_to_jd(mjd: float):
    """Convert a Modified Julian Date to a Julian date.

    Parameters
    ----------
    mjd : float
        Modified Julian Date (any time standard).

    Returns
    -------
    jd : float
        Julian date, in the same time standard as `mjd`.

    References
    ----------
    Vallado 4e, p. 183.
    """
    return mjd + 2400000.5


def jd_to_julian_centuries(jd: float):
    """Convert a Julian date to Julian centuries since J2000.

    Parameters
    ----------
    jd : float
        Julian date in any time standard (e.g. TT, TAI, UT1).

    Returns
    -------
    float
        Julian centuries since J2000, in the same time standard as `jd`.

    References
    ----------
    Vallado 4e, Equation 3-42, p. 184.
    """
    return (jd - J2000_JD) / JD_CENTURY_TO_DAY


def julian_centuries_to_jd(T: float):
    """Convert Julian centuries since J2000 to a Julian date.

    Parameters
    ----------
    T : float
        Julian centuries since J2000 in any time standard (e.g. TT, TAI, UT1).

    Returns
    -------
    jd : float
        Julian date, in the same time standard as `julian_centuries`.

    References
    ----------
    Vallado 4e, Equation 3-42, p. 184.
    """
    return (T * JD_CENTURY_TO_DAY) + J2000_JD


def greg_to_jd(yr: int, mo: int, d: int, h: int, min: int, s: float, meeus=True):
    """Convert a Gregorian calendar date and time to a Julian date.

    Parameters
    ----------
    yr : int
        Year (e.g. 2026).
    mo : int
        Month, 1-12.
    d : int
        Day of the month.
    h : int
        Hour, 0-23.
    min : int
        Minute, 0-59.
    s : float
        Second.
    meeus : bool, optional
        If True, use Meeus's method, which is valid over a wider range of dates.
        Otherwise use Vallado's method (valid 1900-2100). Default is True.

    Returns
    -------
    jd : float
        Julian date, in the same time standard as the input date.

    References
    ----------
    Vallado 4e, Algorithm 14 (Julian Date), p. 183.
    """

    if meeus:
        if mo in [1, 2]:
            yr -= 1
            mo += 12

        A = yr // 100
        B = 2 - A + A // 4
        C = ((s / 60 + min) / 60 + h) / 24
        jd = (
            np.floor(365.25 * (yr + 4716))
            + np.floor(30.6001 * (mo + 1))
            + d
            + B
            - 1524.5
            + C
        )
    else:
        jd = (
            367 * yr
            - np.floor(7 * (yr + (mo + 9) // 12) / 4)
            + np.floor(275 * mo / 9)
            + d
            + 1721013.5
            + ((s / 60 + min) / 60 + h) / 24
        )
    return jd


def jd_to_greg(jd: float):
    """Convert a Julian date to a Gregorian calendar date and time.

    Parameters
    ----------
    jd : float
        Julian date.

    Returns
    -------
    year : int
    month : int
    day : int
    hour : int
    minute : int
    second : float

    References
    ----------
    Vallado 4e, Algorithm 22 (JDToGregorianDate), p. 202.
    """
    month_lengths = list(LMonth)

    T_1900 = (jd - 2415019.5) / 365.25
    year = 1900 + np.trunc(T_1900)
    leap_yrs = np.trunc((year - 1900 - 1) * 0.25)
    days = (jd - 2415019.5) - ((year - 1900) * (365.0) + leap_yrs)
    if days < 1:
        year -= 1
        leap_yrs = np.trunc((year - 1900 - 1) * 0.25)
        days = (jd - 2415019.5) - ((year - 1900) * (365.0) + leap_yrs)
    if year % 4 == 0:
        month_lengths[1] = 29
    day_of_yr = np.trunc(days)

    day_sum = 0
    month = 0
    for month_length in month_lengths:
        if day_sum + month_length > day_of_yr:
            break
        day_sum += month_length
        month += 1

    T = (days - day_of_yr) * 24
    hr = np.trunc(T)
    min = np.trunc((T - hr) * 60)
    sec = (T - hr - min / 60) * 3600

    return (
        int(year),
        int(month + 1),
        int(day_of_yr - day_sum),
        int(hr),
        int(min),
        float(sec),
    )


################################################################################
#               Degrees
################################################################################


def dms_to_rad(deg: int, min: int, sec: float):
    """Convert an angle in degrees, arcminutes, and arcseconds to radians.

    Parameters
    ----------
    deg : int
        Whole degrees.
    min : int
        Arcminutes.
    sec : float
        Arcseconds.

    Returns
    -------
    angle_rad : float
        Angle [rad].

    References
    ----------
    Vallado 4e, Algorithm 17 (DMSToRad), p. 197.
    """
    return (deg + min / 60 + sec / 3600) * np.pi / 180.0


def rad_to_dms(angle_rad: float):
    """Convert an angle in radians to degrees, arcminutes, and arcseconds.

    Parameters
    ----------
    angle_rad : float
        Angle [rad].

    Returns
    -------
    deg : int
        Whole degrees.
    min : int
        Whole arcminutes.
    sec : float
        Arcseconds.

    References
    ----------
    Vallado 4e, Algorithm 18 (RadToDMS), p. 197.
    """
    temp = angle_rad * 180.0 / np.pi
    deg = np.trunc(temp)
    min = np.trunc((temp - deg) * 60)
    sec = (temp - deg - min / 60) * 3600
    return (deg, min, sec)


################################################################################
#               Standards
################################################################################


def tdb_minus_tt_s(jd: float):
    """TDB - TT offset, main periodic term only (~±1.657 ms).

    Parameters
    ----------
    jd : float
        Julian date in TT or TDB (the offset is insensitive to which).

    Returns
    -------
    float
        TDB - TT [s]. Use like `sec_tdb = tdb_minus_tt_s() + sec_tt`

    References
    ----------
    Vallado 4e, Equation 3-53.
    """
    t = jd_to_julian_centuries(jd)
    return 0.001657 * np.sin(628.3076 * t + 6.2401)


def jd_tt_to_tdb(jd_tt: float):
    """Convert a TT Julian date to TDB.

    Parameters
    ----------
    jd_tt : float
        Julian date in TT.

    Returns
    -------
    jd_tdb : float
        Julian date in TDB.

    References
    ----------
    Vallado 4e, Equation 3-53.
    """
    return jd_tt + tdb_minus_tt_s(jd_tt) * SEC_TO_DAY


def jd_tdb_to_tt(jd_tdb: float):
    """Convert a TDB Julian date to TT.

    Evaluating the offset at TDB instead of TT changes it by < 1e-12 s.

    Parameters
    ----------
    jd_tdb : float
        Julian date in TDB.

    Returns
    -------
    jd_tt : float
        Julian date in TT.

    References
    ----------
    Vallado 4e, Equation 3-53.
    """
    return jd_tdb - tdb_minus_tt_s(jd_tdb) * SEC_TO_DAY


def jd_to_sec_j2000(jd: float):
    """Convert a Julian date to seconds since the J2000 epoch.

    The time standard is unchanged, so a TDB Julian date gives TDB seconds
    past J2000 (the epoch convention used by `RpoTimeline`).

    Parameters
    ----------
    jd : float
        Julian date (any time standard).

    Returns
    -------
    sec_j2000 : float
        Seconds since J2000 (JD 2451545.0) in the same time standard [s].
    """
    return (jd - J2000_JD) * DAY_TO_SEC


def sec_j2000_to_jd(sec_j2000: float):
    """Convert seconds since the J2000 epoch to a Julian date.

    Inverse of `jd_to_sec_j2000`; the time standard is unchanged.

    Parameters
    ----------
    sec_j2000 : float
        Seconds since J2000 (JD 2451545.0) [s].

    Returns
    -------
    jd : float
        Julian date in the same time standard.
    """
    return J2000_JD + sec_j2000 * SEC_TO_DAY


TT_MINUS_TAI_S = 32.184
"""TT - TAI [s], exact by definition."""

TAI_MINUS_UTC_S = 37.0
"""TAI - UTC (leap seconds, Delta AT) [s], in effect since 2017-01-01. No later leap
second has been announced, so this value is also used for future epochs."""


def jd_tt_to_utc(jd_tt: float, tai_minus_utc_s: float = TAI_MINUS_UTC_S):
    """Convert a TT Julian date to UTC.

    UTC = TT - (TT - TAI) - (TAI - UTC). For a TDB epoch, pass `jd_tdb_to_tt(jd)`
    first (the difference is under 2 ms).

    Parameters
    ----------
    jd_tt : float
        Julian date in TT.
    tai_minus_utc_s : float, optional
        Accumulated leap seconds, Delta AT [s]. Default `TAI_MINUS_UTC_S` (37 s).

    Returns
    -------
    jd_utc : float
        Julian date in UTC.
    """
    return jd_tt - (TT_MINUS_TAI_S + tai_minus_utc_s) * SEC_TO_DAY


def jd_to_greg_str(jd: float, decimals: int = 0, suffix: str = "") -> str:
    """Format a Julian date as 'YYYY-MM-DD HH:MM:SS[.fff]' in its own time standard.

    Rounds to `decimals` places of seconds with carries into minutes, hours and the
    date (59.9996 s never prints as 60). Proleptic Gregorian calendar; ignores
    leap-second boundaries.

    Parameters
    ----------
    jd : float
        Julian date (any time standard).
    decimals : int, optional
        Digits after the seconds' decimal point, 0-6. Default 0.
    suffix : str, optional
        Appended after a space, e.g. "UTC" or "TDB".

    Returns
    -------
    str
        Formatted date, e.g. '2035-12-25 00:00:00 UTC'.
    """
    if not 0 <= decimals <= 6:
        raise ValueError("decimals must be between 0 and 6")
    # Integer count of rounded units since J2000 keeps the carry exact
    units = round((jd - J2000_JD) * DAY_TO_SEC * 10**decimals)
    stamp = datetime(2000, 1, 1, 12) + timedelta(
        microseconds=units * 10 ** (6 - decimals)
    )
    text = stamp.strftime("%Y-%m-%d %H:%M:%S")
    if decimals:
        text += f".{stamp.microsecond // 10 ** (6 - decimals):0{decimals}d}"
    return f"{text} {suffix}" if suffix else text
