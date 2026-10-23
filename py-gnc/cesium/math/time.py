import numpy as np

from ..constants import JD_CENTURY_TO_DAY, J2000_JD

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

def julian_centuries_to_jd(julian_centuries: float):
    """Convert Julian centuries since J2000 to a Julian date.

    Parameters
    ----------
    julian_centuries : float
        Julian centuries since J2000 in any time standard (e.g. TT, TAI, UT1).

    Returns
    -------
    jd : float
        Julian date, in the same time standard as `julian_centuries`.

    References
    ----------
    Vallado 4e, Equation 3-42, p. 184.
    """
    return (julian_centuries * JD_CENTURY_TO_DAY) + J2000_JD


def greg_to_jd(
    yr: int, mo: int, d: int, h: int, min: int, s: float, meeus=True
):
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
    Vallado 4e, p. 183.
    """

    if meeus:
        if mo in [1, 2]:
            yr -= 1
            mo += 12

        A = yr // 100
        B = 2 - A + A // 4
        C = ((s / 60 + min) / 60 + h) / 24
        jd = np.floor(365.25 * (yr + 4716)) + np.floor(30.6001 * (mo + 1)) + d + B - 1524.5 + C
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
    Vallado 4e, Algorithm 22, p. 202.
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

    return year, month + 1, day_of_yr - day_sum, hr, min, sec


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
    Vallado 4e, Algorithm 17, p. 197.
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
    Vallado 4e, Algorithm 18, p. 198.
    """
    temp = angle_rad * 180.0 / np.pi
    deg = np.trunc(temp)
    min = np.trunc((temp - deg) * 60)
    sec = (temp - deg - min / 60) * 3600
    return (deg, min, sec)
