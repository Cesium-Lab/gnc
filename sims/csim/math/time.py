import numpy as np

LMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
"""List of month lengths as discussed in Vallado 3.6.4"""

################################################################################
#               Julian Date
################################################################################


def jd_to_mjd(jd: float):
    """Julian Date to Mean Julian Date (Vallado 4e p. 183)"""
    return jd - 2400000.5


def mjd_to_jd(mjd: float):
    """Julian Date to Mean Julian Date (Vallado 4e p. 183)"""
    return mjd + 2400000.5


def jd_to_julian_centuries(jd: float):
    """Works with any time standard (e.g. TT, TAI, UT1) Vallado 4e 3-42 p. 184"""
    return (jd - 2451545.0) / 36525


def julian_centuries_to_jd(julian_centuries: float):
    """Works with any time standard (e.g. TT, TAI, UT1) Vallado 4e 3-42 p. 184"""
    return (julian_centuries * 36525) + 2451545.0


def greg_to_jd(
    yr: float, mo: float, d: float, h: float, min: float, s: float, meeus=True
):
    """Gregorian date to Julian Date. \\
    Vallado 4e p. 183

    Args:
        yr (float): Year
        mo (float): Month
        d (float): Day
        h (float): Hour
        m (float): Minute
        s (float): Second
        meeus (bool, optional): Whether to use Meeus's method (more accurate). Defaults to True.

    Returns:
        float: Julian Date
    """

    if meeus:
        if mo in [1, 2]:
            yr -= 1
            mo += 12

        B = 2 - int(yr / 100) + int(int(yr / 100) / 4)
        C = ((s / 60 + min) / 60 + h) / 24
        jd = int(365.25 * (yr + 4716)) + int(30.6001 * (mo + 1)) + d + B - 1524.5 + C
    else:
        jd = (
            367 * yr
            - int(7 * (yr + int((mo + 9) / 12)) / 4)
            + int(275 * mo / 9)
            + d
            + 1721013.5
            + ((s / 60 + min) / 60 + h) / 24
        )
    return jd


def jd_to_greg(jd: float):
    """Turns julian date into Gregorian calendar date (Vallado 4e Algorithm 22 p. 202)

    Args:
        jd (float): Julian date

    Returns:
        tuple: (year, month, day, hour, minute, second)
    """
    month_lengths = LMonth.copy()

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

    return (year, month + 1, day_of_yr - day_sum, hr, min, sec)


################################################################################
#               Degrees
################################################################################


def dms_to_rad(deg: float, min: float, sec: float):
    """Just to make it easier (Vallado 4e Algorithm 17 p. 197)

    Args:
        deg: Degrees
        min: Minutes
        sec: Seconds

    Returns:
        float: angle [rad]
    """
    return (deg + min / 60 + sec / 3600) * np.pi / 180.0


def rad_to_dms(angle_rad: float):
    """Just to make it easier (Vallado 4e Algorithm 18 p. 198)

    Args:
        angle_rad (float): Angle [rad]

    Returns:
        tuple: deg, min, sec
    """
    temp = angle_rad * 180.0 / np.pi
    deg = np.trunc(temp)
    min = np.trunc((temp - deg) * 60)
    sec = (temp - deg - min / 60) * 3600
    return (deg, min, sec)
