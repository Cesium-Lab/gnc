"""
General mathematical, physical, time, and unit-conversion constants.

References
----------
- David A. Vallado, Fundamentals of Astrodynamics and Applications,
  4th Edition, p. 132.

All values are in SI units unless otherwise indicated.
"""

import numpy as np

# ==================================================
#                  Angles
# ==================================================

DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 1 / DEG_TO_RAD

DEG_TO_ARCSEC = 3600.0
ARCSEC_TO_DEG = 1 / DEG_TO_ARCSEC

ARCSEC_TO_RAD = ARCSEC_TO_DEG * DEG_TO_RAD
RAD_TO_ARCSEC = 1 / ARCSEC_TO_RAD

# ==================================================
#                    Time
# ==================================================

# Seconds, minutes, hours, and days

MIN_TO_SEC = 60.0
SEC_TO_MIN = 1 / MIN_TO_SEC

HOUR_TO_SEC = 3600.0
SEC_TO_HOUR = 1 / HOUR_TO_SEC

DAY_TO_SEC = 86400.0
SEC_TO_DAY = 1 / DAY_TO_SEC

# Sidereal time

SIDEREAL_DAY_TO_SEC = 86164.090517
"""Sidereal day [s]. Source: Vallado 4e, p. 132."""

SEC_TO_SIDEREAL_DAY = 1 / SIDEREAL_DAY_TO_SEC
"""Seconds to sidereal days [sidereal day/s]"""


# Years

JULIAN_YEAR_TO_DAY = 365.25

YEAR_TO_DAY = 365.2421897
"""Tropical year [mean solar days]. Source: Vallado 4e, p. 132."""

DAY_TO_YEAR = 1 / YEAR_TO_DAY
"""Mean solar days to tropical years [year/day]"""

SIDEREAL_YEAR_TO_DAY = 365.256363
"""Sidereal year [mean solar days]. Source: Vallado 4e, p. 132."""

DAY_TO_SIDEREAL_YEAR = 1 / SIDEREAL_YEAR_TO_DAY
"""Mean solar days to sidereal years [year/day]"""

ANOMALISTIC_YEAR_TO_DAY = 365.259636
"""Anomalistic year [mean solar days]. Source: Vallado 4e, p. 132."""

DAY_TO_ANOMALISTIC_YEAR = 1 / ANOMALISTIC_YEAR_TO_DAY
"""Mean solar days to anomalistic years [year/day]"""


# Julian dates

JD_CENTURY_TO_DAY = 36525.0
"""Julian century [days]"""

J2000_JD = 2451545.0
"""J2000 epoch [Julian date, TT]"""


# Time systems

UTC_TO_GPS = 18
"""GPS minus UTC [s]. Valid for the current leap-second regime, not historical dates."""

GPS_TO_UTC = -UTC_TO_GPS
"""UTC minus GPS [s]. Valid for the current leap-second regime, not historical dates."""


# ==================================================
#                   Distance
# ==================================================

KM_TO_M = 1000.0
"""Kilometers to meters [m/km]"""

M_TO_KM = 1 / KM_TO_M
"""Meters to kilometers [km/m]"""

AU_TO_KM = 149597870.7
"""Astronomical unit [km/AU]"""

AU_TO_M = AU_TO_KM * KM_TO_M
"""Astronomical unit [m/AU]"""

FT_TO_M = 0.3048
"""Feet to meters [m/ft], exact. Source: Vallado 4e, p. 132."""

M_TO_FT = 1 / FT_TO_M
"""Meters to feet [ft/m]"""

MILE_TO_M = 1609.344
"""Statute miles to meters [m/mile], exact. Source: Vallado 4e, p. 132."""

M_TO_MILE = 1 / MILE_TO_M
"""Meters to statute miles [mile/m]"""

NM_TO_M = 1852.0
"""Nautical miles to meters [m/nm], exact. Source: Vallado 4e, p. 132."""

M_TO_NM = 1 / NM_TO_M
"""Meters to nautical miles [nm/m]"""


# ==================================================
#                     Force
# ==================================================

LBF_TO_N = 4.44822
"""Pound-force to newtons [N/lbf]. Source: Original; undocumented."""

N_TO_LBF = 1 / LBF_TO_N
"""Newtons to pound-force [lbf/N]"""


# ==================================================
#                     Mass
# ==================================================

KG_TO_LBM = 2.20462
"""Kilograms to pounds-mass [lbm/kg]. Source: Original; undocumented."""

LBM_TO_KG = 1 / KG_TO_LBM
"""Pounds-mass to kilograms [kg/lbm]"""


# ==================================================
#              Physical Constants
# ==================================================

C_LIGHT = 2.99792458e8
"""Speed of light in vacuum [m/s]. Source: Vallado 4e, p. 132."""

H_PLANCK = 6.6260755e-34
"""Planck constant [J·s]. Source: Vallado 4e, p. 132."""

K_BOLTZMANN = 1.380658e-23
"""Boltzmann constant [J/K]. Source: Vallado 4e, p. 132."""
