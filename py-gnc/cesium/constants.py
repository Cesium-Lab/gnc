import numpy as np

########################################
#               Angle
########################################

DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 1 / DEG_TO_RAD

HOUR_TO_SEC = 3600

DEG_TO_ARCSEC = 3600
ARCSEC_TO_DEG = 1 / 3600

ARCSEC_TO_RAD = ARCSEC_TO_DEG * DEG_TO_RAD
RAD_TO_ARCSEC = 1 / ARCSEC_TO_RAD


########################################
#               Time
########################################

UTC_TO_GPS = 18
"""Add to UTC to make GPS"""
GPS_TO_UTC = -18
"""Add to GPS to make UTC"""

DAY_TO_SEC = 86400
SEC_TO_DAY = 1 / DAY_TO_SEC

MIN_TO_SEC = 60
SEC_TO_MIN = 1.0 / 60

SIDEREAL_DAY_TO_SEC = 86164.090517
SEC_TO_SIDEREAL_DAY = 1 / SIDEREAL_DAY_TO_SEC

YEAR_TO_DAY = 365.2421897
DAY_TO_YEAR = 1 / YEAR_TO_DAY
SIDEREAL_YEAR_TO_DAY = 365.256363
DAY_TO_SIDEREAL_YEAR = 1 / SIDEREAL_YEAR_TO_DAY

JD_CENTURY_TO_DAY = 36525.0
J2000_JD = 2451545.0

########################################
#               Physics Stuff
########################################

# Force
LBF_TO_N = 4.44822
N_TO_LBF = 1 / LBF_TO_N

# Mass
KG_TO_LBM = 2.20462
LBM_TO_KG = 1 / KG_TO_LBM

########################################
#               Distance
########################################

M_TO_KM = 1.0/1000
KM_TO_M = 1000.0

AU_TO_KM = 149597870.7  # km/AU