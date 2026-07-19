import numpy as np

# Angles

DEG2RAD = np.pi / 180.0
RAD2DEG = 1 / DEG2RAD

# Time

MIN2SEC = 60.0
SEC2MIN = 1 / MIN2SEC

HOUR2MIN = 60.0
MIN2HOUR = 1 / HOUR2MIN

SEC2HOUR = SEC2MIN * MIN2HOUR
HOUR2SEC = 1 / SEC2HOUR

DAY2HOUR = 24
HOUR2DAY = 1 / DAY2HOUR

SEC2DAY = SEC2HOUR * HOUR2DAY
DAY2SEC = 1 / SEC2DAY

# Lengths

MILE2FT = 5280
FT2MILE = 1 / MILE2FT

KM2M = 1000.0
M2KM = 1/KM2M

M2FT = 3.28084
FT2M = 1 / M2FT

M2IN = 39.3701
IN2M = 1 / M2IN

# Velocities

MPS2KNOTS = 1.94384
KNOTS2MPS = 1 / MPS2KNOTS

# Mass

KG2LBM = 2.20462
LBM2KG = 1 / KG2LBM


# Forces

LBF2N = 4.44822
N2LBF = 1 / LBF2N


G_METER = 6.6743e-11 # m3/kg s2
G_KM = G_METER * M2KM**3

SUN = {
    'name': 'Sun',
    'mass': 1.989e30,
    'mu': 1.32712e11,
    'radius': 695700.0
}

EARTH = {
    'name': 'Earth',
    'mass': 5.972e24,
    'mu': 5.972e24*G_KM, # km3/s2
    'mu_meter': 5.972e24*G_METER,
    'radius': 6378.0
}

