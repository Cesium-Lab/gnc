# ==================================================
#              Physical Constants
# ==================================================

G_CONST = 6.67430e-11
"""Universal gravitational constant [m3/(kg·s2)]"""

G0 = 9.80665
"""Standard gravity [m/s2]"""


# ==================================================
#                      Sun
# ==================================================

R_SUN = 6.957e8
"""Sun radius [m] VERY CRUDE"""

MU_SUN = 1.32712440018e20
"""Sun standard gravitational parameter [m3/s2] (Vallado 4e p. 1043 + more from internet)"""


# ==================================================
#                     Earth
# ==================================================

# Mass and gravitational parameter

M_EARTH = 5.9733328e24
"""Earth mass [kg]"""

MU_EARTH = 3.986004418e14
"""Earth standard gravitational parameter [m3/s2]"""

MU_EARTH_KM = 398600.4418
"""Earth standard gravitational parameter [km3/s2]"""


# Geometry

R_EARTH = 6378137.630
"""Earth mean equatorial radius [m] (Vallado 4e p. 132)"""

R_EARTH_POLAR = 6356751
"""Earth polar radius [m] (Vallado 4e p. 132)"""

ECC_EARTH = 0.081819221456
"""Earth's geometric eccentricity (Vallado 4e p. 132)"""


# Rotation and orientation

W_EARTH = 0.000072921158553
"""Angular velocity of Earth in inertial frame [rad/s] (Vallado 4e p. 132)"""

ECLIPTIC_ANGLE = 23.43928
"""Ecliptic angle [deg]"""


# Gravity

GRAV_EARTH_POLE = 9.8321849378
"""Earth's gravitational acceleration at the pole (Vallado 4e p. 141)"""

GRAV_EARTH_EQUA = 9.7803253359
"""Earth's gravitational acceleration at the equator (Vallado 4e p. 141)"""


# Zonal harmonics

J2_EARTH = 0.0010826267

J3_EARTH = -0.0000025327


# ==================================================
#                  Other Planets
# ==================================================

MU_JUPITER = 126.687e15
"""Jupiter standard gravitational parameter [m3/s2]"""

MU_SATURN = 37.931e15
"""Saturn standard gravitational parameter [m3/s2]"""


# ==================================================
#              Additional Reference Values
# ==================================================

# Gravitational parameters [km3/s2]
# MU_EARTH = 398600.4418
# MU_MOON = 4903.0
# MU_VENUS = 324858.630
# MU_PLUTO = 870.9
# MU_CHARON = 101.0

# Radii [km]
# R_MOON = 1738.0
# R_SUN = 695508.0
