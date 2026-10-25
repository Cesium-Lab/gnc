"""
Physical and astronomical constants.

References
----------
- Howard D. Curtis, Orbital Mechanics for Engineering Students,
    Table A.1: Astronomical Data for the Sun, Planets, and Moon.
- David A. Vallado, Fundamentals of Astrodynamics and Applications,
    4th Edition

All values are in SI units unless otherwise indicated.
"""


# ==================================================
#              Physical Constants
# ==================================================

G_CONST = 6.67430e-11
"""Universal gravitational constant [m3/(kg·s2)]. Source: Undocumented."""

G0 = 9.80665
"""Standard gravity [m/s2]. Source: Undocumented."""

ECLIPTIC_ANGLE = 23.43928
"""Obliquity of the ecliptic [deg]. Source: Undocumented."""


# ==================================================
#                       Sun
# ==================================================

M_SUN = 1.989e30
"""Sun mass [kg]. Source: Curtis, Table A.1."""

R_SUN = 6.957e8
"""Sun radius [m]. Source: Original value; undocumented."""

MU_SUN = 1.32712440018e20
"""Sun standard gravitational parameter [m3/s2]. Source: Vallado 4e, p. 1043."""


# ==================================================
#                     Mercury
# ==================================================

M_MERCURY = 330.2e21
"""Mercury mass [kg]. Source: Curtis, Table A.1."""

R_MERCURY = 2440e3
"""Mercury radius [m]. Source: Curtis, Table A.1."""

A_MERCURY = 57.91e9
"""Mercury orbital semimajor axis [m]. Source: Curtis, Table A.1."""

ECC_MERCURY_ORBIT = 0.2056
"""Mercury orbital eccentricity. Source: Curtis, Table A.1."""

INC_MERCURY = 7.00
"""Mercury orbital inclination to ecliptic [deg]. Source: Curtis, Table A.1."""


# ==================================================
#                      Venus
# ==================================================

M_VENUS = 4.869e24
"""Venus mass [kg]. Source: Curtis, Table A.1."""

R_VENUS = 6052e3
"""Venus radius [m]. Source: Curtis, Table A.1."""

A_VENUS = 108.2e9
"""Venus orbital semimajor axis [m]. Source: Curtis, Table A.1."""

ECC_VENUS_ORBIT = 0.0067
"""Venus orbital eccentricity. Source: Curtis, Table A.1."""

INC_VENUS = 3.39
"""Venus orbital inclination to ecliptic [deg]. Source: Curtis, Table A.1."""


# ==================================================
#                      Earth
# ==================================================

# Mass and gravitational parameter

M_EARTH = 5.9733328e24
"""Earth mass [kg]. Source: Original value; undocumented."""

MU_EARTH = 3.986004418e14
"""Earth standard gravitational parameter [m3/s2]. Source: Undocumented."""

MU_EARTH_KM = 398600.4418
"""Earth standard gravitational parameter [km3/s2]. Source: Undocumented."""


# Geometry

R_EARTH = 6378137.630
"""Earth equatorial radius [m]. Source: Vallado 4e, p. 132."""

R_EARTH_POLAR = 6356751
"""Earth polar radius [m]. Source: Vallado 4e, p. 132."""

ECC_EARTH = 0.081819221456
"""Earth geometric eccentricity. Source: Vallado 4e, p. 132."""


# Rotation and gravity

W_EARTH = 0.000072921158553
"""Earth angular velocity in inertial frame [rad/s]. Source: Vallado 4e, p. 132."""

GRAV_EARTH_POLE = 9.8321849378
"""Gravitational acceleration at Earth's pole [m/s2]. Source: Vallado 4e, p. 141."""

GRAV_EARTH_EQUA = 9.7803253359
"""Gravitational acceleration at Earth's equator [m/s2]. Source: Vallado 4e, p. 141."""


# Zonal harmonics

J2_EARTH = 0.0010826267
"""Earth second zonal harmonic coefficient. Source: Undocumented."""

J3_EARTH = -0.0000025327
"""Earth third zonal harmonic coefficient. Source: Undocumented."""


# Heliocentric orbital elements

A_EARTH = 149.6e9
"""Earth orbital semimajor axis [m]. Source: Curtis, Table A.1."""

ECC_EARTH_ORBIT = 0.0167
"""Earth orbital eccentricity. Source: Curtis, Table A.1."""

INC_EARTH = 0.0
"""Earth orbital inclination to ecliptic [deg]. Source: Curtis, Table A.1."""


# ==================================================
#                       Moon
# ==================================================

M_MOON = 73.48e21
"""Moon mass [kg]. Source: Curtis, Table A.1."""

R_MOON = 1737e3
"""Moon radius [m]. Source: Curtis, Table A.1."""

A_MOON = 384.4e6
"""Moon orbital semimajor axis about Earth [m]. Source: Curtis, Table A.1."""

ECC_MOON_ORBIT = 0.0549
"""Moon orbital eccentricity. Source: Curtis, Table A.1."""

INC_MOON = 5.145
"""Moon orbital inclination to ecliptic [deg]. Source: Curtis, Table A.1."""


# ==================================================
#                       Mars
# ==================================================

M_MARS = 641.9e21
"""Mars mass [kg]. Source: Curtis, Table A.1."""

R_MARS = 3396e3
"""Mars radius [m]. Source: Curtis, Table A.1."""

A_MARS = 227.9e9
"""Mars orbital semimajor axis [m]. Source: Curtis, Table A.1."""

ECC_MARS_ORBIT = 0.0935
"""Mars orbital eccentricity. Source: Curtis, Table A.1."""

INC_MARS = 1.850
"""Mars orbital inclination to ecliptic [deg]. Source: Curtis, Table A.1."""


# ==================================================
#                     Jupiter
# ==================================================

M_JUPITER = 1.899e27
"""Jupiter mass [kg]. Source: Curtis, Table A.1."""

R_JUPITER = 71490e3
"""Jupiter radius [m]. Source: Curtis, Table A.1."""

MU_JUPITER = 126.687e15
"""Jupiter standard gravitational parameter [m3/s2]. Source: Original value; undocumented."""

A_JUPITER = 778.6e9
"""Jupiter orbital semimajor axis [m]. Source: Curtis, Table A.1."""

ECC_JUPITER_ORBIT = 0.0489
"""Jupiter orbital eccentricity. Source: Curtis, Table A.1."""

INC_JUPITER = 1.304
"""Jupiter orbital inclination to ecliptic [deg]. Source: Curtis, Table A.1."""


# ==================================================
#                      Saturn
# ==================================================

M_SATURN = 568.5e24
"""Saturn mass [kg]. Source: Curtis, Table A.1."""

R_SATURN = 60270e3
"""Saturn radius [m]. Source: Curtis, Table A.1."""

MU_SATURN = 37.931e15
"""Saturn standard gravitational parameter [m3/s2]. Source: Original value; undocumented."""

A_SATURN = 1.433e12
"""Saturn orbital semimajor axis [m]. Source: Curtis, Table A.1."""

ECC_SATURN_ORBIT = 0.0565
"""Saturn orbital eccentricity. Source: Curtis, Table A.1."""

INC_SATURN = 2.485
"""Saturn orbital inclination to ecliptic [deg]. Source: Curtis, Table A.1."""


# ==================================================
#                      Uranus
# ==================================================

M_URANUS = 86.83e24
"""Uranus mass [kg]. Source: Curtis, Table A.1."""

R_URANUS = 25560e3
"""Uranus radius [m]. Source: Curtis, Table A.1."""

A_URANUS = 2.872e12
"""Uranus orbital semimajor axis [m]. Source: Curtis, Table A.1."""

ECC_URANUS_ORBIT = 0.0457
"""Uranus orbital eccentricity. Source: Curtis, Table A.1."""

INC_URANUS = 0.772
"""Uranus orbital inclination to ecliptic [deg]. Source: Curtis, Table A.1."""


# ==================================================
#                     Neptune
# ==================================================

M_NEPTUNE = 102.4e24
"""Neptune mass [kg]. Source: Curtis, Table A.1."""

R_NEPTUNE = 24760e3
"""Neptune radius [m]. Source: Curtis, Table A.1."""

A_NEPTUNE = 4.495e12
"""Neptune orbital semimajor axis [m]. Source: Curtis, Table A.1."""

ECC_NEPTUNE_ORBIT = 0.0113
"""Neptune orbital eccentricity. Source: Curtis, Table A.1."""

INC_NEPTUNE = 1.769
"""Neptune orbital inclination to ecliptic [deg]. Source: Curtis, Table A.1."""


# ==================================================
#                       Pluto
# ==================================================

M_PLUTO = 12.5e21
"""Pluto mass [kg]. Source: Curtis, Table A.1."""

R_PLUTO = 1195e3
"""Pluto radius [m]. Source: Curtis, Table A.1."""

A_PLUTO = 5.870e12
"""Pluto orbital semimajor axis [m]. Source: Curtis, Table A.1."""

ECC_PLUTO_ORBIT = 0.2444
"""Pluto orbital eccentricity. Source: Curtis, Table A.1."""

INC_PLUTO = 17.16
"""Pluto orbital inclination to ecliptic [deg]. Source: Curtis, Table A.1."""


# ==================================================
#             Additional Reference Values
# ==================================================

# Original reference values; sources undocumented.
# Gravitational parameters [km3/s2]:
# MU_MOON = 4903.0
# MU_VENUS = 324858.630
# MU_PLUTO = 870.9
# MU_CHARON = 101.0

# Alternative radii [km]:
# R_MOON = 1738.0
# R_SUN = 695508.0
