"""Non-COES transformations"""

import numpy as np

from ..world import R_EARTH, R_EARTH_POLAR, ECC_EARTH
from ..constants import DEG_TO_RAD, SEC_TO_DAY, ARCSEC_TO_RAD
from .time import jd_to_julian_centuries
from .CIP.parse import import_table, get_summation

################################################################################
#               Angle
################################################################################


def wrap_to_pi(x):
  return (x + np.pi) % (2 * np.pi) - np.pi

def dcm_x(theta: float):
    """Passive DCM from frame 1 to frame 2, rotated by theta [rad] (x2 = dcm @ x1)"""
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), np.sin(theta)],
            [0, -np.sin(theta), np.cos(theta)],
        ]
    )


def dcm_y(theta: float):
    """Passive DCM from frame 1 to frame 2, rotated by theta [rad] (x2 = dcm @ x1)"""
    return np.array(
        [
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)],
        ]
    )


def dcm_z(theta: float):
    """Passive DCM from frame 1 to frame 2, rotated by theta [rad] (x2 = dcm @ x1)"""
    return np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    
################################################################################
#               LLA <--> ECEF
################################################################################


def geodetic_LLA_to_ECEF(lat: float, lon: float, alt_m = 0.0, deg = True):
    """Convert WGS84 geodetic lat, lon, and alt to ECEF\\
    Vallado 4e Example 3.1 p. 140

    Parameters
    ----------
    lat, lon : float
        Lat,lon [deg if deg=True, else rad]
    alt_km : float
        Altitude above SPHERICAL surface [km]
    r_body_km : float
        Body radius [km]
    deg : bool
        Whether input angles are in degrees

    Returns
    -------
    r_ECEF : np.ndarray (3,)
        Earth-fixed position [m]
    """
    if deg:
        lat *= DEG_TO_RAD
        lon *= DEG_TO_RAD


    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)

    e2 = ECC_EARTH**2
    C = R_EARTH / np.sqrt(1 - e2 * sin_lat**2)
    S = C * (1 - e2)

    # Site coordinates [m]
    r_horizontal = (C + alt_m) * cos_lat
    r_vertical = (S + alt_m) * sin_lat

    r_ECEF = np.array(
        [r_horizontal * np.cos(lon), r_horizontal * np.sin(lon), r_vertical]
    )

    return r_ECEF

def ECEF_to_geodetic_LLA(r_ECEF: np.ndarray, deg=True) -> tuple[float, float, float]:
    """Vallado 4e Algorithm 13 p. 173 (Borkowski method).

    Parameters
    ----------
    r_ECEF : np.ndarray (3,)
        Earth-centered, Earth-fixed position [m].
    deg : bool
        Whether output angles are in degrees.

    Returns
    -------
    lat, lon : float
        Geodetic latitude and longitude [deg if deg=True, else rad].
    alt_m : float
        Height above WGS84 reference ellipsoid [m].
    """
    x, y, z = r_ECEF

    r_horizontal_sat = np.hypot(x, y)
    a = R_EARTH
    b = R_EARTH_POLAR

    if r_horizontal_sat < 1e-9:
        if abs(z) < 1e-9:
            raise ValueError("LLA undefined at Earth's center.")

        lat_gd = np.copysign(np.pi / 2, z)
        lon_gd = 0.0
        alt_m = abs(z) - b

    else:
        lon_gd = np.arctan2(y, x)

        E = (b*abs(z) - (a**2 - b**2)) / (a*r_horizontal_sat)
        F = (b*abs(z) + (a**2 - b**2)) / (a*r_horizontal_sat)

        P = 4*(E*F + 1) / 3
        Q = 2*(E**2 - F**2)
        D = P**3 + Q**2

        if D > 0:
            v = np.cbrt(np.sqrt(D) - Q) - np.cbrt(np.sqrt(D) + Q)
        else:
            v = 2*np.sqrt(-P)*np.cos(
                np.arccos(Q/(P*np.sqrt(-P))) / 3
            )

        G = (np.sqrt(E**2 + v) + E) / 2
        t = np.sqrt(G**2 + (F - v*G)/(2*G - E)) - G

        lat_gd = np.arctan2(a*(1-t**2), 2*b*t)
        lat_gd = np.copysign(lat_gd, z)

        alt_m = (
            (r_horizontal_sat - a*t)*np.cos(lat_gd)
            + (abs(z) - b)*abs(np.sin(lat_gd))
        )

    if deg:
        lat_gd /= DEG_TO_RAD
        lon_gd /= DEG_TO_RAD

    return lat_gd, lon_gd, alt_m

def geocentric_LLA_to_PCPF(lat: float, lon: float, alt_m = 0.0, r_body_m = R_EARTH, deg = True):
    """Convert latitude, longitude, and altitude to planet-centered, planet-fixed.
    Spherical body.

    Parameters
    ----------
    lat, lon : float
        Lat,lon [deg if deg=True, else rad]
    alt_m : float
        Altitude above SPHERICAL surface [m]
    r_body_m : float
        Body radius [m]
    deg : bool
        Whether input angles are in degrees

    Returns
    -------
    r_PCPF : np.ndarray (3,)
        Body-fixed position [m]
    """
    if deg:
        lat *= DEG_TO_RAD
        lon *= DEG_TO_RAD
        
    rho = r_body_m + alt_m
    
    r_PCPF = np.array([
        rho * np.cos(lat) * np.cos(lon),
        rho * np.cos(lat) * np.sin(lon),
        rho * np.sin(lat)
    ])
    
    return r_PCPF

def PCPF_to_geocentric_LLA(r_PCPF: np.ndarray, r_body_m=R_EARTH, deg=True):
    x, y, z = r_PCPF

    rho = np.linalg.norm(r_PCPF)
    if rho < 1e-9:
        raise ValueError("Coordinates undefined at body center.")

    lat = np.arctan2(z, np.hypot(x, y))
    lon = np.arctan2(y, x)
    alt_m = rho - r_body_m

    if deg:
        lat /= DEG_TO_RAD
        lon /= DEG_TO_RAD

    return lat, lon, alt_m

################################################################################
#               ITRF <--> GCRS
################################################################################

"""
Alright so this is tough.
I didn't have the mental capacity to implement this at Vast (skill issue)
The crux of it is that `r_gcrs = [P(t)][N(t)][R(t)][W(t)] @ r_itrf` where:
- P(t) = Precession Matrix of date t
- N(t) = Nutation Matrix of date t
- R(t) = Sidereal Rotation Matrix of date t
- W(t) = Polar-motion Matrix of date t

Might be faster to do IAU-76/FK5 Reduction though
"""


def approx_5th_deg_spline(t_tt, coeffs):
    return (
        coeffs[0]
        + coeffs[1] * t_tt
        + coeffs[2] * t_tt**2
        + coeffs[3] * t_tt**3
        + coeffs[4] * t_tt**4
        + coeffs[5] * t_tt**5
    )


"""
need EOP data (pg 623 pdf, 604 normal)

Need:
- JD_UT1 from UT1-UTC (pdf)
- xp,yp,s' (pdf)
- X,Y,s (other pdf)
- 


ITRF -> TIRS with [W]
- xp, yp, s'

TIRS -> CIRS with [R]
- JD_UT1 -> ERA
- w_+ for velocity

CIRS -> GCRF with [PN]
- a,X,Y,s with T_TT and all the values
- pdf 241
- pdf 1077? for 
https://aa.usno.navy.mil/downloads/reports/Hiltonetal2006.pdf ?

3-62 -> get ERA
need T_TT and T_TDB to get X,Y,s (JUST T_TT)



a_p_i
A_yc0_i

Maia 2011 tab5.2a/b/d.txt

"""


def W_matrix(xp: float, yp: float, t_tt: float):
    """Polar-motion matrix for ITRF <--> GCRS transformation.
    Computes intermediate TIO locator `s'` using `t_tt` and a very accurate approximation. \\
    Vallado 4e p. 212 \\
    Could also augment with ocean tides and libration, but TODO for another day

    Args:
        xp (float): Polar x coord. of polar motion of CIP in ITRS
        yp (float): Polar y coord. of polar motion of CIP in ITRS
        t_tt (float): Julian Century (Terrestrial time)

    Returns:
        np.ndarray: Polar-motion matrix [W(t)]
    """
    # Coefficient varies by less than 0.0004" over next century
    # Vallado 4e 3-61 p. 212
    s = -0.000047 * t_tt * ARCSEC_TO_RAD

    cs = np.cos(s)
    ss = np.sin(s)

    cx = np.cos(xp)
    sx = np.sin(xp)

    cy = np.cos(yp)
    sy = np.sin(yp)

    return np.array(
        [
            [cx * cs, -cy * ss + sy * sx * cs, -sy * ss - cy * sx * cs],
            [cx * ss, cy * cs + sy * sx * ss, sy * cs - cy * sx * ss],
            [sx, -sy * cx, cy * cx],
        ]
    )


def R_matrix(jd_ut1: float):
    """Sidereal rotation matrix for ITRF <--> GCRS.
    Computes intermediate Earth Rotation Angle using `jd_ut1` \\
    Vallado 4e p. 213

    Args:
        jd_ut1 (float): Julian date (UT1)

    Returns:
        np.ndarray: Sidereal-rotation matrix [R(t)]
    """

    # Vallado 4e 3-62 p. 213
    theta_ERA = 2 * np.pi * (0.779057273264 + 1.00273781191135448 * (jd_ut1 - 2451545))

    C = np.cos(-theta_ERA)
    S = np.sin(-theta_ERA)
    return np.array([[C, S, 0], [-S, C, 0], [0, 0, 1]])


# TODO: fix with dX and dY
def _X_Y_s_a(t_tt: float):
    t = t_tt
    t2 = t * t
    t3 = t * t2

    X_coeffs, X_data = import_table("tab5.2a.txt")
    Y_coeffs, Y_data = import_table("tab5.2b.txt")
    s_coeffs, s_data = import_table("tab5.2d.txt")

    X_u_arcsec = (
        approx_5th_deg_spline(t_tt, X_coeffs)
        + get_summation(X_data[0], t_tt)
        + get_summation(X_data[1], t_tt) * t
        + get_summation(X_data[2], t_tt) * t2
        + get_summation(X_data[3], t_tt) * t3
    )

    Y_u_arcsec = (
        approx_5th_deg_spline(t_tt, Y_coeffs)
        + get_summation(Y_data[0], t_tt)
        + get_summation(Y_data[1], t_tt) * t
        + get_summation(Y_data[2], t_tt) * t2
        + get_summation(Y_data[3], t_tt) * t3
    )

    # Microarcseconds
    X = X_u_arcsec / 1e6 * ARCSEC_TO_RAD
    Y = Y_u_arcsec / 1e6 * ARCSEC_TO_RAD

    # Normal radians
    #
    s = (
        -X * Y / 2
        + (
            approx_5th_deg_spline(t_tt, s_coeffs)
            + get_summation(s_data[0], t_tt)
            + get_summation(s_data[1], t_tt) * t
            + get_summation(s_data[2], t_tt) * t2
            + get_summation(s_data[3], t_tt) * t3
        )
        / 1e6
        * ARCSEC_TO_RAD
    )

    # TODO: IMPLEMENT THE LOOKUPS
    # CIP unit vector X,Y and angle between CIO and GCRS equator s

    # Vallado 4e p. 213 approximation
    a = 1 / 2 + 1 / 8 * (X * X + Y * Y)

    return X, Y, s, a


def PN_matrix(t_tt: float, dX=0.0, dY=0.0):
    """Precession and nutation matrix for ITRF <--> GCRS.
    Computes intermediate `a` with an approximation. \\
    Vallado 4e p. 213

    Args:
        t_tt (float): Julian Century (Terrestrial time)
        dX (float): X correction (from EOP) [arcsec]
        dY (float): Y correction (from EOP) [arcsec]

    Returns:
        _type_: _description_
    """
    X, Y, s, a = _X_Y_s_a(t_tt)
    mat = np.array(
        [
            [1 - a * X * X, -a * X * Y, X],
            [-a * X * Y, 1 - a * Y * Y, Y],
            [-X, -Y, 1 - a * (X * X + Y * Y)],
        ]
    )

    Cs = np.cos(s)
    Ss = np.sin(s)

    rot3_s = np.array([[Cs, Ss, 0], [-Ss, Cs, 0], [0, 0, 1]])

    return mat @ rot3_s


def itrf_to_gcrs_matrices(
    xp: float,
    yp: float,
    jd_utc: float,
    dX: float = 0,
    dY: float = 0,
    deltaUT1_s: float = 0,
    deltaAT_s: float = 32,
):
    """Generates PN, R, and W matrices from ITRF <--> GCRS.

    - Obtain xp, yp, deltaUT1_s, dX, and dY from EOP data.
    - For vel and accel, modify the W matrix with golden rule


    Vallado 4e Algorithm 23 p. 221

    Args:
        xp (float): Polar x coord. of polar motion of CIP in ITRS [rad]
        yp (float): Polar y coord. of polar motion of CIP in ITRS [rad]
        jd_utc (float): Julian date (UTC)
        dX (float): X correction (from EOP) Defaults to 0.
        dY (float): Y correction (from EOP) Defaults to 0.
        deltaUT1_s (float): UTC to UT1 offset [s] Defaults to 0.
        deltaAT_s (float): UTC to TAI offset [s] Defaults to 32.

    Returns:
        tuple: (PN, R, W) Rotation matrices such that r_GCRS = PNRW * r_ITRF
    """

    jd_ut1 = jd_utc + deltaUT1_s * SEC_TO_DAY
    jd_tai = jd_utc + deltaAT_s * SEC_TO_DAY
    jd_tt = jd_tai + 32.184 * SEC_TO_DAY
    t_tt = jd_to_julian_centuries(jd_tt)

    W = W_matrix(xp, yp, t_tt)
    R = R_matrix(jd_ut1)
    PN = PN_matrix(t_tt, dX, dY)

    return PN, R, W


def calc_v_tirs(
    R: np.ndarray, v_tirs: np.ndarray, w_earth_tirs: np.ndarray, r_tirs: np.ndarray
):
    """Formula normally `R @ v_tirs + np.cross(w_cirs, r_tirs))`
    but we need have `w_tirs` so we use R and distribute:  `R @ (v_tirs + np.cross(w_tirs, r_tirs)) )` \\
    Vallado 4e p. 213

    Args:
        R (np.ndarray): Sidereal-rotation matrix (ITRS --> CIRS)
        v_tirs (np.ndarray): Velocity in TIRS system [m/s] or [km/s]
        w_earth_tirs (np.ndarray): Angular velocity of Earth in TIRS system
        r_tirs (np.ndarray): Position in TIRS system [m] or [km] (must be same as `v_tirs`)

    Returns:
        np.ndarray: velocity (CIRS) [m/s] or [km/s] (same as r and v inputs)
    """
    return R @ (v_tirs + np.cross(w_earth_tirs, r_tirs))


# TODO: ECI ECEF

# TODO: RAZEL

# TODO: SITE
