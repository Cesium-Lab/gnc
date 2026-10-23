"""Non-COES transformations"""

from dataclasses import dataclass
import numpy as np

from ..world import R_EARTH, R_EARTH_POLAR, ECC_EARTH, W_EARTH
from ..constants import DEG_TO_RAD, SEC_TO_DAY, ARCSEC_TO_RAD
from ..math.time import jd_to_julian_centuries
from ..math.rotations import dcm_z
from .CIP.parse import import_table, get_summation
from ..math.quaternion import unit

################################################################################
#               Intermediate
################################################################################

# TODO: test
def PCI_to_RSW(rv_PCI: np.ndarray) -> np.ndarray:
    """DCM from the planet-centered inertial (PCI) frame to the satellite RSW frame.

    - `R` - radial outward
    - `S` - along-track
    - `W` - orbit normal (angular momentum)

    Parameters
    ----------
    rv_PCI : np.ndarray (6,)
        Satellite state in PCI, `[x, y, z, vx, vy, vz]` [m, m/s]. Any units.

    Returns
    -------
    dcm_RSW_wrt_PCI : np.ndarray (3, 3)
        Columns are the PCI unit vectors expressed in RSW, so `x_RSW = dcm @ x_PCI`.
    """
    r, v = rv_PCI[0:3], rv_PCI[3:6]
    r_hat = unit(r)
    w_hat = unit(np.cross(r, v))
    s_hat = np.cross(w_hat, r_hat)

    return np.vstack((r_hat, s_hat, w_hat))

def PCI_to_NTW(rv_PCI: np.ndarray) -> np.ndarray:
    """DCM from the planet-centered inertial (PCI) frame to the satellite NTW frame.

    NTW axes are N (in-plane, normal to velocity), T (tangent, along velocity),
    and W (orbit normal, along the angular momentum).
    - `N` - in-plane normal to velocity
    - `T` - tangent along velocity
    - `W` - orbit normal (angular momentum)

    Parameters
    ----------
    rv_PCI : np.ndarray (6,)
        Satellite state in PCI, `[x, y, z, vx, vy, vz]` [m, m/s]. Any units.

    Returns
    -------
    dcm_NTW_wrt_PCI : np.ndarray (3, 3)
        Columns are the PCI unit vectors expressed in NTW, so `x_NTW = dcm @ x_PCI`.
    """
    r, v = rv_PCI[0:3], rv_PCI[3:6]
    w_hat = unit(np.cross(r, v))
    t_hat = unit(v)
    n_hat = np.cross(t_hat, w_hat)

    return np.vstack((n_hat, t_hat, w_hat))

################################################################################
#               LLA <--> ECEF
################################################################################


def geodetic_LLA_to_ECEF(lat: float, lon: float, alt_m = 0.0, deg = True):
    """Convert WGS84 geodetic lat, lon, and alt to ECEF.

    Parameters
    ----------
    lat : float
        Geodetic latitude of the site [deg if `deg` is True, else rad].
    lon : float
        Longitude of the site, positive east [deg if `deg` is True, else rad].
    alt_m : float, optional
        Height above the WGS84 reference ellipsoid [m]. Default is 0.0.
    deg : bool, optional
        If True, `lat` and `lon` are in degrees; otherwise radians. Default is True.

    Returns
    -------
    r_ECEF : np.ndarray (3,)
        Earth-centered, Earth-fixed position of the site [m].

    References
    ----------
    Vallado 4e, Example 3-1, p. 140.
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
    """Convert ECEF position to WGS84 geodetic lat, lon, and alt.

    Uses Borkowski closed-form method.

    Parameters
    ----------
    r_ECEF : np.ndarray (3,)
        Earth-centered, Earth-fixed position of the point (e.g. satellite or site) [m].
    deg : bool, optional
        If True, return `lat` and `lon` in degrees; otherwise radians. Default is True.

    Returns
    -------
    lat : float
        Geodetic latitude [deg if `deg` is True, else rad].
    lon : float
        Longitude, positive east [deg if `deg` is True, else rad].
    alt_m : float
        Height above WGS84 reference ellipsoid [m].

    Raises
    ------
    ValueError
        If `r_ECEF` is at Earth's center (subject to 1e-9m norm)

    References
    ----------
    Vallado 4e, Algorithm 13, p. 173.
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
    """Convert geocentric latitude, longitude, and altitude to PCPF position.

    Assumes a spherical body.

    Parameters
    ----------
    lat : float
        Geocentric latitude [deg if `deg` is True, else rad].
    lon : float
        Longitude, positive east [deg if `deg` is True, else rad].
    alt_m : float, optional
        Altitude above the spherical body surface [m]. Default is 0.0.
    r_body_m : float, optional
        Radius of the spherical body [m]. Default is `R_EARTH`.
    deg : bool, optional
        If True, `lat` and `lon` are in degrees; otherwise radians. Default is True.

    Returns
    -------
    r_PCPF : np.ndarray (3,)
        Planet-centered, planet-fixed position [m].
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
    """Convert PCPF position to geocentric lat, lon, and alt.

    Assumes a spherical body.

    Parameters
    ----------
    r_PCPF : np.ndarray (3,)
        Planet-centered, planet-fixed position of the point [m].
    r_body_m : float, optional
        Radius of the spherical body [m]. Default is `R_EARTH`.
    deg : bool, optional
        If True, return `lat` and `lon` in degrees; otherwise radians. Default is True.

    Returns
    -------
    lat : float
        Geocentric latitude [deg if `deg` is True, else rad].
    lon : float
        Longitude, positive east [deg if `deg` is True, else rad].
    alt_m : float
        Altitude above the spherical body surface [m].

    Raises
    ------
    ValueError
        If `r_PCPF` is at the body center (subject to 1e-9m norm)
    """
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
#               PCI <--> PCPF
################################################################################

# TODO: test
def PCI_from_PCPF(r_PCPF: np.ndarray, v_PCPF: np.ndarray, t: float, theta_ref: float, t_ref: float, w_body = W_EARTH):
    """Convert planet-fixed (PCPF) position and velocity to planet-centered inertial (PCI).

    Simplified version of the ITRF to GCRS reduction. Uses a constant body rotation
    rate about the z-axis, neglecting precession, nutation, and polar motion.

    Parameters
    ----------
    r_PCPF : np.ndarray (3,)
        Position in PCPF [m] (or [km], matching `v_PCPF`).
    v_PCPF : np.ndarray (3,)
        Velocity in PCPF [m/s] (or [km/s]).
    t : float
        Current timestamp [s].
    theta_ref : float
        Rotation angle of PCPF with respect to PCI at `t_ref` (e.g. GMST) [rad].
    t_ref : float
        Timestamp of the reference epoch for `theta_ref` [s].
    w_body : float, optional
        Body rotation rate about the PCI z-axis [rad/s]. Default is `W_EARTH`.

    Returns
    -------
    r_PCI : np.ndarray (3,)
        Inertial position, same units as `r_PCPF`.
    v_PCI : np.ndarray (3,)
        Inertial velocity, same units as `v_PCPF`.

    References
    ----------
    Vallado 4e, Section 3.7, p. 203.
    """
    # Rotation angle
    theta_PCPF_wrt_PCI = (theta_ref + w_body * (t - t_ref)) % (2 * np.pi)
    
    dcm_PCI_wrt_PCPF = dcm_z(theta_PCPF_wrt_PCI).T
    
    r_PCI = dcm_PCI_wrt_PCPF @ r_PCPF
    v_PCI = dcm_PCI_wrt_PCPF @ (v_PCPF + np.cross([0,0,w_body], r_PCPF))
    
    return r_PCI, v_PCI

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
    """Evaluate the 5th-degree polynomial part of the IAU 2006 CIP series.

    Parameters
    ----------
    t_tt : float
        Julian centuries since J2000 in Terrestrial Time (TT).
    coeffs : array_like (6,)
        Polynomial coefficients, constant term first [microarcsec].

    Returns
    -------
    float
        Polynomial value [microarcsec].
    """
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

    Computes TIO locator `s'` from `t_tt` with an accurate linear approximation.

    Parameters
    ----------
    xp : float
        x polar-motion coordinate of CIP in ITRS, from EOP data [rad].
    yp : float
        y polar-motion coordinate of CIP in ITRS, from EOP data [rad].
    t_tt : float
        Julian centuries since J2000 in Terrestrial Time (TT).

    Returns
    -------
    W : np.ndarray (3, 3)
        Polar-motion matrix [W(t)], taking ITRF vectors to TIRS.

    Notes
    -----
    Could be augmented with ocean tides and libration (not yet implemented).

    References
    ----------
    Vallado 4e, p. 212.
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
    """Sidereal rotation matrix for ITRF <--> GCRS transformation.

    Computes Earth Rotation Angle (ERA) from `jd_ut1`.

    Parameters
    ----------
    jd_ut1 : float
        Julian date in UT1.

    Returns
    -------
    R : np.ndarray (3, 3)
        Sidereal-rotation matrix [R(t)], taking TIRS vectors to CIRS.

    References
    ----------
    Vallado 4e, Equation 3-62, p. 213.
    """

    # Vallado 4e 3-62 p. 213
    theta_ERA = 2 * np.pi * (0.779057273264 + 1.00273781191135448 * (jd_ut1 - 2451545))

    C = np.cos(-theta_ERA)
    S = np.sin(-theta_ERA)
    return np.array([[C, S, 0], [-S, C, 0], [0, 0, 1]])


# TODO: fix with dX and dY
def _X_Y_s_a(t_tt: float):
    """Compute CIP coordinates `X`, `Y`, CIO locator `s`, and the `a` term.

    Parameters
    ----------
    t_tt : float
        Julian centuries since J2000 in Terrestrial Time (TT).

    Returns
    -------
    X : float
        x-coordinate of the CIP unit vector in the GCRS [rad].
    Y : float
        y-coordinate of the CIP unit vector in the GCRS [rad].
    s : float
        CIO locator [rad].
    a : float
        Approximate `a` term used in the precession-nutation matrix.

    References
    ----------
    Vallado 4e, p. 213-214.
    """
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


# TODO: fix to use dX and dY
def PN_matrix(t_tt: float, dX=0.0, dY=0.0):
    """Precession-nutation matrix for the ITRF <--> GCRS transformation.

    Computes intermediate `a` term with an approximation.

    Parameters
    ----------
    t_tt : float
        Julian centuries since J2000 in Terrestrial Time (TT).
    dX : float, optional
        CIP X correction from EOP data [rad]. Default is 0.0. Currently unused.
    dY : float, optional
        CIP Y correction from EOP data [rad]. Default is 0.0. Currently unused.

    Returns
    -------
    PN : np.ndarray (3, 3)
        Precession-nutation matrix [PN(t)], taking CIRS vectors to GCRS.

    References
    ----------
    Vallado 4e, p. 213.
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


@dataclass(frozen=True)
class EOP:
    """Earth orientation parameters for a single epoch (from IERS data).

    Fields other than `deltaAT_s` default to zero. Leaving them at zero gives a
    lower-accuracy transformation (ignoring polar motion alone is ~10 m in LEO).

    Attributes
    ----------
    xp, yp : float
        Polar-motion coordinates of the CIP in the ITRS [rad].
    dX, dY : float
        CIP offset corrections [rad].
    deltaUT1_s : float
        UT1 - UTC [s].
    deltaAT_s : float
        TAI - UTC (leap seconds) [s]. Default is 37 (valid since 2017).
    lod_s : float
        Excess length of day [s]. (Vallado Equation 3-75)
    """

    xp: float = 0.0
    yp: float = 0.0
    dX: float = 0.0
    dY: float = 0.0
    deltaUT1_s: float = 0.0
    deltaAT_s: float = 37.0
    lod_s: float = 0.0


def itrf_to_gcrs_matrices(jd_utc: float, eop: EOP = EOP()):
    """Compute the PN, R, and W matrices for the ITRF <--> GCRS transformation.

    Parameters
    ----------
    jd_utc : float
        Julian date in UTC.
    eop : EOP, optional
        Earth orientation parameters at `jd_utc`. Default is `EOP()` (zero
        polar motion, UT1 = UTC, 37 leap seconds).

    Returns
    -------
    PN : np.ndarray (3, 3)
        Precession-nutation matrix (CIRS to GCRS).
    R : np.ndarray (3, 3)
        Sidereal-rotation matrix (TIRS to CIRS).
    W : np.ndarray (3, 3)
        Polar-motion matrix (ITRF to TIRS).

    Notes
    -----
    Position transforms as `r_GCRS = PN @ R @ W @ r_ITRF`. For velocity, use
    `itrf_to_gcrs`, which adds the Earth-rotation term at the R step.

    References
    ----------
    Vallado 4e, Algorithm 23, p. 221.
    """

    jd_ut1 = jd_utc + eop.deltaUT1_s * SEC_TO_DAY
    jd_tai = jd_utc + eop.deltaAT_s * SEC_TO_DAY
    jd_tt = jd_tai + 32.184 * SEC_TO_DAY
    t_tt = jd_to_julian_centuries(jd_tt)

    W = W_matrix(eop.xp, eop.yp, t_tt)
    R = R_matrix(jd_ut1)
    PN = PN_matrix(t_tt, eop.dX, eop.dY)

    return PN, R, W


def itrf_to_gcrs(
    r_itrf: np.ndarray,
    v_itrf: np.ndarray,
    jd_utc: float,
    eop: EOP = EOP(),
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a satellite's position and velocity from ITRF (ECEF) to GCRS (ECI).

    Velocity picks up the Earth-rotation term at the TIRS to CIRS step:
    `v_GCRS = PN @ R @ (W @ v_ITRF + w_earth x (W @ r_ITRF))`.

    Parameters
    ----------
    r_itrf : np.ndarray (3,)
        Satellite position in ITRF [m] (or [km], matching `v_itrf`).
    v_itrf : np.ndarray (3,)
        Satellite velocity relative to ITRF [m/s] (or [km/s], matching `r_itrf`).
    jd_utc : float
        Julian date in UTC.
    eop : EOP, optional
        Earth orientation parameters at `jd_utc`. `eop.lod_s` slightly reduces
        Earth's rotation rate in the velocity term. Default is `EOP()`.

    Returns
    -------
    r_gcrs : np.ndarray (3,)
        Satellite position in GCRS, same units as `r_itrf`.
    v_gcrs : np.ndarray (3,)
        Satellite velocity in GCRS, same units as `v_itrf`.

    References
    ----------
    Vallado 4e, Algorithm 23, p. 221.
    """
    PN, R, W = itrf_to_gcrs_matrices(jd_utc, eop)

    r_tirs = W @ r_itrf
    v_tirs = W @ v_itrf
    w_earth_tirs = np.array([0, 0, W_EARTH * (1 - eop.lod_s * SEC_TO_DAY)])

    r_gcrs = PN @ R @ r_tirs
    v_gcrs = PN @ R @ (v_tirs + np.cross(w_earth_tirs, r_tirs))

    return r_gcrs, v_gcrs


# TODO: ECI ECEF

# TODO: RAZEL

# TODO: SITE
