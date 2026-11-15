"""Non-COES transformations"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.linalg import norm

from ..constants import (
    ARCSEC_TO_RAD,
    DEG_TO_RAD,
    J2000_JD,
    RAD_TO_DEG,
    SEC_TO_DAY,
    TAI_TO_TT,
)
from ..math.quaternion import unit
from ..math.rotations import dcm_x, dcm_z, wrap_to_2pi
from ..math.time import jd_to_julian_centuries
from ..world.constants import (
    ECC_EARTH,
    ECLIPTIC_ANGLE,
    MARS_POLE_DEC_J2000,
    MARS_POLE_DEC_RATE,
    MARS_POLE_RA_J2000,
    MARS_POLE_RA_RATE,
    MARS_PRIME_MERIDIAN_J2000,
    MARS_PRIME_MERIDIAN_RATE,
    R_EARTH,
    R_EARTH_POLAR,
    W_EARTH,
    W_MARS,
)
from .CIP.parse import approx_5th_deg_spline, get_summation, import_table

################################################################################
#               Intermediate
################################################################################


# TODO: test
def calc_dcm_rsw_wrt_pci(rv_PCI: np.ndarray) -> np.ndarray:
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


def calc_dcm_ntw_wrt_pci(rv_PCI: np.ndarray) -> np.ndarray:
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
#               SEZ <--> ECEF
################################################################################


def geodetic_ECEF_to_SEZ(
    r_sat_ecef_m: np.ndarray,
    v_sat_ecef_m: np.ndarray,
    lat: float,
    lon: float,
    alt_m=0.0,
    deg=True,
):
    """Return geodetic site azimuth, elevation, slant range, and their rates.

    Parameters
    ----------
    r_sat_ecef_m : ndarray, shape (3,)
        Satellite Earth-fixed position in meters.
    v_sat_ecef_m : ndarray, shape (3,)
        Satellite Earth-fixed velocity in meters per second.
    lat, lon : float
        Site geodetic latitude and east-positive longitude; degrees when
        deg=True, otherwise radians.
    alt_m : float, optional
        Site height above the reference ellipsoid in meters.
    deg : bool, optional
        Use degrees and deg/s for angles and angular rates when True (default),
        otherwise radians and rad/s. Also controls input latitude and longitude.

    Returns
    -------
    azimuth, elevation, rho : float
        Azimuth clockwise from north in [0, 360) degrees (or [0, 2*pi)),
        elevation above the local horizon, and site-to-satellite range in m.
        Azimuth is undefined for an exactly vertical line of sight.
    azimuth_dot, elevation_dot, rho_dot : float
        Angular rates in deg/s when deg=True, otherwise rad/s, followed by
        range rate in m/s (positive when receding). Angular rates are NaN
        when the horizontal range divided by slant range is at most 1e-9.

    Raises
    ------
    ValueError
        If the satellite and site coincide, making the viewing angles undefined.

    References
    ------
    Vallado 4e, Algorithm 27 (RAZEL), p. 265.
    """
    r_site_ecef_m = geodetic_LLA_to_ECEF(lat, lon, alt_m, deg)
    if deg:
        lat *= DEG_TO_RAD
        lon *= DEG_TO_RAD

    # dcm
    south = np.array(
        [np.sin(lat) * np.cos(lon), np.sin(lat) * np.sin(lon), -np.cos(lat)]
    )
    east = np.array([-np.sin(lon), np.cos(lon), 0])
    zenith = np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )
    dcm_sez_wrt_ecef = np.stack(
        (south, east, zenith)
    )  # columns makes sez->ecef so rows makes ecef->sez
    r_sat_sez = dcm_sez_wrt_ecef @ (r_sat_ecef_m - r_site_ecef_m)
    v_sat_sez = dcm_sez_wrt_ecef @ v_sat_ecef_m

    rho = norm(r_sat_sez)
    if rho == 0:
        raise ValueError("Satellite and site coincide; viewing angles are undefined")

    horizontal = np.hypot(r_sat_sez[0], r_sat_sez[1])
    elevation = np.arctan2(r_sat_sez[2], horizontal)

    vertical = horizontal / rho <= 1e-9
    azimuth = 0 if vertical else wrap_to_2pi(np.arctan2(r_sat_sez[1], -r_sat_sez[0]))

    rho_dot = np.dot(r_sat_sez, v_sat_sez) / rho
    if vertical:
        azimuth_dot = elevation_dot = np.nan
    else:
        azimuth_dot = (
            v_sat_sez[0] * r_sat_sez[1] - r_sat_sez[0] * v_sat_sez[1]
        ) / horizontal**2
        elevation_dot = (v_sat_sez[2] - rho_dot * np.sin(elevation)) / horizontal

    if deg:
        azimuth *= RAD_TO_DEG
        elevation *= RAD_TO_DEG
        azimuth_dot *= RAD_TO_DEG
        elevation_dot *= RAD_TO_DEG
    return azimuth, elevation, rho, azimuth_dot, elevation_dot, rho_dot


################################################################################
#               LLA <--> ECEF
################################################################################


def geodetic_LLA_to_ECEF(lat: float, lon: float, alt_m=0.0, deg=True):
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
    Vallado 4e, Algorithm 13 (ECEF TO LatLon), p. 173.
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

        E = (b * abs(z) - (a**2 - b**2)) / (a * r_horizontal_sat)
        F = (b * abs(z) + (a**2 - b**2)) / (a * r_horizontal_sat)

        P = 4 * (E * F + 1) / 3
        Q = 2 * (E**2 - F**2)
        D = P**3 + Q**2

        if D > 0:
            v = np.cbrt(np.sqrt(D) - Q) - np.cbrt(np.sqrt(D) + Q)
        else:
            v = 2 * np.sqrt(-P) * np.cos(np.arccos(Q / (P * np.sqrt(-P))) / 3)

        G = (np.sqrt(E**2 + v) + E) / 2
        t = np.sqrt(G**2 + (F - v * G) / (2 * G - E)) - G

        lat_gd = np.arctan2(a * (1 - t**2), 2 * b * t)
        lat_gd = np.copysign(lat_gd, z)

        alt_m = (r_horizontal_sat - a * t) * np.cos(lat_gd) + (abs(z) - b) * abs(
            np.sin(lat_gd)
        )

    if deg:
        lat_gd *= RAD_TO_DEG
        lon_gd *= RAD_TO_DEG

    return lat_gd, lon_gd, alt_m


def geocentric_LLA_to_PCPF(
    lat: float, lon: float, alt_m=0.0, r_body_m=R_EARTH, deg=True
):
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

    r_PCPF = np.array(
        [
            rho * np.cos(lat) * np.cos(lon),
            rho * np.cos(lat) * np.sin(lon),
            rho * np.sin(lat),
        ]
    )

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


###########################################################################
#                   HCI <--> ECI
###########################################################################


def calc_dcm_eci_wrt_hci():
    """HCI to ICRF/J2000 ECI"""
    return np.array(
        [
            [1, 0, 0],
            [
                0,
                np.cos(ECLIPTIC_ANGLE * DEG_TO_RAD),
                -np.sin(ECLIPTIC_ANGLE * DEG_TO_RAD),
            ],
            [
                0,
                np.sin(ECLIPTIC_ANGLE * DEG_TO_RAD),
                np.cos(ECLIPTIC_ANGLE * DEG_TO_RAD),
            ],
        ]
    )


################################################################################
#               ICRF <--> Mars (MCI, MCMF)
################################################################################

"""
IAU rotational-elements model for Mars, same construction as any IAU body:
- MCI  = R1(90 - delta_0) R3(90 + alpha_0) @ ICRF   (Mars mean equator of date, inertial)
- MCMF = R3(W) @ MCI                                (Mars body-fixed, i.e. SPICE IAU_MARS)

The MCI x-axis is the ascending node of the Mars equator on the ICRF equator.
For a strictly inertial frame (e.g. for propagation), evaluate at one fixed epoch
and reuse the DCM rather than re-evaluating it every step.

TODO: IAU 2015 (Archinal et al. 2018) adds periodic terms to alpha_0, delta_0, W;
less than 0.01 deg, so ignored here.
"""


def mars_rotational_elements(jd_tdb: float):
    """IAU Mars pole and prime meridian angles at a given epoch.

    Parameters
    ----------
    jd_tdb : float
        Julian date in TDB (TT is fine, since it is less than a millisecond difference).

    Returns
    -------
    alpha_0 : float
        Right ascension of the Mars north pole in ICRF [rad].
    delta_0 : float
        Declination of the Mars north pole in ICRF [rad].
    W : float
        Prime meridian angle, measured east along the Mars equator from the
        MCI x-axis (ascending node on the ICRF equator) [rad], wrapped to [0, 2pi).

    References
    ----------
    Folkner et al. (1997), "Interior Structure and Seasonal Mass Redistribution
        of Mars from Radio Tracking of Mars Pathfinder", Science 278, 1749-1752.
        Source of the pole (alpha_0, delta_0, rates) and rotation rate.
    Duxbury, Kirk, Archinal, and Neumann (2002), "Mars Geodesy/Cartography Working
        Group Recommendations on Mars Cartographic Constants and Coordinate Systems"
    """
    d = jd_tdb - J2000_JD
    T = jd_to_julian_centuries(jd_tdb)

    alpha_0 = (MARS_POLE_RA_J2000 + MARS_POLE_RA_RATE * T) * DEG_TO_RAD
    delta_0 = (MARS_POLE_DEC_J2000 + MARS_POLE_DEC_RATE * T) * DEG_TO_RAD
    W = wrap_to_2pi(  # orientation at epoch
        (MARS_PRIME_MERIDIAN_J2000 + MARS_PRIME_MERIDIAN_RATE * d) * DEG_TO_RAD
    )

    return alpha_0, delta_0, W


def calc_dcm_mci_wrt_eci(jd_tdb: float = J2000_JD):
    """DCM from ICRF/J2000 ECI axes to Mars-centered inertial (Mars equator) axes.
    BARELY changes during a given mission.

    Rotation only; translate to Mars-centered first (e.g. `r_sat_hci - r_mars_hci`).

    Parameters
    ----------
    jd_tdb : float, optional
        Epoch of the Mars pole orientation [JD, TDB]. Default is J2000, a fixed
        frame (Mars mean equator of J2000).

    Returns
    -------
    dcm_mci_wrt_eci : np.ndarray (3, 3)
        `r_MCI = dcm_mci_wrt_eci @ r_ECI`. Rows are the MCI axes in ECI.

    Notes
    -----
    First two rotations of the IAU 3-1-3 sequence
    `[W]_3 [pi/2 - Dec]_1 [pi/2 + RA]_3`. `pi/2 + RA` puts x on the
    ascending node of the body equator on the J2000 equator; `pi/2 - Dec` tilts z
    onto the body pole.

    References
    ----------
    NAIF, "PCK Required Reading", section "Orientation Models used by PCK Software".
        https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/pck.html
    """
    alpha_0, delta_0, _ = mars_rotational_elements(jd_tdb)

    # spun being intermediate axis
    dcm_spun_wrt_hci = dcm_z(np.pi / 2 + alpha_0)
    dcm_mci_wrt_spun = dcm_x(np.pi / 2 - delta_0)
    return dcm_mci_wrt_spun @ dcm_spun_wrt_hci


def calc_dcm_mci_wrt_hci(jd_tdb: float = J2000_JD):
    """DCM from HCI (ecliptic J2000) axes to Mars-centered inertial axes.
    BARELY changes during a given mission.

    Rotation only; translate to Mars-centered first (e.g. `r_sat_hci - r_mars_hci`).

    Parameters
    ----------
    jd_tdb : float, optional
        Epoch of the Mars pole orientation [JD, TDB]. Default is J2000.

    Returns
    -------
    dcm_mci_wrt_hci : np.ndarray (3, 3)
        `r_MCI = dcm_mci_wrt_hci @ r_HCI`.
    """
    return calc_dcm_mci_wrt_eci(jd_tdb) @ calc_dcm_eci_wrt_hci()


def calc_dcm_mcmf_wrt_mci(jd_tdb: float):
    """DCM from Mars-centered inertial to Mars-centered Mars-fixed (IAU_MARS) axes.

    Both frames use the Mars pole at `jd_tdb`. Rotation only; for states use
    `mci_to_mcmf` / `mcmf_to_mci`, which add the `w_mars x r` velocity term.

    Parameters
    ----------
    jd_tdb : float
        Julian date in TDB.

    Returns
    -------
    dcm_mcmf_wrt_mci : np.ndarray (3, 3)
        `r_MCMF = dcm_mcmf_wrt_mci @ r_MCI`.
    """
    _, _, W = mars_rotational_elements(jd_tdb)

    return dcm_z(W)


def mci_to_mcmf(
    r_mci: np.ndarray, v_mci: np.ndarray, jd_tdb: float
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a position and velocity from Mars inertial (MCI) to Mars-fixed (MCMF).

    `v_MCMF = C @ v_MCI - w_mars x r_MCMF`, with `C = calc_dcm_mcmf_wrt_mci(jd_tdb)` and
    `w_mars = [0, 0, W_MARS]` (Mars pole is the MCI/MCMF z-axis).

    Parameters
    ----------
    r_mci : np.ndarray (3,)
        Position in MCI [m] (or [km], matching `v_mci`).
    v_mci : np.ndarray (3,)
        Inertial velocity in MCI [m/s] (or [km/s], matching `r_mci`).
    jd_tdb : float
        Julian date in TDB.

    Returns
    -------
    r_mcmf : np.ndarray (3,)
        Position in MCMF, same units as `r_mci`.
    v_mcmf : np.ndarray (3,)
        Velocity relative to the rotating Mars-fixed frame, same units as `v_mci`.

    References
    ----------
    Vallado 4e, Section 3.7, p. 203 (rotating-frame velocity, as in `pcpf_to_pci`).
    """
    dcm_mcmf_wrt_mci = calc_dcm_mcmf_wrt_mci(jd_tdb)
    r_mcmf = dcm_mcmf_wrt_mci @ r_mci
    v_mcmf = dcm_mcmf_wrt_mci @ v_mci - np.cross([0, 0, W_MARS], r_mcmf)

    return r_mcmf, v_mcmf


def mcmf_to_mci(
    r_mcmf: np.ndarray, v_mcmf: np.ndarray, jd_tdb: float
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a position and velocity from Mars-fixed (MCMF) to Mars inertial (MCI).

    `v_MCI = C.T @ (v_MCMF + w_mars x r_MCMF)`, inverse of `mci_to_mcmf`.

    Parameters
    ----------
    r_mcmf : np.ndarray (3,)
        Position in MCMF [m] (or [km], matching `v_mcmf`).
    v_mcmf : np.ndarray (3,)
        Velocity relative to the Mars-fixed frame [m/s] (or [km/s]).
    jd_tdb : float
        Julian date in TDB.

    Returns
    -------
    r_mci : np.ndarray (3,)
        Position in MCI, same units as `r_mcmf`.
    v_mci : np.ndarray (3,)
        Inertial velocity in MCI, same units as `v_mcmf`.

    References
    ----------
    Vallado 4e, Section 3.7, p. 203 (rotating-frame velocity, as in `pcpf_to_pci`).
    """
    dcm_mci_wrt_mcmf = calc_dcm_mcmf_wrt_mci(jd_tdb).T
    r_mci = dcm_mci_wrt_mcmf @ r_mcmf
    v_mci = dcm_mci_wrt_mcmf @ (v_mcmf + np.cross([0, 0, W_MARS], r_mcmf))

    return r_mci, v_mci


def calc_dcm_mcmf_wrt_eci(jd_tdb: float):
    """DCM of Mars-fixed (IAU_MARS) axes from ICRF/J2000 ECI axes at `jd_tdb`.

    Parameters
    ----------
    jd_tdb : float
        Julian date in TDB.

    Returns
    -------
    dcm_mcmf_wrt_eci : np.ndarray (3, 3)
        `r_MCMF = dcm_mcmf_wrt_eci @ r_ECI`.
    """
    return calc_dcm_mcmf_wrt_mci(jd_tdb) @ calc_dcm_mci_wrt_eci(jd_tdb)


################################################################################
#               PCI <--> PCPF
################################################################################


# TODO: test
def pcpf_to_pci(
    r_PCPF: np.ndarray,
    v_PCPF: np.ndarray,
    t: float,
    theta_ref: float,
    t_ref: float,
    w_body=W_EARTH,
):
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
    v_PCI = dcm_PCI_wrt_PCPF @ (v_PCPF + np.cross([0, 0, w_body], r_PCPF))

    return r_PCI, v_PCI


#########################################################################################################
#               cart to radec
#########################################################################################################


def cartesian_to_radec(r_cart: np.ndarray, v_cart: np.ndarray, deg=True):
    """Vallado 4e. Algorithm 25 (Geocentric Radec), p. 259 (and Curtis Equation 4.5 p. 155 for position)"""
    r_hat = unit(r_cart)
    v_hat = unit(v_cart)

    horizontal = np.hypot(r_hat[0], r_hat[1])

    dla = np.arcsin(r_hat[2])  # -pi to pi

    if abs(horizontal) < 1e-12:  # meters
        rla = np.arctan2(v_hat[1], v_hat[0])
    else:
        rla = np.arctan2(r_hat[1], r_hat[0])

    rla = wrap_to_2pi(rla)

    rho_dot = np.dot(r_cart, v_cart) / norm(r_cart)
    rla_dot = (v_cart[0] * r_cart[1] - v_cart[1] * r_cart[0]) / (
        -(r_cart[1] ** 2) - r_cart[0] ** 2
    )
    dla_dot = (v_cart[2] - rho_dot * r_hat[2]) / np.sqrt(
        r_cart[0] ** 2 + r_cart[1] ** 2
    )

    if deg:
        dla *= RAD_TO_DEG
        rla *= RAD_TO_DEG

    return rla, dla, norm(r_cart), rla_dot, dla_dot, rho_dot


def radec_to_cartesian(
    rla: float,
    dla: float,
    rho: float,
    rla_dot: float,
    dla_dot: float,
    rho_dot: float,
    deg=True,
):
    """Vallado 4e. Equation 4-1 p. 257 and Vallado 4e. Equation 4-2 p. 258 (and Curtis Equation 4.5 p. 155 for position)"""

    if deg:
        rla *= DEG_TO_RAD
        dla *= DEG_TO_RAD

    cr = np.cos(rla)
    sr = np.sin(rla)
    cd = np.cos(dla)
    sd = np.sin(dla)

    r = np.array([cd * cr * rho, cd * sr * rho, sd * rho])

    v = np.array(
        [
            rho_dot * cd * cr - rho * sd * cr * dla_dot - rho * cd * sr * rla_dot,
            rho_dot * cd * sr - rho * sd * sr * dla_dot + rho * cd * cr * rla_dot,
            rho_dot * sd + rho * cd * dla_dot,
        ]
    )

    return r, v


def azel_to_radec(az: float, el: float, lat: float, lst: float, deg=True):
    """Convert site azimuth/elevation to topocentric right ascension/declination.

    Parameters
    ----------
    az : float
        Azimuth, clockwise from north.
    el : float
        Elevation above the local horizon.
    lat : float
        Site latitude (geodetic for SEZ from `geodetic_ECEF_to_SEZ`).
    lst : float
        Local sidereal time (Greenwich sidereal angle + east longitude).
    deg : bool, optional
        Inputs and outputs in degrees when True (default), otherwise radians.

    Returns
    -------
    ra : float
        Topocentric right ascension in [0, 360) degrees (or [0, 2*pi)).
    dec : float
        Topocentric declination.

    References
    ----------
    Vallado 4e, Algorithm 28 (AzElToRaDec), p. 268.
    """
    if deg:
        az *= DEG_TO_RAD
        el *= DEG_TO_RAD
        lat *= DEG_TO_RAD
        lst *= DEG_TO_RAD

    sin_dec = np.sin(el) * np.sin(lat) + np.cos(el) * np.cos(lat) * np.cos(az)
    dec = np.arcsin(np.clip(sin_dec, -1.0, 1.0))

    # cos(dec) factor drops out of atan2.
    sin_lha = -np.sin(az) * np.cos(el)
    cos_lha = np.cos(lat) * np.sin(el) - np.sin(lat) * np.cos(az) * np.cos(el)
    lha = np.arctan2(sin_lha, cos_lha)

    ra = wrap_to_2pi(lst - lha)

    if deg:
        ra *= RAD_TO_DEG
        dec *= RAD_TO_DEG

    return ra, dec


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


def calc_dcm_tirs_wrt_itrf(xp: float, yp: float, t_tt: float):
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


def theta_g(jd_ut1):
    return 2 * np.pi * (0.779057273264 + 1.00273781191135448 * (jd_ut1 - 2451545))


def calc_dcm_cirs_wrt_tirs(jd_ut1: float):
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
@lru_cache(maxsize=16384)
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

    Notes
    -----
    Caches the last 16,384 exact epochs, allowing repeated trajectory
    conversions to reuse the CIP series without rounding timestamps.
    Clear this cache as well as ``import_table`` after editing CIP tables.

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
def calc_dcm_gcrs_wrt_cirs(t_tt: float, dX=0.0, dY=0.0):
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
    X += dX
    Y += dY
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


def calc_dcm_gcrs_wrt_itrf(jd_utc: float, eop: EOP = EOP()):
    """Compute the matrix for the ITRF <--> GCRS transformation.

    Parameters
    ----------
    jd_utc : float
        Julian date in UTC.
    eop : EOP, optional
        Earth orientation parameters at `jd_utc`. Default is `EOP()` (zero
        polar motion, UT1 = UTC, 37 leap seconds).

    Returns
    -------
    dcm_gcrs_wrt_itrf : np.ndarray (3, 3)
        Transforms a vector/matrix from itrf to gcrs with `r_GCRS = dcm_gcrs_wrt_itrf @ r_ITRF`

    Notes
    -----
    For velocity, use
    `itrf_to_gcrs`, which adds the Earth-rotation term at the R step.

    References
    ----------
    Vallado 4e, Algorithm 23 (IAU-2000, CIO-Based), p. 219.
    """

    jd_ut1 = jd_utc + eop.deltaUT1_s * SEC_TO_DAY
    jd_tai = jd_utc + eop.deltaAT_s * SEC_TO_DAY
    jd_tt = jd_tai + TAI_TO_TT * SEC_TO_DAY
    t_tt = jd_to_julian_centuries(jd_tt)

    W = calc_dcm_tirs_wrt_itrf(eop.xp, eop.yp, t_tt)
    R = calc_dcm_cirs_wrt_tirs(jd_ut1)
    PN = calc_dcm_gcrs_wrt_cirs(t_tt, eop.dX, eop.dY)

    return PN @ R @ W


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
    Vallado 4e, Algorithm 23 (IAU-2000, CIO-Based), p. 219.
    """

    jd_ut1 = jd_utc + eop.deltaUT1_s * SEC_TO_DAY
    jd_tai = jd_utc + eop.deltaAT_s * SEC_TO_DAY
    jd_tt = jd_tai + TAI_TO_TT * SEC_TO_DAY
    t_tt = jd_to_julian_centuries(jd_tt)

    W = calc_dcm_tirs_wrt_itrf(eop.xp, eop.yp, t_tt)
    R = calc_dcm_cirs_wrt_tirs(jd_ut1)
    PN = calc_dcm_gcrs_wrt_cirs(t_tt, eop.dX, eop.dY)

    r_tirs = W @ r_itrf
    v_tirs = W @ v_itrf
    w_earth_tirs = np.array([0, 0, W_EARTH * (1 - eop.lod_s * SEC_TO_DAY)])

    r_gcrs = PN @ R @ r_tirs
    v_gcrs = PN @ R @ (v_tirs + np.cross(w_earth_tirs, r_tirs))

    return r_gcrs, v_gcrs


def gcrs_to_itrf(
    r_gcrs: np.ndarray,
    v_gcrs: np.ndarray,
    jd_utc: float,
    eop: EOP = EOP(),
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a GCRS (ECI) position and velocity into ITRF (ECEF).

    Parameters
    ----------
    r_gcrs, v_gcrs : ndarray, shape (3,)
        Inertial position and velocity, in m and m/s or km and km/s.
    jd_utc : float
        Absolute Julian date in UTC, not elapsed seconds.
    eop : EOP, optional
        Earth orientation parameters, with the same defaults and model
        limitations as `itrf_to_gcrs`.

    Returns
    -------
    r_itrf, v_itrf : ndarray, shape (3,)
        Earth-fixed position and velocity relative to the rotating Earth,
        preserving the input units.

    References
    ----------
    Vallado 4e, Algorithm 23 (IAU-2000, CIO-Based), p. 219.

    Notes
    -----
    Like the forward model, omits time derivatives of polar motion and precession/nutation
    """
    dcm_gcrs_wrt_itrf = calc_dcm_gcrs_wrt_itrf(jd_utc, eop)
    jd_tt = jd_utc + (eop.deltaAT_s + 32.184) * SEC_TO_DAY
    W = calc_dcm_tirs_wrt_itrf(eop.xp, eop.yp, jd_to_julian_centuries(jd_tt))
    w_earth_tirs = np.array([0, 0, W_EARTH * (1 - eop.lod_s * SEC_TO_DAY)])

    r_itrf = dcm_gcrs_wrt_itrf.T @ r_gcrs
    v_itrf = dcm_gcrs_wrt_itrf.T @ v_gcrs - W.T @ np.cross(w_earth_tirs, W @ r_itrf)
    return r_itrf, v_itrf


# TODO ecliptic lat,lon
