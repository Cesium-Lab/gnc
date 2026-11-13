"""Ideal (lossless) surface-to-orbit delta-v: no atmosphere, no gravity or steering losses."""

import numpy as np

from ..math.base import unit
from .coplanar import guid_hohmann


# TODO: verify
def launch_azimuth(lat: float, inc: float, ascending: bool = True) -> float:
    """Launch azimuth [rad, clockwise from north] into inclination `inc` from latitude `lat` [rad].

    Spherical-trig relation sin(az) = cos(i) / cos(lat); needs |lat| <= i <= 180 deg - |lat|.
    `ascending` picks the northbound pass (az in [-90, 90] deg), else the southbound one.
    Ignores the site's rotation (the inertial azimuth).
    """
    s = np.cos(inc) / np.cos(lat)
    if abs(s) > 1.0:
        raise ValueError(
            "inclination not reachable from this latitude (need |lat| <= i)"
        )
    az = np.arcsin(s)
    return az if ascending else np.pi - az


# TODO: verify
def ideal_ascent_dv(
    r_site: float, r_orbit: float, v_east: float, azimuth: float, mu: float
) -> dict:
    """Minimum two-impulse delta-v from a rotating body's surface to a circular orbit.

    Hohmann transfer from the site radius to `r_orbit`: a horizontal burn at the site
    along `azimuth`, then circularization at apoapsis. The site already moves east at
    `v_east` (inertial, e.g. w * r * cos(lat)); its along-track part helps and its
    cross-track part must be cancelled.

    Parameters
    ----------
    r_site : float
        Site distance from the body center [m].
    r_orbit : float
        Target circular-orbit radius [m].
    v_east : float
        Inertial eastward speed of the site from body rotation [m/s].
    azimuth : float
        Launch azimuth, clockwise from north [rad] (90 deg = due east).
    mu : float
        Gravitational parameter [m3/s2].

    Returns
    -------
    dict
        dv_burn1, dv_circularize, dv_total [m/s]; dv_no_rotation [m/s] for a
        non-rotating body; rotation_gain [m/s] (dv_no_rotation - dv_total);
        t_coast [s] from burn 1 to apoapsis; v_periapsis [m/s] inertial.
    """
    _, t_trans, dv_a, dv_b = guid_hohmann(r_site, r_orbit, mu)
    v_peri = np.sqrt(mu / r_site) + dv_a  # inertial speed needed at the site
    # Site velocity in local (along-track, cross-track) axes of the launch direction
    v_along, v_cross = v_east * np.sin(azimuth), v_east * np.cos(azimuth)
    dv_1 = np.hypot(v_peri - v_along, v_cross)
    no_rot = v_peri + dv_b
    return {
        "dv_burn1": dv_1,
        "dv_circularize": dv_b,
        "dv_total": dv_1 + dv_b,
        "dv_no_rotation": no_rot,
        "rotation_gain": no_rot - (dv_1 + dv_b),
        "t_coast": t_trans,
        "v_periapsis": v_peri,
    }


# TODO: verify
def azimuth_from_plane(r_site_inertial: np.ndarray, h_hat: np.ndarray) -> float:
    """Launch azimuth [rad, clockwise from north] of the orbit plane with normal `h_hat` at a site.

    The site must lie in the plane; inertial frame with +z along the body's spin axis.
    """
    r_hat = unit(r_site_inertial)
    east = unit(np.cross([0.0, 0.0, 1.0], r_hat))
    north = np.cross(r_hat, east)
    t_hat = unit(np.cross(h_hat, r_hat))  # along-track direction at the site
    return float(np.arctan2(t_hat @ east, t_hat @ north))
