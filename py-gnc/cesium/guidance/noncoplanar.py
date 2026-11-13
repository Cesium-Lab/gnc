import numpy as np
from numpy.linalg import norm

from ..math.base import angle_between, unit
from ..math.rotations import wrap_to_2pi


def node_line_finder(rv_chaser: np.ndarray, rv_target: np.ndarray):
    """Line of nodes between two orbit planes, and how far the chaser is from it.

    Parameters
    ----------
    rv_chaser, rv_target : np.ndarray (6,)
        Chaser and target states in one inertial frame [m, m/s].

    Returns
    -------
    n_hat : np.ndarray (3,)
        Unit vector along `h_chaser x h_target`.
    d_ta : float
        True-anomaly sweep from the chaser to `n_hat`, in `[0, 2*pi)` [rad].
    h_chaser, h_target : np.ndarray (3,)
        Specific angular momenta [m2/s].
    """
    r_c, v_c = rv_chaser[0:3], rv_chaser[3:6]
    h_chaser = np.cross(r_c, v_c)
    h_target = np.cross(rv_target[0:3], rv_target[3:6])
    n_hat = unit(np.cross(h_chaser, h_target))

    # Negative if the chaser has already passed n_hat this revolution
    d_ta = wrap_to_2pi(angle_between(r_c, n_hat) * np.sign(np.dot(v_c, n_hat)))

    return n_hat, d_ta, h_chaser, h_target


# TODO: Example 6-4 p. 344
def guid_i_change(di: float, phi_fpa: float, v_i: float):
    """Delta-v for an inclination-only change.

    Parameters
    ----------
    di : float
        Change in inclination [rad].
    phi_fpa : float
        Flight path angle at burn point [rad].
    v_i : float
        Satellite speed at burn point [m/s].

    Returns
    -------
    dv : float
        Delta-v magnitude [m/s].

    References
    ----------
    Vallado 4e, Algorithm 39 (Inclination Only), p. 344.
    """
    dv = 2 * v_i * np.cos(phi_fpa) * np.sin(di / 2.0)
    return dv


# TODO: test with vallado above
def guid_plane_change_3d(rv_chaser: np.ndarray, h_target: np.ndarray):
    """Keep radial speed but just change the direction of planar speed"""
    h_hat = unit(h_target)
    r_hat = unit(rv_chaser[0:3])
    v = rv_chaser[3:6]
    v_radial = np.dot(v, r_hat) * r_hat
    v_planar = v - v_radial
    v_new = v_radial + unit(np.cross(h_hat, r_hat)) * norm(v_planar)
    return v_new - v


# TODO: Example 6-6 p. 349
def guid_combined_i_raan_change(i_i: float, i_f: float, draan: float, v_i: float):
    """Delta-v for an inclination an RAAN change of a circular orbit.

    Parameters
    ----------
    i_i, i_f : float
        Inital and final inclinations of the orbits [rad].
    draan : float
        Change in right ascension of the ascending node [rad].
    v_i : float
        Initial circular orbital speed [m/s].

    Returns
    -------
    dv : float
        Delta-v magnitude [m/s].

    References
    ----------
    Vallado 4e, Algorithm 41 (Combined Changes to i and Ω—Circular), p. 347.
    """
    ta = np.arccos(
        np.cos(i_i) * np.cos(i_f) + np.sin(i_i) * np.sin(i_f) * np.cos(draan)
    )
    dv = 2 * v_i * np.sin(ta / 2)
    return dv


# TODO: Example 6-5 p. 347
def guid_raan_change_circular(draan: float, i: float, v_i: float):
    """Delta-v for a RAAN-only change of a circular orbit.

    Parameters
    ----------
    draan : float
        Change in right ascension of the ascending node [rad].
    i : float
        Inclination of the orbit, unchanged by the maneuver [rad].
    v_i : float
        Circular orbital speed [m/s].

    Returns
    -------
    dv : float
        Delta-v magnitude [m/s].

    References
    ----------
    Vallado 4e, Algorithm 40 (Change in the Ascending Node—Circular), p. 347.
    """
    return guid_combined_i_raan_change(i, i, draan, v_i)
