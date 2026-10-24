import numpy as np

from ...constants import DEG_TO_RAD
from ...math.coes import E_from_ta
from ...world.constants import MU_EARTH


# TODO: Example 6-1 p. 326
def guid_hohmann(r_i: float, r_f: float, mu=MU_EARTH):
    """Hohmann transfer between two coplanar circular orbits.

    Parameters
    ----------
    r_i : float
        Radius of initial circular orbit [m].
    r_f : float
        Radius of final circular orbit [m].
    mu : float, optional
        Gravitational parameter of central body [m3/s2]. Default is `MU_EARTH`.

    Returns
    -------
    a_trans : float
        Semi-major axis of transfer ellipse [m].
    t_trans : float
        Transfer time (half transfer-orbit period) [s].
    dv_a : float
        Delta-v of first burn, at departure; positive is prograde [m/s].
    dv_b : float
        Delta-v of second burn, at arrival; positive is prograde [m/s].

    References
    ----------
    Vallado 4e, Algorithm 36, p. 325.
    """
    a_trans = (r_i + r_f) / 2.0

    v_i = np.sqrt(mu / r_i)
    v_trans_a = np.sqrt(2 * mu / r_i - mu / a_trans)
    v_f = np.sqrt(mu / r_f)
    v_trans_b = np.sqrt(2 * mu / r_f - mu / a_trans)

    dv_a = v_trans_a - v_i
    dv_b = v_f - v_trans_b

    t_trans = np.pi * np.sqrt(a_trans**3 / mu)

    return a_trans, t_trans, dv_a, dv_b


# TODO: Example 6-2 p. 327
def guid_bielliptic(
    r_i: float, r_b: float, r_f: float, mu=MU_EARTH
) -> tuple[float, float, float, float, float, float, float]:
    """Bi-elliptic transfer between two coplanar circular orbits.

    Parameters
    ----------
    r_i : float
        Radius of initial circular orbit [m].
    r_b : float
        Radius of intermediate burn point, i.e. apoapsis of both transfer
        ellipses [m].
    r_f : float
        Radius of final circular orbit [m].
    mu : float, optional
        Gravitational parameter of central body [m3/s2]. Default is `MU_EARTH`.

    Returns
    -------
    a_trans1 : float
        Semi-major axis of first transfer ellipse [m].
    a_trans2 : float
        Semi-major axis of second transfer ellipse [m].
    t_trans1 : float
        Time on first transfer ellipse [s].
    t_trans2 : float
        Time on second transfer ellipse [s].
    dv_a : float
        Delta-v of first burn, at `r_i`; positive is prograde [m/s].
    dv_b : float
        Delta-v of second burn, at `r_b`; positive is prograde [m/s].
    dv_c : float
        Delta-v of third burn, at `r_f`; positive is prograde [m/s].

    References
    ----------
    Vallado 4e, Algorithm 37, p. 326.
    """
    a_trans1 = (r_i + r_b) / 2.0
    a_trans2 = (r_b + r_f) / 2.0

    v_i = np.sqrt(mu / r_i)
    v_trans1_a = np.sqrt(2 * mu / r_i - mu / a_trans1)
    v_trans1_b = np.sqrt(2 * mu / r_b - mu / a_trans1)
    v_trans2_b = np.sqrt(2 * mu / r_b - mu / a_trans2)
    v_trans2_c = np.sqrt(2 * mu / r_f - mu / a_trans2)
    v_f = np.sqrt(mu / r_f)

    dv_a = v_trans1_a - v_i
    dv_b = v_trans2_b - v_trans1_b
    dv_c = v_f - v_trans2_c

    t_trans1 = np.pi * np.sqrt(a_trans1**3 / mu)
    t_trans2 = np.pi * np.sqrt(a_trans2**3 / mu)

    return a_trans1, a_trans2, t_trans1, t_trans2, dv_a, dv_b, dv_c


# TODO: Example 6-3 p. 334
def guid_one_tangent(
    r_i: float, r_f: float, ta_trans_b, burn_at_periapsis: True, mu=MU_EARTH, deg=False
):
    """One-tangent burn transfer between two coplanar circular orbits.

    First burn is tangent to initial orbit; second burn at final orbit is not tangent.

    Parameters
    ----------
    r_i : float
        Radius of initial circular orbit [m].
    r_f : float
        Radius of final circular orbit [m].
    ta_trans_b : float
        True anomaly on transfer orbit at second burn [rad, or deg if
        `deg` is True].
    burn_at_periapsis : bool
        True if first burn is at periapsis of transfer orbit, False if
        at apoapsis.
    mu : float, optional
        Gravitational parameter of central body [m3/s2]. Default is `MU_EARTH`.
    deg : bool, optional
        If True, `ta_trans_b` is in degrees; otherwise radians. Default is False.

    Returns
    -------
    a_trans : float
        Semi-major axis of transfer orbit [m].
    t_trans : float
        Transfer time [s].
    dv_a : float
        Delta-v of first (tangential) burn; positive is prograde [m/s].
    dv_b : float
        Magnitude of second burn delta-v [m/s].

    References
    ----------
    Vallado 4e, Algorithm 38, p. 333.
    """

    R_inv = r_i / r_f

    if deg:
        ta_trans_b *= DEG_TO_RAD

    e_trans = (
        (R_inv - 1) / (np.cos(ta_trans_b) - R_inv)
        if burn_at_periapsis
        else (R_inv - 1) / (np.cos(ta_trans_b) + R_inv)
    )

    a_trans = r_i / (1 - e_trans) if burn_at_periapsis else r_i / (1 + e_trans)

    v_i = np.sqrt(mu / r_i)
    v_trans_a = np.sqrt(2 * mu / r_i - mu / a_trans)
    v_f = np.sqrt(mu / r_f)
    v_trans_b = np.sqrt(2 * mu / r_f - mu / a_trans)

    dv_a = v_trans_a - v_i

    # Phase angle!
    phi_trans_b = np.arctan2(
        e_trans * np.sin(ta_trans_b), 1 + e_trans * np.cos(ta_trans_b)
    )

    dv_b = np.sqrt(v_trans_b**2 + v_f**2 - 2 * v_trans_b * v_f * np.cos(phi_trans_b))

    # for t_trans
    E = E_from_ta(ta_trans_b, e_trans)

    k = 0  # Compeltely short
    E0 = 0.0 if burn_at_periapsis else np.pi
    M = E - e_trans * np.sin(E)
    M0 = E0 - e_trans * np.sin(E0)
    t_trans = (
        np.pi * np.sqrt(a_trans**3 / mu) * (2 * k * np.pi + (M - M0) % (2 * np.pi))
    )

    return a_trans, t_trans, dv_a, dv_b
