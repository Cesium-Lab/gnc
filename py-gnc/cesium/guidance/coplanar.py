import numpy as np

from ..constants import DEG_TO_RAD
from ..orbital.coes import E_from_ta, vis_viva
from ..world.constants import MU_EARTH


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
    Vallado 4e, Algorithm 36 (Hohmann Transfer), p. 325.
    """
    a_trans = (r_i + r_f) / 2.0

    v_i = np.sqrt(mu / r_i)
    v_trans_a = vis_viva(r_i, a_trans, mu)
    v_f = np.sqrt(mu / r_f)
    v_trans_b = vis_viva(r_f, a_trans, mu)

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
    Vallado 4e, Algorithm 37 (Bi-elliptic Transfer), p. 326.
    """
    a_trans1 = (r_i + r_b) / 2.0
    a_trans2 = (r_b + r_f) / 2.0

    v_i = np.sqrt(mu / r_i)
    v_trans1_a = vis_viva(r_i, a_trans1, mu)
    v_trans1_b = vis_viva(r_b, a_trans1, mu)
    v_trans2_b = vis_viva(r_b, a_trans2, mu)
    v_trans2_c = vis_viva(r_f, a_trans2, mu)
    v_f = np.sqrt(mu / r_f)

    dv_a = v_trans1_a - v_i
    dv_b = v_trans2_b - v_trans1_b
    dv_c = v_f - v_trans2_c

    t_trans1 = np.pi * np.sqrt(a_trans1**3 / mu)
    t_trans2 = np.pi * np.sqrt(a_trans2**3 / mu)

    return a_trans1, a_trans2, t_trans1, t_trans2, dv_a, dv_b, dv_c


# TODO: Example 6-3 p. 334
def guid_one_tangent(
    r_i: float,
    r_f: float,
    ta_trans_b: float,
    burn_at_periapsis=True,
    mu=MU_EARTH,
    deg=False,
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
    burn_at_periapsis : bool, optional
        True (default) if first burn is at periapsis of transfer orbit, False
        if at apoapsis.
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
    Vallado 4e, Algorithm 38 (One-Tangent Burn), p. 333.
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
    v_trans_a = vis_viva(r_i, a_trans, mu)
    v_f = np.sqrt(mu / r_f)
    v_trans_b = vis_viva(r_f, a_trans, mu)

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
    # Kepler time of flight: sqrt(a^3/mu) * (2 k pi + M - M0) (Vallado Eq. 2-66)
    t_trans = np.sqrt(a_trans**3 / mu) * (2 * k * np.pi + (M - M0) % (2 * np.pi))

    return a_trans, t_trans, dv_a, dv_b


# TODO: test even though I think this looks rather correct


def circularize(a: float, e: float, burn_at_periapsis: bool = True, mu=MU_EARTH):
    """Tangential burn at an apsis that circularizes the orbit at that radius.

    Parameters
    ----------
    a : float
        Semi-major axis of the current orbit [m].
    e : float
        Eccentricity of the current orbit, `0 <= e < 1`.
    burn_at_periapsis : bool, optional
        True (default) to burn at periapsis (circle of radius `a (1 - e)`),
        False to burn at apoapsis (circle of radius `a (1 + e)`).
    mu : float, optional
        Gravitational parameter of central body [m3/s2]. Default is `MU_EARTH`.

    Returns
    -------
    dv : float
        Delta-v of the tangential burn; positive is prograde [m/s].

    References
    ----------
    Vallado 4e, Equation 1-31 (vis-viva), p. 28.
    """
    r_burn = a * (1 - e) if burn_at_periapsis else a * (1 + e)
    v_burn = vis_viva(r_burn, a, mu)
    v_circ = np.sqrt(mu / r_burn)

    return v_circ - v_burn


def guid_coaxial_hohmann(
    r_p1: float,
    r_a1: float,
    r_p2: float,
    r_a2: float,
    depart_at_periapsis: bool = True,
    arrive_at_periapsis: bool = True,
    mu=MU_EARTH,
):
    """Tangential apse-to-apse transfer between coaxial elliptical orbits.

    Burn 1 at an apsis of orbit 1, coast half of the transfer ellipse, burn 2
    at an apsis of orbit 2. Reduces to `guid_hohmann` for two circular orbits
    and handles elliptic-to-circular shape corrections.

    The departure apsis (orbit 1) and arrival apsis (orbit 2) are chosen
    independently, but the caller must pick a pair that lies 180 deg apart in
    inertial space; the transfer always covers half an ellipse. For example,
    if both periapsides point the same way, periapsis 1 pairs with apoapsis 2;
    if they point opposite ways, periapsis 1 pairs with periapsis 2. Either
    choice is valid for a circular orbit.

    Parameters
    ----------
    r_p1, r_a1 : float
        Periapsis and apoapsis radii of the initial orbit [m].
    r_p2, r_a2 : float
        Periapsis and apoapsis radii of the final orbit [m].
    depart_at_periapsis : bool, optional
        True (default) to make the first burn at periapsis of orbit 1, False
        for apoapsis.
    arrive_at_periapsis : bool, optional
        True (default) to make the second burn at periapsis of orbit 2, False
        for apoapsis.
    mu : float, optional
        Gravitational parameter of central body [m3/s2]. Default is `MU_EARTH`.

    Returns
    -------
    a_trans : float
        Semi-major axis of transfer ellipse [m].
    t_trans : float
        Transfer time (half transfer-orbit period) [s].
    dv_a : float
        Delta-v of first burn, at orbit 1; positive is prograde [m/s].
    dv_b : float
        Delta-v of second burn, at orbit 2; positive is prograde [m/s].

    References
    ----------
    Curtis, Orbital Mechanics for Engineering Students, Ch. 6 (Hohmann transfer
    between coaxial elliptical orbits).
    """
    if r_p1 > r_a1 or r_p2 > r_a2:
        raise ValueError("Periapsis radius must not exceed apoapsis radius")

    r_1 = r_p1 if depart_at_periapsis else r_a1
    r_2 = r_p2 if arrive_at_periapsis else r_a2

    a_1 = (r_p1 + r_a1) / 2
    a_2 = (r_p2 + r_a2) / 2
    a_trans = (r_1 + r_2) / 2

    dv_a = vis_viva(r_1, a_trans, mu) - vis_viva(r_1, a_1, mu)
    dv_b = vis_viva(r_2, a_2, mu) - vis_viva(r_2, a_trans, mu)
    t_trans = np.pi * np.sqrt(a_trans**3 / mu)

    return float(a_trans), float(t_trans), float(dv_a), float(dv_b)


def guid_phasing_same_orbit(
    a_target: float,
    d_ta: float,
    mu=MU_EARTH,
    k_target: int = 1,
    k_chaser: int = 1,
    deg=True,
):
    """Vallado Algorithm 44. Negative dv means retrograde"""
    w_target = np.sqrt(mu / a_target**3)
    if deg:
        d_ta *= DEG_TO_RAD

    t_phase = (2 * np.pi * k_target + d_ta) / w_target

    a_phase = np.cbrt(mu * (t_phase / (2 * np.pi * k_chaser)) ** 2)

    dv = 2 * (np.sqrt(2 * mu / a_target - mu / a_phase) - np.sqrt(mu / a_target))

    return float(dv), float(t_phase), float(a_phase)


def guid_phasing_different_orbits(
    a_chaser: float, a_target: float, d_ta: float, mu=MU_EARTH, k: int = 1, deg=True
):
    """Vallado Algorithm 45."""
    w_target = np.sqrt(mu / a_target**3)
    w_chaser = np.sqrt(mu / a_chaser**3)

    a_trans = (a_target + a_chaser) / 2.0
    t_trans = np.pi * np.sqrt(a_trans**3 / mu)

    if deg:
        d_ta *= DEG_TO_RAD

    alpha_L = w_target * t_trans
    ta = alpha_L - np.pi
    t_wait = (ta - d_ta + 2 * np.pi * k) / (w_chaser - w_target)

    dv = np.abs(
        np.sqrt(2 * mu / a_chaser - mu / a_trans) - np.sqrt(mu / a_chaser)
    ) + np.abs(np.sqrt(2 * mu / a_target - mu / a_trans) - np.sqrt(mu / a_target))

    return float(dv), float(t_trans), float(t_wait), float(a_trans)
