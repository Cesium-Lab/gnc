"""
n-body
"""

import numpy as np
from scipy.integrate import solve_ivp

from ..world.constants import MU_EARTH, R_EARTH
from .gravity import grav_accel


def two_body(rv_I: np.ndarray, mu=MU_EARTH, j2=None, r_cb=R_EARTH) -> np.ndarray:
    """Time derivative under two-body gravity, optionally with J2.

    Parameters
    ----------
    rv_I : np.ndarray (6,)
        Satellite position and velocity in the inertial frame (e.g. ECI),
        `[x, y, z, vx, vy, vz]` [m, m/s].
    mu : float, optional
        Gravitational parameter of the central body [m3/s2]. Default is `MU_EARTH`.
    j2 : float, optional
        J2 value for perturbation. If None, J2 is not applied.
        Default is None.
    r_cb : float, optional
        Equatorial radius of the central body, used for J2 if it is enabled [m]. Default is `R_EARTH`.

    Returns
    -------
    rv_dot_I : np.ndarray (6,)
        State derivative, `[vx, vy, vz, ax, ay, az]` [m/s, m/s2].
    """
    r, v = rv_I[0:3], rv_I[3:6]
    a = grav_accel(r, mu, j2, r_cb)

    rv_dot_I = np.zeros_like(rv_I)
    rv_dot_I[0:3] = v
    rv_dot_I[3:6] = a

    return rv_dot_I


def propagate_two_body(
    rv0_I: np.ndarray,
    t_eval_s: np.ndarray,
    mu=MU_EARTH,
    j2=None,
    r_cb=R_EARTH,
    rtol=1e-9,
    atol=1e-12,
) -> np.ndarray:
    """Integrate `two_body` from an initial state over a time grid.

    `rv0_I` is the state at `t_eval_s[0]`. A decreasing grid (e.g.
    `np.linspace(0, -T, n)`) propagates backwards in time; positions and
    velocities both stay physical, so no velocity flipping is needed.

    Parameters
    ----------
    rv0_I : np.ndarray (6,)
        Initial position and velocity in an inertial frame [m, m/s].
    t_eval_s : np.ndarray (N,)
        Monotonic output times [s]; the first entry is the epoch of `rv0_I`.
    mu : float, optional
        Gravitational parameter of the central body [m3/s2]. Default is `MU_EARTH`.
    j2 : float, optional
        J2 coefficient. If None, J2 is not applied. Default is None.
    r_cb : float, optional
        Equatorial radius of the central body, used for J2 [m]. Default is `R_EARTH`.
    rtol, atol : float, optional
        `solve_ivp` tolerances. Defaults are 1e-9 and 1e-12.

    Returns
    -------
    rv_I : np.ndarray (N, 6)
        States at `t_eval_s` [m, m/s].

    Raises
    ------
    RuntimeError
        If the integrator fails, does not reach every requested time, or
        returns non-finite states.
    """
    return _propagate(
        lambda _, s: two_body(s, mu, j2, r_cb),
        rv0_I,
        t_eval_s,
        rtol=rtol,
        atol=atol,
        method="RK45",
        label="Two-body",
    )


def _propagate(f, rv0, t_eval_s, *, rtol, atol, method, label) -> np.ndarray:
    """`solve_ivp` over `t_eval_s` from `rv0`; raises if it fails or goes non-finite."""
    t_eval_s = np.asarray(t_eval_s, dtype=float)
    sol = solve_ivp(
        f,
        (t_eval_s[0], t_eval_s[-1]),
        np.asarray(rv0, dtype=float),
        t_eval=t_eval_s,
        rtol=rtol,
        atol=atol,
        method=method,
    )
    if (
        not sol.success
        or sol.y.shape[1] != t_eval_s.size
        or not np.all(np.isfinite(sol.y))
    ):
        raise RuntimeError(f"{label} propagation failed: {sol.message}")

    return sol.y.T


def synodic_dynamics(
    rv_S: np.ndarray, mu1: float, mu2: float, r1_bc: float, r2_bc: float, w_s: float
) -> np.ndarray:
    """Time derivative under circular restricted three-body problem.

    The state is expressed in the rotating synodic frame centered at the
    barycenter, with the larger body on the -x axis and the smaller body on the
    +x axis. The primaries are on circular orbits with constant rate `w_s`.

    Parameters
    ----------
    rv_S : np.ndarray (6,)
        Satellite position and velocity in the synodic frame,
        `[x, y, z, vx, vy, vz]` [m, m/s].
    mu1 : float
        Gravitational parameter of larger central body [m3/s2].
    mu2 : float
        Gravitational parameter of smaller central body [m3/s2].
    r1_bc : float
        Distance from larger central body to barycenter [m].
    r2_bc : float
        Distance from smaller central body to barycenter [m].
    w_s : float
        Angular velocity of the synodic frame [rad/s].

    Returns
    -------
    rv_dot_S : np.ndarray (6,)
        State derivative in the synodic frame, `[vx, vy, vz, ax, ay, az]`
        [m/s, m/s2].
    """

    assert mu1 > mu2
    assert r1_bc < r2_bc

    # atol=0: w_s**2 ~ 1e-12, far below np.isclose's default atol of 1e-8
    assert np.isclose(mu1 * r1_bc, mu2 * r2_bc, rtol=1e-9, atol=0)
    assert np.isclose(w_s**2, (mu1 + mu2) / (r1_bc + r2_bc) ** 3, rtol=1e-9, atol=0)

    x, y, z, x_dot, y_dot, z_dot = rv_S

    r_1_sat = np.sqrt((x + r1_bc) ** 2 + y**2 + z**2)
    r_2_sat = np.sqrt((x - r2_bc) ** 2 + y**2 + z**2)

    x_ddot = (
        -mu1 * (x + r1_bc) / (r_1_sat**3)
        - mu2 * (x - r2_bc) / (r_2_sat**3)
        + 2 * w_s * y_dot
        + w_s**2 * x
    )

    y_ddot = (
        -mu1 * y / (r_1_sat**3) - mu2 * y / (r_2_sat**3) - 2 * w_s * x_dot + w_s**2 * y
    )

    z_ddot = -mu1 * z / (r_1_sat**3) - mu2 * z / (r_2_sat**3)

    rv_dot_S = np.array([x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot])

    return rv_dot_S


def propagate_synodic(
    rv0_S: np.ndarray,
    t_eval_s: np.ndarray,
    mu1: float,
    mu2: float,
    r1_bc: float,
    r2_bc: float,
    w_s: float,
    rtol=1e-12,
    atol=1e-6,
) -> np.ndarray:
    """Integrate `synodic_dynamics` (CR3BP) from an initial state over a time grid.

    `rv0_S` is the state at `t_eval_s[0]`; a decreasing grid propagates backwards.
    Uses DOP853 with tight defaults, since three-body trajectories are sensitive to
    integration error (especially on close approaches to either primary).

    Parameters
    ----------
    rv0_S : np.ndarray (6,)
        Initial position and velocity in the synodic frame [m, m/s].
    t_eval_s : np.ndarray (N,)
        Monotonic output times [s]; the first entry is the epoch of `rv0_S`.
    mu1, mu2, r1_bc, r2_bc, w_s : float
        System parameters, as in `synodic_dynamics`.
    rtol, atol : float, optional
        `solve_ivp` tolerances. Defaults are 1e-12 and 1e-6.

    Returns
    -------
    rv_S : np.ndarray (N, 6)
        States at `t_eval_s` [m, m/s].

    Raises
    ------
    RuntimeError
        If the integrator fails, does not reach every requested time, or
        returns non-finite states.
    """
    return _propagate(
        lambda _, s: synodic_dynamics(s, mu1, mu2, r1_bc, r2_bc, w_s),
        rv0_S,
        t_eval_s,
        rtol=rtol,
        atol=atol,
        method="DOP853",
        label="Synodic",
    )
