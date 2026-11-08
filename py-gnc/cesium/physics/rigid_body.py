"""Similar to AA 242A"""

from __future__ import annotations

import numpy as np

from ..math.quaternion import hamilton_product


def rigid_body_derivative(
    t: float,
    state: np.ndarray,
    mass: float,
    force_I: np.ndarray,
    I: np.ndarray = None,
    torque_B: np.ndarray = None,
):
    """Time derivative of a 3-DOF or 6-DOF rigid-body state.

    Parameters
    ----------
    t : float
        Current time [s]. Unused, but required by the integrator interface.
    state : np.ndarray (13,)
        `[r, v, q, w]`: inertial position [m], inertial velocity [m/s], scalar-first
        attitude quaternion `[w, x, y, z]`, and body-frame angular velocity [rad/s].
    mass : float
        Mass [kg].
    force_I : np.ndarray (3,)
        Total inertial-frame force [N].
    I : np.ndarray (3, 3) or None
        If using 6-DOF, body-frame inertia about the center of mass [kg*m2]. None disables attitude dynamics:
        both quaternion and angular velocity derivatives are zero.
    torque_B : np.ndarray (3,) or None
        If using 6-DOF, total body-frame torque about the center of mass [Nm].

    Returns
    -------
    np.ndarray (13,)
        State derivative `[v, a, q_dot, w_dot]` [m/s, m/s2, 1/s, rad/s2] if 6DOF or `[v, a]` [m/s, m/s2] if 3DOF

    References
    ----------
    Schaub, Equations 2.15, 3.111, and 4.34-4.35.
    """
    del t

    v = state[3:6]

    # Position derivative is velocity
    drdt = v

    # Velocity derivative is acceleration (Schaub 2.15)
    dvdt = np.asarray(force_I) / mass

    # Only evaluate attitude if desired
    if I is not None and torque_B is not None and len(state) > 6:
        q = state[6:10]
        w = state[10:13]
        # Quaternion kinematics (Schaub 3.111) and Euler dynamics (4.34-35).
        dqdt = 0.5 * hamilton_product(q, w)
        dwdt = np.linalg.solve(I, np.asarray(torque_B) - np.cross(w, I @ w))
        return np.hstack((drdt, dvdt, dqdt, dwdt))

    return np.hstack((drdt, dvdt))
