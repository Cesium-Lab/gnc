"""Similar to AA 242A"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from ..math.quaternion import hamilton_product


@dataclass
class RigidBodyParams:
    """Mass properties and applied loads for `rigid_body_derivative`.

    Attributes
    ----------
    mass_kg : float
        Mass of the body [kg].
    I : np.ndarray (3, 3) or None
        Moment of inertia tensor about the center of mass, in the body frame
        [kg*m2]. If None, attitude rates are held constant.
    force_N : np.ndarray (3,)
        Total external force on the body, in the inertial frame [N].
    torque_Nm : np.ndarray (3,)
        Total external torque about the center of mass, in the body frame [N*m].
    """

    mass_kg: float
    I: np.ndarray
    force_N: np.ndarray
    torque_Nm: np.ndarray


def rigid_body_derivative(t: float, state: np.ndarray, params: RigidBodyParams):
    """Time derivative of a 6-DOF rigid-body state.

    Parameters
    ----------
    t : float
        Current time [s]. Unused, but required by the integrator interface.
    state : np.ndarray (13,)
        `[r, v, q, w]`: inertial position [m], inertial velocity [m/s], scalar-first
        attitude quaternion `[w, x, y, z]`, and body-frame angular velocity [rad/s].
    params : RigidBodyParams
        Mass properties and applied force and torque.

    Returns
    -------
    np.ndarray (13,)
        State derivative `[v, a, q_dot, w_dot]` [m/s, m/s2, 1/s, rad/s2].

    References
    ----------
    Schaub, Equations 2.15, 3.111, and 4.34-4.35.
    """
    del t

    v = state[3:6]
    q = state[6:10]
    w = state[10:13]

    # Position derivative is velocity
    drdt = v

    # Velocity derivative is acceleration (Schaub 2.15)
    dvdt = np.asarray(params.force_N) / params.mass_kg

    # Quaternion derivative is based on hamilton product (Schaub 3.111)
    # print(q)
    # print(w)
    dqdt = 0.5 * hamilton_product(q, w)
    # print(dqdt)
    # Angular derivative based on (Schaub 4.34-35)

    I = params.I
    if I is not None:
        I_inv = np.linalg.inv(I)  # TODO: maybe precalc this
        torque = np.asarray(params.torque_Nm)

        # τ = parameters.torque_body
        dwdt = I_inv @ (torque - np.cross(w, I @ w))
    else:
        dwdt = np.zeros(3)

    return np.hstack((drdt, dvdt, dqdt, dwdt))
