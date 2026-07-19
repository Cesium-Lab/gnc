"""Two-vehicle relative motion (RPO): ECI <--> Hill/RSW frame, and the
Clohessy-Wiltshire linearized relative-motion solution used by guidance.
"""
# ruff: noqa: E741

import numpy as np
from numpy.linalg import norm

from ..world import MU_EARTH


def mean_motion(a: float, mu: float = MU_EARTH):
    """Mean motion of a Keplerian orbit `n = sqrt(mu/a^3)`

    Args:
        a (float): Semi-major axis [m]
        mu (float, optional): Gravitational parameter of central body [m3/s2]. Defaults to MU_EARTH.

    Returns:
        float: Mean motion [rad/s]
    """
    return np.sqrt(mu / a**3)


def hill_dcm(r_target: np.ndarray, v_target: np.ndarray):
    """DCM from ECI to the target's Hill/RSW frame (Radial, along-track/transverse, cross-track). \\
    Vallado 4e p. 157

    Args:
        r_target (np.ndarray): Target position (ECI) [m]
        v_target (np.ndarray): Target velocity (ECI) [m/s]

    Returns:
        np.ndarray: DCM [3x3] such that `v_rsw = DCM @ v_eci`
    """
    R_hat = r_target / norm(r_target)
    h = np.cross(r_target, v_target)
    W_hat = h / norm(h)
    S_hat = np.cross(W_hat, R_hat)

    return np.array([R_hat, S_hat, W_hat])


def eci_relative_to_hill(r_target: np.ndarray, v_target: np.ndarray,
                          r_chaser: np.ndarray, v_chaser: np.ndarray):
    """Exact (nonlinear) relative position/velocity of a chaser w.r.t. a target,
    expressed in the target's rotating Hill/RSW frame.

    Args:
        r_target (np.ndarray): Target position (ECI) [m]
        v_target (np.ndarray): Target velocity (ECI) [m/s]
        r_chaser (np.ndarray): Chaser position (ECI) [m]
        v_chaser (np.ndarray): Chaser velocity (ECI) [m/s]

    Returns:
        np.ndarray: [6] relative state `[x,y,z,xdot,ydot,zdot]` (Hill frame) [m], [m/s]
    """
    DCM = hill_dcm(r_target, v_target)

    rho = DCM @ (r_chaser - r_target)

    # Hill frame rotates about W_hat at rate h/r^2 (exact for any orbit, not just circular)
    r_norm = norm(r_target)
    h_norm = norm(np.cross(r_target, v_target))
    omega = np.array([0, 0, h_norm / r_norm**2])

    rho_dot = DCM @ (v_chaser - v_target) - np.cross(omega, rho)

    return np.hstack((rho, rho_dot))


def cw_stm(t: float, n: float):
    """Clohessy-Wiltshire state-transition matrix for linearized relative motion about a
    circular reference orbit. `rho(t) = Phi(t,n) @ rho0` where `rho = [x,y,z,xdot,ydot,zdot]`
    is expressed in the target's Hill/RSW frame (x radial, y along-track, z cross-track). \\
    Curtis 3e Eq. 7.53-56 p. 397

    Args:
        t (float): Time since `rho0` [s]
        n (float): Target's orbital mean motion [rad/s]

    Returns:
        np.ndarray: State-transition matrix `Phi` [6x6]
    """
    nt = n * t
    s, c = np.sin(nt), np.cos(nt)

    Phi_rr = np.array([
        [4 - 3*c, 0, 0],
        [6*(s - nt), 1, 0],
        [0, 0, c],
    ])
    Phi_rv = np.array([
        [s/n, 2/n*(1 - c), 0],
        [-2/n*(1 - c), (4*s - 3*nt)/n, 0],
        [0, 0, s/n],
    ])
    Phi_vr = np.array([
        [3*n*s, 0, 0],
        [-6*n*(1 - c), 0, 0],
        [0, 0, -n*s],
    ])
    Phi_vv = np.array([
        [c, 2*s, 0],
        [-2*s, 4*c - 3, 0],
        [0, 0, c],
    ])

    return np.block([
        [Phi_rr, Phi_rv],
        [Phi_vr, Phi_vv],
    ])
