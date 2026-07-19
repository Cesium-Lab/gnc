"""RPO maneuver planning against a linear relative-motion model (Hill/Clohessy-Wiltshire).

This module never touches the nonlinear truth propagator or truth state directly --
it only ever sees a navigation estimate (e.g. the output of the EKF in
`estimation.filtering.ekf`) and a state-transition matrix for the planning model
(e.g. `sims.csim.physics.relative_motion.cw_stm`), so guidance stays agnostic to
whatever fidelity truth is actually propagated at.
"""

import numpy as np


def two_impulse_rendezvous(rho0: np.ndarray, Phi: np.ndarray, rho_f: np.ndarray | None = None):
    """Classic 2-impulse targeting problem for a linear relative-motion model: find the
    Delta-v's that take a chaser from its current relative state to a desired relative
    position `rho_f` (with zero relative velocity on arrival), using `Phi` as the
    state-transition matrix for the chosen time-of-flight. \\
    Curtis 3e Section 7.6 p. 400 (two-impulse rendezvous)

    Args:
        rho0 (np.ndarray): Current relative state estimate `[x,y,z,xdot,ydot,zdot]` [m],[m/s]
        Phi (np.ndarray): [6x6] state-transition matrix for the time-of-flight of the maneuver
            (e.g. `cw_stm(t_go, n)`)
        rho_f (np.ndarray, optional): Desired relative position [m] at arrival. Defaults to the origin.

    Returns:
        tuple: `(dv1, dv2)`, each `np.ndarray[3]` -- Delta-v [m/s] at departure and arrival
    """
    if rho_f is None:
        rho_f = np.zeros(3)

    Phi_rr, Phi_rv = Phi[:3, :3], Phi[:3, 3:]
    Phi_vr, Phi_vv = Phi[3:, :3], Phi[3:, 3:]

    r0, v0 = rho0[:3], rho0[3:]

    v0_needed = np.linalg.solve(Phi_rv, rho_f - Phi_rr @ r0)
    dv1 = v0_needed - v0

    vf_arrival = Phi_vr @ r0 + Phi_vv @ v0_needed
    dv2 = -vf_arrival

    return dv1, dv2
