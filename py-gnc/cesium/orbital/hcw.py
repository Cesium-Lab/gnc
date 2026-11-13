from __future__ import annotations

import numpy as np

from .transformations import calc_dcm_rsw_wrt_pci


# AA 279B
def calc_Phi(t: float, n: float):
    """State transition matrix blocks for the HCW equations (circular target orbit).

    Frame is the target's RSW frame: x radial outward, y along-track, z cross-track.

    Parameters
    ----------
    t : float
        Time since the initial relative state [s].
    n : float
        Mean motion of the target's circular orbit [rad/s].

    Returns
    -------
    Phi_rr : np.ndarray (3, 3)
        Sensitivity of `r` to initial `r0`.
    Phi_rv : np.ndarray (3, 3)
        Sensitivity of `r` to initial `v0` [s].
    Phi_vr : np.ndarray (3, 3)
        Sensitivity of `v` to initial `r0` [1/s].
    Phi_vv : np.ndarray (3, 3)
        Sensitivity of `v` to initial `v0`.
    """
    nt = n * t
    c, s = np.cos(nt), np.sin(nt)
    Phi_rr = np.array([[4 - 3 * c, 0, 0], [6 * (s - nt), 1, 0], [0, 0, c]])
    Phi_rv = np.array(
        [
            [1 / n * s, 2 / n * (1 - c), 0],
            [2 / n * (c - 1), 4 / n * s - 3 / n * nt, 0],
            [0, 0, s / n],
        ]
    )
    Phi_vr = np.array([[3 * n * s, 0, 0], [6 * n * (c - 1), 0, 0], [0, 0, -n * s]])
    Phi_vv = np.array([[c, 2 * s, 0], [-2 * s, 4 * c - 3, 0], [0, 0, c]])
    return Phi_rr, Phi_rv, Phi_vr, Phi_vv


def rv_hcw_3d(
    t: float, n: float, r0: np.ndarray, v0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a 3D relative state with the closed-form HCW solution.

    Assumes the target is in a circular orbit.

    Frame is the target's RSW frame: x radial outward, y along-track, z cross-track.

    Parameters
    ----------
    t : float
        Time since the initial state [s].
    n : float
        Mean motion of the target's circular orbit [rad/s].
    r0 : np.ndarray (3,)
        Initial chaser position relative to the target, in RSW [m].
    v0 : np.ndarray (3,)
        Initial chaser velocity relative to the target, in RSW [m/s].

    Returns
    -------
    r : np.ndarray (3,)
        Chaser position relative to the target at `t`, in RSW [m].
    v : np.ndarray (3,)
        Chaser velocity relative to the target at `t`, in RSW [m/s].

    References
    ----------
    Vallado 4e, Algorithm 48 (Hill's Equations), p. 396-397.
    """

    Phi_rr, Phi_rv, Phi_vr, Phi_vv = calc_Phi(t, n)
    return (Phi_rr @ r0 + Phi_rv @ v0, Phi_vr @ r0 + Phi_vv @ v0)


def rv_hcw_2d(
    t: float, n: float, r0: np.ndarray, v0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate an in-plane relative state with the closed-form HCW solution.

    Wrapper around `rv_hcw_3d` with the cross-track components set to zero.

    Parameters
    ----------
    t : float
        Time since the initial relative state [s].
    n : float
        Mean motion of the target's circular orbit [rad/s].
    r0 : np.ndarray (2,)
        Initial chaser position relative to the target, `[radial, along-track]` [m].
    v0 : np.ndarray (2,)
        Initial chaser velocity relative to the target, `[radial, along-track]` [m/s].

    Returns
    -------
    r : np.ndarray (2,)
        Relative position at `t`, `[radial, along-track]` [m].
    v : np.ndarray (2,)
        Relative velocity at `t`, `[radial, along-track]` [m/s].

    References
    ----------
    Vallado 4e, Algorithm 48 (Hill's Equations), p. 396-397.
    """
    r, v = rv_hcw_3d(t, n, np.pad(r0, (0, 1)), np.pad(v0, (0, 1)))
    return r[:2], v[:2]


# TODO: integrate with closed-form solution
def hcw_dynamics(state: np.ndarray, a_rsw: np.ndarray, n: float):
    """Time derivative of the relative state under HCW dynamics (circular target orbit).

    Parameters
    ----------
    state : np.ndarray (6,)
        Chaser state relative to the target in RSW, `[x, y, z, vx, vy, vz]` [m, m/s].
    a_rsw : np.ndarray (3,)
        External (e.g. thrust) acceleration on the chaser in RSW [m/s2].
    n : float
        Mean motion of the target's circular orbit [rad/s].

    Returns
    -------
    state_dot : np.ndarray (6,)
        Relative state derivative, `[vx, vy, vz, ax, ay, az]` [m/s, m/s2].

    References
    ----------
    Vallado 4e, Equation 6-61, p. 393.
    """
    r, v = state[0:3], state[3:6]

    a = a_rsw + np.array(
        [2 * n * v[1] + 3 * n * n * r[0], -2 * n * v[0], -n * n * r[2]]
    )

    return np.concatenate((v, a))


def hcw_rsw_to_pci(rv_target_pci: np.ndarray, rv_chaser_rsw: np.ndarray) -> np.ndarray:
    """Convert a chaser's relative RSW state to its absolute PCI state.

    Inverse of `hcw_pci_to_rsw`. The RSW frame rotates with the target at
    `w = h / r^2`, so the relative velocity
    picks up a `w x r_rel` term.

    Parameters
    ----------
    rv_target_pci : np.ndarray (6,)
        Target position and velocity in PCI (e.g. MCI) [m, m/s].
    rv_chaser_rsw : np.ndarray (6,)
        Chaser position and velocity relative to the target, in the target's
        rotating RSW frame [m, m/s].

    Returns
    -------
    rv_chaser_pci : np.ndarray (6,)
        Absolute chaser position and velocity in PCI [m, m/s].

    References
    ----------
    Simplified version of Vallado 4e, Algorithm 49 (HillEQCM to ECI), p. 414.
    """
    r_t, v_t = rv_target_pci[0:3], rv_target_pci[3:6]
    dcm_rsw_wrt_pci = calc_dcm_rsw_wrt_pci(rv_target_pci)
    w_rsw_pci = np.cross(r_t, v_t) / np.linalg.norm(r_t) ** 2  # RSW frame rate, in PCI

    r_rel_pci = dcm_rsw_wrt_pci.T @ rv_chaser_rsw[0:3]
    v_rel_pci = dcm_rsw_wrt_pci.T @ rv_chaser_rsw[3:6] + np.cross(w_rsw_pci, r_rel_pci)

    return np.concatenate((r_t + r_rel_pci, v_t + v_rel_pci))


def hcw_pci_to_rsw(rv_target_pci: np.ndarray, rv_chaser_pci: np.ndarray) -> np.ndarray:
    """Chaser state relative to the target, in the target's rotating RSW frame.

    Inverse of `hcw_rsw_to_pci`; the result is the HCW initial state for
    `plan_hcw_two_impulse`.

    Parameters
    ----------
    rv_target_pci : np.ndarray (6,)
        Target position and velocity in PCI (e.g. MCI) [m, m/s].
    rv_chaser_pci : np.ndarray (6,)
        Chaser position and velocity in the same PCI frame [m, m/s].

    Returns
    -------
    rv_chaser_rsw : np.ndarray (6,)
        Chaser position and velocity relative to the target, in the target's
        rotating RSW frame [m, m/s].

    References
    ----------
    Simplified version of Vallado 4e, Algorithm 50 (ECI to HillEQCM), p. 415.
    """
    r_t, v_t = rv_target_pci[0:3], rv_target_pci[3:6]
    dcm_rsw_wrt_pci = calc_dcm_rsw_wrt_pci(rv_target_pci)
    w_rsw_pci = np.cross(r_t, v_t) / np.linalg.norm(r_t) ** 2  # RSW frame rate, in PCI

    r_rel_pci = rv_chaser_pci[0:3] - r_t
    v_rel_pci = rv_chaser_pci[3:6] - v_t - np.cross(w_rsw_pci, r_rel_pci)

    return np.concatenate((dcm_rsw_wrt_pci @ r_rel_pci, dcm_rsw_wrt_pci @ v_rel_pci))
