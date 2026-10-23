from __future__ import annotations
import numpy as np
from .transformations import PCI_to_RSW

# AA 279B
def calc_Phi(t: float, n: float):
    """Phi for HCW circular orbit"""
    nt = n * t
    c,s = np.cos(nt), np.sin(nt)
    Phi_rr = np.array([
        [4 - 3*c, 0, 0],
        [6 * (s - nt), 1, 0],
        [0, 0, c]])
    Phi_rv = np.array([
        [1/n * s, 2/n * (1 - c), 0],
        [2/n * (c - 1), 4/n * s - 3/n * nt, 0],
        [0, 0, s/n]
        ])
    Phi_vr = np.array([
        [3*n*s, 0, 0],
        [6*n*(c - 1), 0, 0],
        [0, 0, -n*s]])
    Phi_vv = np.array([
        [c, 2*s, 0],
        [-2*s, 4*c - 3, 0],
        [0, 0, c]
        ])
    return Phi_rr, Phi_rv, Phi_vr, Phi_vv

def rv_hcw_3d(t: float, n: float, r0: np.ndarray, v0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """3D HCW rv from initial conditions in a circular orbit
    Vallado 4e Algorithm 48 p. 396-7 \\
        
    - x: radial outward
    - y: along-track (direction of orbital motion)
    - z: cross-track (orbital angular momentum direction)
    
    Parameters
    ----------
    t : float
        Time [s]
    n : float
        Mean motion [rad/s]
    r0 : np.ndarray (3,)
        Initial position [m]
    v0 : np.ndarray (3,)
        Initial velocity [m/s]

    Returns
    -------
    r : np.ndarray (3,)
        Position [m]
    v : np.ndarray (3,)
        Velocity [m/s]
    """
    
    Phi_rr, Phi_rv, Phi_vr, Phi_vv = calc_Phi(t,n)
    return (Phi_rr @ r0 + Phi_rv @ v0,
            Phi_vr @ r0 + Phi_vv @ v0)
    
def rv_hcw_2d(t: float, n: float, r0: np.ndarray, v0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2D HCW rv from initial conditions in a circular orbit
    Vallado 4e Algorithm 48 p. 396-7 \\
    Uses `rv_hcw_3d()` (see docstring)
    """
    r,v = rv_hcw_3d(t, n, np.pad(r0, (0,1)), np.pad(v0, (0,1)))
    return r[:2], v[:2]

# TODO: integrate with closed-form solution
def hcw_dynamics(state: np.ndarray, a_rsw: np.ndarray, n: float):
    """HCW dynamics (circular orbit)
    Vallado 4e Equation 6-61 p. 393

    Parameters
    ----------
    state : np.ndarray
        state in RSW coordinates [m, m/s]
    a : np.ndarray
        acceleration due to force in RSW coordinates [N]
    n : float
        mean motion [1/s]
    """
    r,v = state[0:3], state[3:6]
    
    a = a_rsw + np.array([
        2*n*v[1] + 3*n*n*r[0],
        -2*n*v[0],
        -n*n*r[2]
    ])
    
    return np.concatenate((v,a))

def hcw_RSW_to_PCI(rv_target_PCI: np.ndarray, rv_chaser_RSW: np.ndarray, mu: float):
    """Convert relative hcw state in RSW to absolute chaser PCI state.
    Simplified version of Vallado 4e Algorithm 49 p. 414

    Parameters
    ----------
    rv_target_PCI : np.ndarray (6,)
        Target position and velocity in PCI [m, m/s]
    rv_chaser_RSW : np.ndarray (6,)
        Chaser relative position and velocity in RSW [m, m/s]

    Returns
    -------
    rv_chaser_PCI : np.ndarray (6,)
        Absolute chaser state in PCI [m, m/s]
    """
    r_t_PCI, v_t_PCI = rv_target_PCI[0:3], rv_target_PCI[3:6]
    
    dcm_RSW_wrt_PCI = PCI_to_RSW(rv_target_PCI)
    r_rel_RSW, v_rel_RSW = rv_chaser_RSW[:3], rv_chaser_RSW[3:]
    
    
    # Angular velocity of RSW frame, expressed in PCI
    omega_RSW_PCI_PCI = np.cross(r_t_PCI, v_t_PCI) / np.linalg.norm(r_t_PCI)**2

    # Relative position and velocity in PCI
    r_rel_PCI = dcm_RSW_wrt_PCI.T @ r_rel_RSW
    v_rel_PCI = dcm_RSW_wrt_PCI.T @ v_rel_RSW + np.cross(omega_RSW_PCI_PCI, r_rel_PCI)

    # Absolute chaser state
    r_chaser_PCI = r_t_PCI + r_rel_PCI
    v_chaser_PCI = v_t_PCI + v_rel_PCI

    return np.concatenate((r_chaser_PCI, v_chaser_PCI))
