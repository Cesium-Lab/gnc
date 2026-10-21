from __future__ import annotations
import numpy as np

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
