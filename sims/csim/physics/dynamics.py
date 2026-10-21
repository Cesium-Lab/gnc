"""
n-body
"""


import numpy as np
from ..world.constants import MU_EARTH, R_EARTH
from .gravity import grav_accel

def two_body(state: np.ndarray, mu=MU_EARTH, j2=None, r_cb=R_EARTH) -> np.ndarray:
    """Two body dynamics
    
    [x, y, z, vx, vy, vz] -> [vx, vy, vz, ax, ay, az]

    Parameters
    ----------
    state : np.ndarray
        State - `[x, y, z, vx, vy, vz]`
    mu : float, optional
        Gravitational parameter of central body, by default `MU_EARTH` [km3/s2]

    Returns
    -------
    x_dot : np.ndarray (6,)
        State derivative - `[vx, vy, vz, ax, ay, az]`
    """
    r,v = state[0:3], state[3:6]
    a = grav_accel(r,mu,j2,r_cb)
    state_dot = np.concatenate((v,a))
    
    return state_dot

def synodic_dynamics(state: np.ndarray, mu1: float, mu2: float, r1_bc: float, r2_bc: float, w_s: float) -> np.ndarray:
    """Circular restricted three-body problem (CR3BP). Constant `w_s` and circular orbits
    
    [x, y, z, vx, vy, vz] -> [vx, vy, vz, ax, ay, az]

    Parameters
    ----------
    state : np.ndarray
        State - `[x, y, z, vx, vy, vz]`
    mu1 : float
        Gravitational parameter of larger central body [km3/s2]
    mu2 : float
        Gravitational parameter of smaller central body [km3/s2]
    r1_bc : float
        Distance from larger central body to barycenter [km]
    r2_bc : float
        Distance from smaller central body to barycenter [km]
    w_s : float
        Angular velocity of system [rad/s]

    Returns
    -------
    state_dot : np.ndarray (6,)
        State derivative - `[vx, vy, vz, ax, ay, az]`
    """
    
    assert mu1 > mu2
    assert r1_bc < r2_bc
    
    assert np.isclose(mu1 * r1_bc, mu2 * r2_bc)
    assert np.isclose(w_s**2, (mu1 + mu2) / (r1_bc + r2_bc)**3)
    
    x, y, z, x_dot, y_dot, z_dot = state
    
    r_1_sat = np.sqrt((x + r1_bc)**2 + y**2 + z**2)
    r_2_sat = np.sqrt((x - r2_bc)**2 + y**2 + z**2)
    
    x_ddot = (
        -mu1 * (x + r1_bc) / (r_1_sat**3)
        -mu2 * (x - r2_bc) / (r_2_sat**3)
        + 2 * w_s * y_dot
        + w_s**2 * x)
    
    y_ddot = (
        -mu1 * y / (r_1_sat**3)
        -mu2 * y / (r_2_sat**3)
        - 2 * w_s * x_dot
        + w_s**2 * y)
    
    z_ddot = (
        -mu1 * z / (r_1_sat**3)
        -mu2 * z / (r_2_sat**3))
    
    state_dot = np.array([x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot])
    
    return state_dot