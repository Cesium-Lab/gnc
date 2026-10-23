"""
n-body
"""
import numpy as np
from numpy.linalg import norm
from ..world.constants import MU_EARTH, R_EARTH, J2_EARTH
from ..orbital.coes import rv_to_coes, coes_to_rv, kepler_eq, E_from_ta, ta_from_E
from .gravity import grav_accel

def two_body(state: np.ndarray, mu=MU_EARTH, j2=None, r_cb=R_EARTH) -> np.ndarray:
    """Time derivative under two-body gravity, optionally with J2.

    Parameters
    ----------
    state : np.ndarray (6,)
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
    state_dot : np.ndarray (6,)
        State derivative, `[vx, vy, vz, ax, ay, az]` [m/s, m/s2].
    """
    r,v = state[0:3], state[3:6]
    a = grav_accel(r,mu,j2,r_cb)
    state_dot = np.concatenate((v,a))
    
    return state_dot

def synodic_dynamics(state: np.ndarray, mu1: float, mu2: float, r1_bc: float, r2_bc: float, w_s: float) -> np.ndarray:
    """Time derivative under circular restricted three-body problem.

    The state is expressed in the rotating synodic frame centered at the
    barycenter, with the larger body on the -x axis and the smaller body on the
    +x axis. The primaries are on circular orbits with constant rate `w_s`.

    Parameters
    ----------
    state : np.ndarray (6,)
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
    state_dot : np.ndarray (6,)
        State derivative in the synodic frame, `[vx, vy, vz, ax, ay, az]`
        [m/s, m/s2].
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
    
        
    