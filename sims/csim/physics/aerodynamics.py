"""
TODO:
Could also be `aero_forces.py`
- base drag
- skin friction drag
- other drags
- lift
"""
import numpy as np

from ..world import W_EARTH

def drag_v_rel(r: np.ndarray, v: np.ndarray, w_cb: np.ndarray | float = W_EARTH):
    """Velocity relative to the atmosphere. Assumes central body (Earth) rotates perfectly in the Z axis 
    (due to motion of the poles being so small). \\
    Vallado 4e p. 552

    Args:
        r (np.ndarray): Absolute position [m]
        v (np.ndarray): Absolute velocity [m/s]
        w_cb (np.ndarray | float, optional): Angular velocity of central body. If array, it is the vector. If single quantity, assumes perfect z-axis rotation.
            Defaults to W_EARTH.

    Returns:
        float: v_rel [m/s]
    """

    if isinstance(w_cb, (list, np.ndarray)):
        w = np.asarray(w_cb)
    else:
        w = np.array([0,0,w_cb])

    v_atmos = np.cross(w, r)

    return v - v_atmos

def drag_accel(v_rel: np.ndarray, rho: float, A: float, m: float, Cd: float = 2.2, *, BC: float = None):
    """Atmospheric drag. Can set A and m to 0 if ballistic coefficient is specified \\
    Vallado 8-28 4e p. 551

    Args:
        v_rel (np.ndarray): Velocity relative to the atmosphere (NOT the same as absolute velocity)
        A (float): Cross sectional area with respect to 
        m (float): Mass [kg]
        rho (float): Atmospheric density [kg/m3]
        Cd (float, optional): Coefficient of drag. Defaults to 2.2.
        BC (float, optional): If ballistic coefficient [kg/m2] is provided, will use this instead

    Returns:
        np.ndarray: Drag acceleration
    """

    if BC is None:
        BC = m / Cd / A

    v_norm = np.linalg.norm(v_rel)

    accel = -1/2/BC * rho * v_norm * v_rel

    return accel
