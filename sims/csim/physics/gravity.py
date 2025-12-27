"""
TODO:
- Normal J1 gravity
- J2, J3 even
- gravity gradient
"""
# ruff: noqa: E741

import numpy as np
from numpy.linalg import norm
from ..world import MU_EARTH, R_EARTH, J2_EARTH

def grav_accel(r: np.ndarray, mu: float = MU_EARTH, *, use_j2 = True, r_cb: float = R_EARTH):
    """Gravity acceleration \\
    Vallado 4e Equation 1-14 p. 23 np.array([axy*x, axy*y, az*z]) \\
    
    Could not for the life of me find the J2 formula in Vallado, but found in [Poliastro docs](https://docs.poliastro.space/en/stable/autoapi/poliastro/core/perturbations/index.html) and in [this paper](https://ntrs.nasa.gov/api/citations/20040031507/downloads/20040031507.pdf)


    Args:
        r (np.ndarray): Position from center of center body [m/s]
        mu (float, optional): Gravitational parameter [m3/s2] Defaults to MU_EARTH.
        use_j2 (bool, optional): Whether to use J2. Defaults to True.
        r_cb (float, optional): (FOR J2) Radius of central body [m]. Defaults to R_EARTH.

    Returns:
        np.ndarray: Gravitational acceleration [m/s2]
    """

    # Prevents singularities (also np.inf * 0 == np.nan)
    if any(r == np.inf):
        return np.zeros(3)
    
    r_norm = norm(r)

    # No radius
    if abs(r_norm) < 1e-6:
        return np.zeros(3)

    # J1
    g = -mu/(norm(r)**3) * r

    # J2
    if use_j2:
        x,y,z = r
        coeff = 3/2 * J2_EARTH * mu * r_cb**2 / (2 * r_norm**5)
        axy = coeff * ((5*z*z)/(r_norm*r_norm) - 1)
        az = coeff * ((5*z*z)/(r_norm*r_norm) - 3)
        g += np.array([axy*x, axy*y, az*z])
    
    return g

#TODO: test?
