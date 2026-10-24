"""
TODO:
- Normal J1 gravity
- J2, J3 even
- gravity gradient
"""

import numpy as np
from numpy.linalg import norm
from ..world import MU_EARTH, R_EARTH


def grav_accel(r: np.ndarray, mu: float = MU_EARTH, j2=None, r_cb: float = R_EARTH):
    """Gravitational acceleration from a point mass, optionally with J2.

    Parameters
    ----------
    r : np.ndarray (3,)
        Satellite position relative to the central body's center, in an inertial
        or body-fixed frame whose z-axis is the body's spin axis (e.g. ECI) [m].
    mu : float, optional
        Gravitational parameter of the central body [m3/s2]. Default is `MU_EARTH`.
    j2 : float, optional
        Enables the J2 perturbation when truthy. If None, J2 is not applied.
        Default is None.
    r_cb : float, optional
        Equatorial radius of the central body, used for J2 [m]. Default is `R_EARTH`.

    Returns
    -------
    np.ndarray (3,)
        Gravitational acceleration in the same frame as `r` [m/s2]. Zeros if `r`
        is at the center or contains infinity.

    References
    ----------
    Vallado 4e, Equation 1-14, p. 23.
    J2 formula from the `Poliastro docs
    <https://docs.poliastro.space/en/stable/autoapi/poliastro/core/perturbations/index.html>`_
    and `NASA NTRS 20040031507
    <https://ntrs.nasa.gov/api/citations/20040031507/downloads/20040031507.pdf>`_.
    """

    # Prevents singularities (also np.inf * 0 == np.nan)
    if any(r == np.inf):
        return np.zeros(3)

    r_norm = norm(r)

    # No radius
    if abs(r_norm) < 1e-6:
        return np.zeros(3)

    # J1
    g = -mu / (norm(r) ** 3) * r

    # J2
    if j2:
        x, y, z = r
        coeff = 3 * j2 * mu * r_cb**2 / (2 * r_norm**5)
        axy = coeff * ((5 * z * z) / (r_norm * r_norm) - 1)
        az = coeff * ((5 * z * z) / (r_norm * r_norm) - 3)
        g += np.array([axy * x, axy * y, az * z])

    return g


# TODO: test?
