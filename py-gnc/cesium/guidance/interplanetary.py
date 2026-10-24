import numpy as np
from numpy.linalg import norm

from ..constants import RAD_TO_DEG


def interplanetary_arrival(
    v_arrival_PCI: float, r_p: float, mu: float, deg=True
) -> tuple[float, float, float]:
    """Interplanetary Intercept along planet velocity vector

    Parameters
    ----------
    v_arrival_PCI : float
        Planet centered arrival velocity in inertial frame [m/s]
    r_p : float
        Periapse radius [m]
    mu : float
        Central body gravitational parameter [m3/s2]

    Returns
    -------
    dv_p : float
        Delta-v at periapsis for circular capture
    ta_asymptote : float
        Arrival true anomaly angle to asymptote [rad]
    aiming_radius : float
        Distance between asymptote and periapsis [m]
    e : float
        Eccentricity


    References
    -------
    AA 279B Lecture 9 p. 23
    """
    v_p = np.sqrt(v_arrival_PCI**2 + 2 * mu / r_p)
    v_circ = np.sqrt(mu / r_p)
    dv_p = norm(v_circ - v_p)

    e = 1 + r_p * v_arrival_PCI**2 / mu

    ta_asymptote = np.arccos(-1 / e)

    aiming_radius = mu / (v_arrival_PCI**2) * np.sqrt(e * e - 1)

    if deg:
        ta_asymptote *= RAD_TO_DEG

    return dv_p, ta_asymptote, aiming_radius, e
