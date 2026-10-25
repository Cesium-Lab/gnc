import numpy as np
from numpy.linalg import norm

from ..constants import RAD_TO_DEG
from ..math.base import angle_between, jacobian


def interplanetary_circ_injection(
    v_arrival_PCI: float, r_p: float, mu: float, deg=True
) -> tuple[float, float, float, float]:
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
    assert e > 1

    ta_asymptote = np.arccos(-1 / e)

    aiming_radius = mu / (v_arrival_PCI**2) * np.sqrt(e * e - 1)

    if deg:
        ta_asymptote *= RAD_TO_DEG

    return dv_p, ta_asymptote, aiming_radius, e

# eps=1e-8rad
def flyby_shoot(v_arrival_PCI: float, v_departure_PCI: float, delta_des: float, 
             mu: float, deg=False, eps = 1e-8, d = 10, verbose=False) -> tuple[float, float, float, float, float]:
    
    delta_des = delta_des / RAD_TO_DEG if deg else delta_des
    
    def delta_tot(r_p):
        return np.arcsin(1/ (1 + r_p * v_arrival_PCI**2 / mu)) + np.arcsin(1/ (1 + r_p * v_departure_PCI**2 / mu))
    
    # Guess if circular velocity with e=2
    e=2
    r_p = mu*(e-1) / (v_arrival_PCI**2)
    
    iter = 20
    for i in range(iter):
        
        delta_k = delta_tot(r_p)
        
        d_delta = delta_des - delta_k
        
        if verbose:
            print(f"d_delta: {d_delta:.5f}m")
        
        if norm(d_delta) < eps:
            # print(f"\nConverged with:\n\tdr = {d_delta:.5f} km")
            break
        # print(f"{mu=}")
        # print(f"{d=}")
        # print(f"{r_p=}")
        J = (delta_tot(r_p + d) - delta_tot(r_p - d)) / (2*d)
        # print(f"{J=}")
        # print(f"{)=}"
        # print(f"{delta_des=}")
        # print(f"{delta_k=}")
        # print(f"{d_delta=}")
        # print(f"{(d_delta / J)=}")
        # r_p += d_delta / J # single value
        
        step = d_delta / J

        # Make sure periapsis is positive
        alpha = 1.0

        while r_p + alpha * step <= 0:
            alpha *= 0.5

        r_p += alpha * step

    else:
        raise ValueError
        
    # dv
    v_p1 = np.sqrt(v_arrival_PCI**2 + 2 * mu / r_p)
    v_p2 = np.sqrt(v_departure_PCI**2 + 2 * mu / r_p)
    dv = abs(v_p2 - v_p1)
    # e
    e1 = 1 + r_p * v_arrival_PCI**2 / mu
    e2 = 1 + r_p * v_departure_PCI**2 / mu

    return dv, v_p1, v_p2, r_p, e1, e2

# def flyby_dv(v_arrival_PCI: float, b_arrival: float, 
#              v_departure_PCI: float, b_departure: float, 
#              mu: float, dv: float = 0, deg=True) -> tuple[float, float, float]:

    # e1 = np.sqrt((b_arrival * v_arrival_PCI**2 / mu)**2  + 1)
    # e2 = np.sqrt((b_arrival * v_arrival_PCI**2 / mu)**2  + 1)
    # assert e1 > 1
    # assert e2 > 1
    
    # r_p = mu*(e-1) / (v_arrival_PCI**2)
    # v_p1 = np.sqrt(v_arrival_PCI**2 - 2 * mu / r_p)
    # v_p2 = v_p1 + dv
    
    # v_departure_PCI = np.sqrt(v_arrival_PCI**2 - 2 * mu / r_p)
    
    # dv = norm()
    
    
    # v_p = np.sqrt(v_arrival_PCI**2 + 2 * mu / r_p)
    # v_circ = np.sqrt(mu / r_p)
    # dv_p = norm(v_circ - v_p)
    
    # delta = angle_between()
    