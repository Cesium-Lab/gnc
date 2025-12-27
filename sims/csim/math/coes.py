"""COES"""
# ruff: noqa: E741
import numpy as np
from ..world import MU_EARTH_KM
from numpy.linalg import norm

####################################################################################################
#               Kepler
####################################################################################################

def kelper_eq_ellipse(M: float, e: float, tol = 1e-9, max_iter = 100):
    """`(M,e) -> E` \\
    Solve Kelper's equation (elliptic) `M = E - e*sin(E)` using iteration \\
    Vallado 4e Algorithm 2 p. 65

    Args:
        M (float): Mean anomaly [rad]
        e (float): Eccentricity
        tol (float, optional): Tolerance needed to break iterating. Defaults to 1e-9.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.

    Raises:
        ValueError: If the number of iterations is exceeded

    Returns:
        float: Eccentric anomaly solution [rad]
    """
    past_180_deg = M > np.pi or (M > -np.pi and M < 0)
    E_n = M - e if past_180_deg else M + e

    for _ in range(max_iter):
        
        E_guess = E_n + (
            M - E_n + e*np.sin(E_n)
        ) / (1 - e*np.cos(E_n))

        if np.abs(E_guess - E_n) < tol:
            return E_n
        
        E_n = E_guess

    raise ValueError(f"Did not converge for {M=}, {e=}")

####################################################################################################
#               rv <--> coes
####################################################################################################

def rv_to_coes(r_eci: np.ndarray, v_eci: np.ndarray, mu: float = MU_EARTH_KM):
    """Convert ECI position and velocity vectors to classical Keplerian orbital elements. \\
    Can put r and v in meters, but `mu` must be in meters as well \\
    Vallado 4e Algorithm 9 p. 113

    Args:
        r_eci (np.ndarray): Position of satellite [km] or [m] (ECI)
        v_eci (np.ndarray): Velocity of satellite [km/s] or [m/s] (ECI)
        mu (float, optional): Gravitational parameter of central body [km3/s2] or [m3/s2]. Defaults to MU_EARTH_KM [km3/s2].

    Returns:
        tuple: (`a`, `e`, `i`, `Omega`, `omega`, `nu`): 
    - `a`: semi-major axis [km] or [m]
    - `e`: eccentricity
    - `i`: inclination [rad]
    - `Omega`: Right ascension of ascending node [rad]
    - `omega`: Argument of perigee [rad]
    - `nu`: True anomaly [rad]
    """
    # Specific relative angular momentum
    h = np.cross(r_eci,v_eci)
    h_norm = norm(h)

    # Vector pointing to ascending node
    n = np.cross([0,0,1], h)
    n_norm = norm(n)
    equatorial_orbit = abs(n_norm - 0.0) < 1e-5

    # Norms to make this easier
    r_norm = norm(r_eci)
    v_norm = norm(v_eci)
    v2 = v_norm**2
    r_dot_v = np.dot(r_eci,v_eci)

    # Eccentricity
    e = ( (v2 - mu/r_norm) * r_eci - r_dot_v * v_eci ) / mu
    e_norm = norm(e)
    circular_orbit = abs(e_norm - 0.0) < 1e-5

    # Specific Orbital Energy [km2/s2]
    E_sp = v2 / 2 - mu/r_norm

    if abs(e_norm - 1.0) > 1e-5:
        a = -mu/2/E_sp
    else:
        a = np.inf
    
    # Inclination
    i = np.arccos(h[2]/h_norm)

    # Right ascension of the ascending node [rad]
    raan = 0
    if not equatorial_orbit:
        raan = np.arccos(n[0] / n_norm)
        if n[1] < 0:
            raan = 2*np.pi - raan

    # Argument of perigee [rad]
    aop = 0
    if not equatorial_orbit and not circular_orbit:
        aop = np.arccos(np.dot(n,e) / n_norm / e_norm)
        if e[2] < 0:
            aop = 2*np.pi - aop

    # True anomaly [rad]
    ta = 0
    if not circular_orbit:
        ta = np.arccos(np.dot(e,r_eci) / e_norm / r_norm)
        if r_dot_v < 0:
            ta = 2*np.pi - ta

    return norm(a), e_norm, i, raan, aop, ta

def coes_to_rv(a: float, e: float, i: float, raan: float, aop: float, ta: float,
               mu: float = MU_EARTH_KM):
    """
    if `mu` in units of [m3/s2] then `a`, `r`, and `v` should/will be too. 

    Args:
        a (float): Semi-major axis [km] or [m]
        e (float): Eccentricity
        i (float): Inclination [rad]
        raan (float): Right ascension of ascending node (RAAN) [rad]
        aop (float): Argument of periapsis [rad]
        ta (float): True anomaly [rad]
        mu (float, optional): Gravitational parameter of central body [km3/s2] or [m3/s2]. Defaults to MU_EARTH_KM [km3/s2]

    Raises:
        NotImplementedError: If circular and equitorial (e=0 and i=0)
        NotImplementedError: If circular inclines (e=0 and i!=0)
        NotImplementedError: If elliptical equitorial (e!=0 and i=0)

    Returns:
        tuple: (`r_eci`, `v_eci`) Position and velocity vectors (ECI) [m] and [m/s] OR [km] and [km/s]  
    """

    # TODO: There should be actual logic for Circular Equatorial, Circular Inclined, and Elliptical Equatorial
    # but those require the r vector (i think) to get lambda_true and omega_tilde_true
    # so I could?? iterate? but the state should be tracked in r,v anyways
    # so I'd realistically only be using the rv_to_coes function
    
    equitorial = abs(i - 0) < 1e-5
    circular = abs(e - 0) < 1e-5

    if circular and equitorial:
        raise NotImplementedError(f"Logic for circular equatorial not implemented yet\n {(a,e,i,raan,aop,ta)}")
    elif circular:
        raise NotImplementedError(f"Logic for circular inclined not implemented yet\n {(a,e,i,raan,aop,ta)}")
    elif equitorial:
        raise NotImplementedError(f"Logic for elliptical equatorial not implemented yet\n {(a,e,i,raan,aop,ta)}")

    # Compute position and velocity in perifocal frame
    p = a * (1 - e**2)  # semi-latus rectum

    # Position in perifocal frame with components (P, Q, W) where P points to periapsis
    r_mag = p / (1 + e * np.cos(ta))
    r_pf = r_mag * np.array([np.cos(ta), np.sin(ta), 0.0])

    # Velocity in perifocal frame
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(ta), e + np.cos(ta), 0.0])

    # Rotation matrix from perifocal to ECI
    # R = R3(-raan) * R1(-i) * R3(-aop)
    sin_raan = np.sin(raan)
    cos_raan = np.cos(raan)
    sin_i = np.sin(i)
    cos_i = np.cos(i)
    sin_aop = np.sin(aop)
    cos_aop = np.cos(aop)

    # Combined rotation matrix (perifocal to ECI)
    R = np.array([
        [cos_raan*cos_aop - sin_raan*sin_aop*cos_i,    -cos_raan*sin_aop - sin_raan*cos_aop*cos_i,     sin_raan*sin_i],
        [sin_raan*cos_aop + cos_raan*sin_aop*cos_i,    -sin_raan*sin_aop + cos_raan*cos_aop*cos_i,    -cos_raan*sin_i],
        [sin_aop*sin_i,                          cos_aop*sin_i,                          cos_i]
    ])

    # Transform to ECI
    r_eci = R @ r_pf
    v_eci = R @ v_pf

    return r_eci, v_eci

    
# TODO: parabolic and hyperbolic depending on which orbits I want to simulate

# TODO: Kepler's problem?

####################################################################################################
#               Random quantities
####################################################################################################
    
# TODO: stuff like sma<-->period, etc.
    

################################################################################
#               Two-line Elements
################################################################################

# TODO: TLEs