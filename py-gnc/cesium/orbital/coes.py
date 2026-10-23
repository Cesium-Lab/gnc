"""COES"""

import numpy as np
from ..world.constants import MU_EARTH
from numpy.linalg import norm
from ..math.rotations import dcm_x, dcm_z
from ..math.quaternion import unit
from ..world import R_EARTH, J2_EARTH

####################################################################################################
#               Kepler
####################################################################################################

def E_from_ta(ta: float, e: float):
    E = 2*np.arctan2(
        np.sqrt(1-e)*np.sin(ta/2),
        np.sqrt(1+e)*np.cos(ta/2)
    )
    return E % (2*np.pi)

def ta_from_E(E: float, e: float):
    ta = 2*np.arctan2(
        np.sqrt(1+e)*np.sin(E/2),
        np.sqrt(1-e)*np.cos(E/2)
    )
    return ta % (2*np.pi)

def kepler_eq(M: float, e: float, tol=1e-9, max_iter=100):
    """`(M,e) -> E` \\
    Solve Kepler's equation (elliptic) `M = E - e*sin(E)` using iteration \\
    Vallado 4e Algorithm 2 p. 65

    Args:
        M (float): Mean anomaly [rad]
        e (float): Eccentricity [0, 1)
        tol (float, optional): Tolerance needed to break iterating. Defaults to 1e-9.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.

    Raises:
        ValueError: If the number of iterations is exceeded

    Returns:
        float: Eccentric anomaly solution [rad]
    """
    
    if not (0 <= e < 1):
        raise ValueError("Kepler solver requires 0 <= e < 1")

    M = M % (2*np.pi)
    E_n = M - e if M > np.pi else M + e

    for _ in range(max_iter):
        E_guess = E_n + (M - E_n + e * np.sin(E_n)) / (1 - e * np.cos(E_n))

        if np.abs(E_guess - E_n) < tol:
            return E_guess % (2*np.pi)
        E_n = E_guess

    raise ValueError(f"Did not converge for {M=}, {e=}")

# TODO: test
def pkepler(r0: np.ndarray, v0: np.ndarray, dt: float, r_cb = R_EARTH, mu = MU_EARTH, j2 = J2_EARTH) -> tuple[np.ndarray, np.ndarray]:
    """Approximate implementation
    Vallado 4e Algorithm 65 p. 691
    """
    a, e, i, raan, aop, ta = rv_to_coes(r0, v0, mu)

    if not (0 <= e < 1):
        raise ValueError("Requires an elliptical orbit")
    
    E = E_from_ta(ta, e)
    M = E - e*np.sin(E)
    
    p = a*(1 - e*e)
    n0 = np.sqrt(mu / a**3)
    
    # No perturbations included for now except J2
    factor = n0 * j2 * (r_cb/p)**2
    raan -= 3/2 * factor * np.cos(i) * dt
    aop += 3/4 * factor * (4 - 5*np.sin(i)**2) * dt
    M += n0*dt
    
    E = kepler_eq(M,e)
    ta = ta_from_E(E, e)
    
    r,v = coes_to_rv(a,e,i,raan,aop,ta,mu)
    
    return r,v
    


####################################################################################################
#               rv <--> coes
####################################################################################################


def rv_to_coes(r_I: np.ndarray, v_I: np.ndarray, mu: float = MU_EARTH):
    """Convert inertial position and velocity vectors to classical Keplerian orbital elements. \\
    Can put r and v in meters, but `mu` must be in meters as well \\
    Vallado 4e Algorithm 9 p. 113

    Args:
        r_I (np.ndarray): Inertial position of satellite [km] or [m]
        v_I (np.ndarray): Inertial velocity of satellite [km/s] or [m/s]
        mu (float, optional): Gravitational parameter of central body [km3/s2] or [m3/s2]. Defaults to MU_EARTH [m3/s2].

    Returns:
        tuple: (`a`, `e`, `i`, `Omega`, `omega`, `nu`): 
    - `a`: semi-major axis [km] or [m]
    - `e`: eccentricity
    - `i`: inclination [rad]
    - `Omega`: Right ascension of ascending node [rad]
    - `omega`: Argument of perigee [rad]
    - `nu`: True anomaly [rad]
    """
    # To make this easier
    r_norm = norm(r_I)
    v2 = np.dot(v_I, v_I)
    r_dot_v = np.dot(r_I, v_I)
    
    # Specific relative angular momentum
    h_vec = np.cross(r_I, v_I)
    h = norm(h_vec)

    # Vector pointing to ascending node
    n_vec = np.cross([0, 0, 1], h_vec)
    n = norm(n_vec)

    # Eccentricity
    e_vec = ((v2 - mu / r_norm) * r_I - r_dot_v * v_I) / mu
    e = norm(e_vec)
    
    # Inclination
    i = np.arctan2(n, h_vec[2])

    circular_orbit = e < 1e-8
    equatorial_orbit = n/h < 1e-8

    # Specific Orbital Energy [km2/s2]
    E_sp = v2 / 2 - mu / r_norm

    if abs(e - 1.0) > 1e-5:
        a = -mu / 2 / E_sp
    else:
        a = np.inf

    # Right ascension of the ascending node [rad]
    raan = 0.0
    if not equatorial_orbit:
        raan = np.arctan2(n_vec[1], n_vec[0]) % (2*np.pi)

    # Argument of perigee [rad]
    aop = 0.0
    if not circular_orbit:
        if not equatorial_orbit:
            aop = np.arctan2(
                np.dot(np.cross(n_vec, e_vec), unit(h_vec)),
                np.dot(n_vec, e_vec)
            ) % (2*np.pi)
        else:
            # Longitude of periapsis
            aop = np.arctan2(
                np.sign(h_vec[2]) * e_vec[1],
                e_vec[0]
            ) % (2*np.pi)

    # True anomaly [rad]
    ta = 0.0
    if not circular_orbit:
        ta = np.arctan2(
            np.dot(np.cross(e_vec, r_I), unit(h_vec)),
            np.dot(e_vec, r_I)
        ) % (2*np.pi)
    elif not equatorial_orbit:
        # Argument of latitude
        ta = np.arctan2(
            np.dot(np.cross(n_vec, r_I), unit(h_vec)),
            np.dot(n_vec, r_I)
        ) % (2*np.pi)
    else:
        # True longitude
        ta = np.arctan2(
            np.sign(h_vec[2]) * r_I[1],
            r_I[0]
        ) % (2*np.pi)

    return a, e, i, raan, aop, ta


def coes_to_rv(
    a: float,
    e: float,
    i: float,
    raan: float,
    aop: float,
    ta: float,
    mu: float = MU_EARTH,
):
    """
    if `mu` in units of [m3/s2] then `a`, `r`, and `v` should/will be too. \\
    Vallado 4e Algorithm 10 p. 118

    Args:
        a (float): Semi-major axis [km] or [m]
        e (float): Eccentricity
        i (float): Inclination [rad]
        raan (float): Right ascension of ascending node (RAAN) [rad]
        aop (float): Argument of periapsis [rad]
        ta (float): True anomaly [rad]
        mu (float, optional): Gravitational parameter of central body [km3/s2] or [m3/s2]. Defaults to MU_EARTH [m3/s2]

    Returns:
        tuple: (`r_I`, `v_I`) Inertial position and velocity vectors [m] and [m/s] OR [km] and [km/s]
    """

    # Compute position and velocity in perifocal frame
    p = a * (1 - e**2)  # semi-latus rectum

    # Position in perifocal frame with components (P, Q, W) where P points to periapsis
    r_mag = p / (1 + e * np.cos(ta))
    r_PQW = r_mag * np.array([np.cos(ta), np.sin(ta), 0.0])

    # Velocity in perifocal frame
    v_PQW = np.sqrt(mu / p) * np.array([-np.sin(ta), e + np.cos(ta), 0.0])

    # Rotation matrix from perifocal to ECI
    dcm_I_wrt_PQW = dcm_z(-raan) @ dcm_x(-i) @ dcm_z(-aop)

    # Transform to inertial
    r_I = dcm_I_wrt_PQW @ r_PQW
    v_I = dcm_I_wrt_PQW @ v_PQW

    return r_I, v_I

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
