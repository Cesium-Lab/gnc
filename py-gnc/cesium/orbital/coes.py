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
    """Convert true anomaly to eccentric anomaly for an elliptical orbit.

    Parameters
    ----------
    ta : float
        True anomaly [rad].
    e : float
        Eccentricity, `0 <= e < 1`.

    Returns
    -------
    E : float
        Eccentric anomaly, wrapped to `[0, 2*pi)` [rad].
        
    References
    ----------
    Vallado 4e, Equation 2-14, p. 48.
    
    """
    E = 2*np.arctan2(
        np.sqrt(1-e)*np.sin(ta/2),
        np.sqrt(1+e)*np.cos(ta/2)
    )
    return E % (2*np.pi)

def ta_from_E(E: float, e: float):
    """Convert eccentric anomaly to true anomaly for an elliptical orbit.

    Parameters
    ----------
    E : float
        Eccentric anomaly [rad].
    e : float
        Eccentricity, `0 <= e < 1`.

    Returns
    -------
    ta : float
        True anomaly, wrapped to `[0, 2*pi)` [rad].
        
    References
    ----------
    Vallado 4e, Equation 2-13, p. 48.
    """
    ta = 2*np.arctan2(
        np.sqrt(1+e)*np.sin(E/2),
        np.sqrt(1-e)*np.cos(E/2)
    )
    return ta % (2*np.pi)

def kepler_eq(M: float, e: float, tol=1e-9, max_iter=100):
    """Solve Kepler's equation `M = E - e*sin(E)` for an elliptical orbit.

    Uses Newton-Raphson iteration.

    Parameters
    ----------
    M : float
        Mean anomaly [rad].
    e : float
        Eccentricity, `0 <= e < 1`.
    tol : float, optional
        Convergence tolerance on successive `E` iterates [rad]. Default is 1e-9.
    max_iter : int, optional
        Maximum number of iterations. Default is 100.

    Returns
    -------
    E : float
        Eccentric anomaly, wrapped to `[0, 2*pi)` [rad].

    Raises
    ------
    ValueError
        If `e` is outside `[0, 1)` or the iteration does not converge.

    References
    ----------
    Vallado 4e, Algorithm 2, p. 65.
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
    """Propagate an elliptical orbit analytically with secular J2 effects.

    Approximate implementation that only includes the secular J2 drift of RAAN
    and argument of periapsis.

    Parameters
    ----------
    r0 : np.ndarray (3,)
        Initial satellite position in the inertial frame (e.g. ECI) [m].
    v0 : np.ndarray (3,)
        Initial satellite velocity in the inertial frame (e.g. ECI) [m/s].
    dt : float
        Propagation time [s].
    r_cb : float, optional
        Equatorial radius of the central body [m]. Default is `R_EARTH`.
    mu : float, optional
        Gravitational parameter of the central body [m3/s2]. Default is `MU_EARTH`.
    j2 : float, optional
        J2 zonal harmonic coefficient of the central body. Default is `J2_EARTH`.

    Returns
    -------
    r : np.ndarray (3,)
        Satellite position in the inertial frame after `dt` [m].
    v : np.ndarray (3,)
        Satellite velocity in the inertial frame after `dt` [m/s].

    Raises
    ------
    ValueError
        If the initial orbit is not elliptical.

    References
    ----------
    Vallado 4e, Algorithm 65, p. 691.
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
    """Convert inertial position and velocity to classical orbital elements.

    Units of `r_I`, `v_I`, and `mu` must be consistent (all meter-based or all
    kilometer-based).

    Parameters
    ----------
    r_I : np.ndarray (3,)
        Satellite position in the inertial frame (e.g. ECI) [m] or [km].
    v_I : np.ndarray (3,)
        Satellite velocity in the inertial frame (e.g. ECI) [m/s] or [km/s].
    mu : float, optional
        Gravitational parameter of the central body [m3/s2] or [km3/s2].
        Default is `MU_EARTH` [m3/s2].

    Returns
    -------
    a : float
        Semi-major axis, same length units as `r_I`. `np.inf` for parabolic orbits.
    e : float
        Eccentricity.
    i : float
        Inclination [rad].
    raan : float
        Right ascension of the ascending node [rad]. 0 for equatorial orbits.
    aop : float
        Argument of periapsis [rad]. For elliptical equatorial orbits this is the
        longitude of periapsis; 0 for circular orbits.
    ta : float
        True anomaly [rad]. For circular orbits this is the argument of latitude
        (inclined) or the true longitude (equatorial).

    References
    ----------
    Vallado 4e, Algorithm 9, p. 113.
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
    """Convert classical orbital elements to inertial position and velocity.

    Units of `a` and `mu` must be consistent (both meter-based or both
    kilometer-based); the outputs use the same units.

    Parameters
    ----------
    a : float
        Semi-major axis [m] or [km].
    e : float
        Eccentricity.
    i : float
        Inclination [rad].
    raan : float
        Right ascension of the ascending node [rad].
    aop : float
        Argument of periapsis [rad].
    ta : float
        True anomaly [rad].
    mu : float, optional
        Gravitational parameter of the central body [m3/s2] or [km3/s2].
        Default is `MU_EARTH` [m3/s2].

    Returns
    -------
    r_I : np.ndarray (3,)
        Satellite position in the inertial frame (e.g. ECI) [m] or [km].
    v_I : np.ndarray (3,)
        Satellite velocity in the inertial frame (e.g. ECI) [m/s] or [km/s].

    References
    ----------
    Vallado 4e, Algorithm 10, p. 118.
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
