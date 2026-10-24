"""Approximate planetary ephemerides."""
from __future__ import annotations
import numpy as np

from ..math.rotations import wrap_to_2pi
from ..orbital.coes import coes_to_rv, kepler_eq, ta_from_E
from ..world.constants import MU_SUN
from ..constants import JD_CENTURY_TO_DAY, J2000_JD, DEG_TO_RAD, AU_TO_KM, KM_TO_M


# Columns: a [AU], e, i [deg], raan [deg], w_bar [deg], L [deg]
# Rows: Mercury, Venus, Earth-Moon barycenter, Mars, Jupiter,
#       Saturn, Uranus, Neptune

J2000_ELEMENTS = np.array([
    [0.38709927, 0.20563593, 7.00497902, 48.33076593, 77.45779628, 252.25032350],
    [0.72333566, 0.00677672, 3.39467605, 76.67984255, 131.60246718, 181.97909950],
    [1.00000261, 0.01671123, -0.00001531, 0.0, 102.93768193, 100.46457166],
    [1.52371034, 0.09339410, 1.84969142, 49.55953891, -23.94362959, -4.55343205],
    [5.20288700, 0.04838624, 1.30439695, 100.47390909, 14.72847983, 34.39644501],
    [9.53667594, 0.05386179, 2.48599187, 113.66242448, 92.59887831, 49.95424423],
    [19.18916464, 0.04725744, 0.77263783, 74.01692503, 170.95427630, 313.23810451],
    [30.06992276, 0.00859048, 1.77004347, 131.78422574, 44.96476227, -55.12002969],
])

# Rates: AU/Cy, 1/Cy, deg/Cy, deg/Cy, deg/Cy, deg/Cy
CENTENNIAL_RATES = np.array([
    [0.00000037, 0.00001906, -0.00594749, -0.12534081, 0.16047689, 149472.67411175],
    [0.00000390, -0.00004107, -0.00078890, -0.27769418, 0.00268329, 58517.81538729],
    [0.00000562, -0.00004392, -0.01294668, 0.0, 0.32327364, 35999.37244981],
    [0.00001847, 0.00007882, -0.00813131, -0.29257343, 0.44441088, 19140.30268499],
    [-0.00011607, -0.00013253, -0.00183714, 0.20469106, 0.21252668, 3034.74612775],
    [-0.00125060, -0.00050991, 0.00193609, -0.28867794, -0.41897216, 1222.49362201],
    [-0.00196176, -0.00004397, -0.00242939, 0.04240589, 0.40805281, 428.48202785],
    [0.00026291, 0.00005105, 0.00035372, -0.00508664, -0.32241464, 218.45945325],
])
"""Centennial rates of the J2000 orbital elements, one planet per row."""

def planet_elements_and_rv(planet_num: int, jd: float) -> tuple[tuple[float, float, float, float, float, float], np.ndarray, np.ndarray]:
    """
    Calculate a planet's heliocentric orbital elements and state vectors
    from a calendar date and Universal Time (UT).

    Parameters
    ----------
    planet_num : int
        1 (Mercury) to 8 (Neptune) BUT further ones beyond Mars technically require more coefficients
    jd : float
        Julian Date

    Returns
    -------
    coe : np.ndarray (6,)
        Heliocentric orbital elements
        [a, e, i, raan, aop, ta], where:
        - `a`: Semimajor axis [m].
        - `e`: Eccentricity.
        - `i`: Inclination [rad].
        - `raan`: Right ascension of ascending node [rad].
        - `aop`: Argument of perihelion [rad].
        - `ta`: True anomaly [rad].
    r_HCI : np.ndarray
        Heliocentric position vector [m].
    v_HCI : np.ndarray
        Heliocentric velocity vector [m/s].

    Notes
    ----------
        t0: Julian centuries elapsed since J2000.
        elements: Orbital elements evaluated at jd.

        Required functions:
            J0
        
    References
    ----------
    Data 
        http://ssd.jpl.nasa.gov/?planet_pos
    Curtis, Algorithm 8.1, p. 643
    """

    tt_jd = (jd - J2000_JD) / JD_CENTURY_TO_DAY
    
    elements, rates = j2000_planetary_elements(planet_num)
    

    
    
    #  1. Value of six elements (angles in degrees)
    a_au, e, i_deg, raan_deg, w_bar_deg, L_deg = [
        x + dx*tt_jd for x,dx in zip(elements, rates)
    ]
    
    a = a_au * AU_TO_KM * KM_TO_M
    i = i_deg * DEG_TO_RAD
    raan = wrap_to_2pi(raan_deg * DEG_TO_RAD)
    
    #  2. aop and M
    aop = wrap_to_2pi((w_bar_deg - raan_deg) * DEG_TO_RAD)
    M = wrap_to_2pi((L_deg - w_bar_deg) * DEG_TO_RAD)
    
    E = kepler_eq(M, e)
    ta = ta_from_E(E, e)
    
    r_HCI, v_HCI = coes_to_rv(a, e, i, raan, aop, ta, MU_SUN)
    
    return (a, e, i, raan, aop, ta), r_HCI, v_HCI

def j2000_planetary_elements(planet_id: int) -> tuple[np.ndarray, np.ndarray]:
    """J2000 heliocentric orbital elements and their centennial rates for a planet.

    Data is valid for 1800 AD - 2050 AD. A "century" (cty) is 36525 days.
    
    AA 279 Function Library, A. K. Barrows (2021), adapted from Curtis,
    Orbital Mechanics for Engineering Students. 

    Parameters
    ----------
    planet_id : int
        Planet number, 1 through 8 for Mercury through Neptune.

    Returns
    -------
    elements : np.ndarray (6,)
        J2000 orbital elements `[a, e, i, OMEGA, omega_bar, L]`:

        - `a`: semi-major axis [AU]
        - `e`: eccentricity
        - `i`: inclination [deg]
        - `OMEGA`: longitude of the ascending node [deg]
        - `omega_bar`: longitude of perihelion [deg]
        - `L`: mean longitude [deg]
    rates : np.ndarray (6,)
        Centennial rates of `elements`, in the same order
        `[a_dot, e_dot, i_dot, OMEGA_dot, omega_bar_dot, L_dot]`
        [AU/cty, 1/cty, deg/cty, deg/cty, deg/cty, deg/cty].

    Raises
    ------
    ValueError
        If `planet_id` is not in 1 through 9.

    References
    ----------
    Data 
        http://ssd.jpl.nasa.gov/?planet_pos
    Curtis, Algorithm 8.1 Subfunction, p. 643
    
    """
    if not 1 <= planet_id <= 8:
        raise ValueError(f"planet_id must be 1 through 8, got {planet_id}")

    return J2000_ELEMENTS[planet_id - 1].copy(), CENTENNIAL_RATES[planet_id - 1].copy()
