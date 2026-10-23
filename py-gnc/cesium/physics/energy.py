# Energy sanity checks
import numpy as np


def calc_potential_energy(r: np.ndarray, mass: float, mu: float):
    """Compute potential energy of a point mass (currently only gravity)

    Parameters
    ----------
    r : np.ndarray (3,)
        Position of the body relative to the central body's center (e.g. ECI) [m].
    mass : float
        Mass of the body [kg].
    mu : float
        Gravitational parameter of the central body (e.g. `MU_EARTH`) [m3/s2].

    Returns
    -------
    float
        Potential energy [J]. 0 if `r` is at the central body's center.

    References
    ----------
    Schaub, Equation 9.76.
    """
    r_norm = np.linalg.norm(r)

    if r_norm < 1e-6:
        return 0

    return -mu * mass / r_norm


def calc_kinetic_energy(
    v: np.ndarray, w: np.ndarray, mass: np.ndarray, I: np.ndarray
):
    """Compute translational plus rotational kinetic energy of a rigid body.

    Parameters
    ----------
    v : np.ndarray (3,)
        Inertial velocity of the body's center of mass [m/s].
    w : np.ndarray (3,)
        Angular velocity of the body, in the same frame as `I` (typically body) [rad/s].
    mass : float
        Mass of the body [kg].
    I : np.ndarray (3, 3) or None
        Moment of inertia tensor about the center of mass, in the same frame as
        `w` [kg*m2]. If None, rotational energy is ignored.

    Returns
    -------
    float
        Kinetic energy [J].

    References
    ----------
    Schaub, Equation 4.55.
    """

    KE_translational = 0.5 * mass * np.dot(v, v)

    if I is not None:
        KE_rotational = 0.5 * (w.T @ I @ w)
    else:
        KE_rotational = 0

    return KE_rotational + KE_translational


def calc_total_energy(
    mass_kg: float,
    I: np.ndarray,
    r: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    mu: float,
):
    """Compute total (kinetic plus potential) energy of a rigid body.

    Parameters
    ----------
    mass_kg : float
        Mass of the body [kg].
    I : np.ndarray (3, 3) or None
        Moment of inertia tensor about the center of mass, in the same frame as
        `w` [kg*m2]. If None, rotational energy is ignored.
    r : np.ndarray (3,)
        Position of the body relative to the central body's center (e.g. ECI) [m].
    v : np.ndarray (3,)
        Inertial velocity of the body's center of mass [m/s].
    w : np.ndarray (3,)
        Angular velocity of the body, in the same frame as `I` (typically body) [rad/s].
    mu : float
        Gravitational parameter of the central body (e.g. `MU_EARTH`) [m3/s2].

    Returns
    -------
    float
        Total energy [J].
    """
    PE = calc_potential_energy(r, mass_kg, mu)
    KE = calc_kinetic_energy(v, w, mass_kg, I)
    return PE + KE
