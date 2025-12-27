# ruff: noqa: E741
# Energy sanity checks
import numpy as np

def calc_potential_energy(r: np.ndarray, mass_kg: float, mu: float):
    """Compute potential energy (currently only gravity). (Schaub 9.76)

    Args:
        mass (float): Mass of rigid body [kg]
        r (np.ndarray): Absolute position vector relative to center of central body [m] (typically the earth)
        mu (float): Gravitational parameter of central body [m3/s2] (typically the earth)

    Returns:
        float: Potential energy [J]
    """
    r_norm = np.linalg.norm(r)

    if r_norm < 1e-6:
        return 0

    return -mu * mass_kg / r_norm

def calc_kinetic_energy(v: np.ndarray, w: np.ndarray, mass_kg: np.ndarray, I: np.ndarray):
    """Compute kinetic energy (rotational and translational). (Schaub 4.55) \\
    Must compute with w and I in the same frame
    Args:
        v (np.ndarray): Absolute velocity vector [m/s] 
        w (np.ndarray): Angular velocity vector in specified frame [rad/s]
        mass_kg (np.ndarray): Mass of rigid body [kg]
        I (np.ndarray): Moment of inertia tensor in specified frame [kg*m2]

    Returns:
        float: Kinetic energy [J]
    """

    KE_translational = 0.5 * mass_kg * np.dot(v,v)

    if I is not None:
        KE_rotational = 0.5 * (w.T @ I @ w)
    else:
        KE_rotational = 0

    return KE_rotational + KE_translational

def calc_total_energy(mass_kg: float, I: np.ndarray, r: np.ndarray | list, 
                      v: np.ndarray, w: np.ndarray, mu: float):
    """Compute total energy (potential is currently only under gravity) \\
    Must compute with w and I in the same frame
    Args:
        mass_kg (float): Mass of rigid body [kg]
        I (np.ndarray): Moment of inertia tensor in specified frame [kg*m2]
        r (np.ndarray | list): Absolute position vector (relative to center of central body [m]) (typically the earth)
        v (np.ndarray): Absolute velocity vector [m/s] 
        w (np.ndarray): Angular velocity vector in specified frame [rad/s]
        mu (float): Gravitational parameter of central body [m3/s2] (typically the earth)

    Returns:
        float: Total energy [J]
    """
    PE = calc_potential_energy(r, mass_kg, mu)
    KE = calc_kinetic_energy(v, w, mass_kg, I)
    return PE + KE