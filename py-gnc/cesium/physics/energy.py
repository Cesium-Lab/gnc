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
    v: np.ndarray, mass: float, w: np.ndarray = None, I: np.ndarray = None
) -> float:
    """Compute translational plus rotational kinetic energy of a rigid body.

    Translational only (point mass, 3DOF) unless both `w` and `I` are given.

    Parameters
    ----------
    v : np.ndarray (3,)
        Inertial velocity of the body's center of mass [m/s].
    mass : float
        Mass of the body [kg].
    w : np.ndarray (3,), optional
        Angular velocity of the body, in the same frame as `I` (typically body)
        [rad/s]. If None, rotational energy is ignored.
    I : np.ndarray (3, 3), optional
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

    if I is not None and w is not None:
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
    KE = calc_kinetic_energy(v, mass_kg, w, I)
    return PE + KE


def calc_jacobi_constant(
    rv_S: np.ndarray, mu1: float, mu2: float, r1_bc: float, r2_bc: float, w_s: float
) -> float:
    """Jacobi constant of the circular restricted three-body problem.

    `C = w_s^2 (x^2 + y^2) + 2 (mu1 / r1 + mu2 / r2) - v^2`, conserved along any
    trajectory of `synodic_dynamics`; equal to minus twice the energy per unit mass
    in the rotating frame. Frame and parameters as in `synodic_dynamics`.

    Parameters
    ----------
    rv_S : np.ndarray (6,)
        Position and velocity in the synodic frame [m, m/s].
    mu1, mu2, r1_bc, r2_bc, w_s : float
        System parameters, as in `synodic_dynamics`.

    Returns
    -------
    float
        Jacobi constant [m2/s2].

    References
    ----------
    Curtis, Section 2.12.
    """
    x, y, z = rv_S[0:3]
    r1 = np.sqrt((x + r1_bc) ** 2 + y**2 + z**2)
    r2 = np.sqrt((x - r2_bc) ** 2 + y**2 + z**2)
    v = rv_S[3:6]
    return w_s**2 * (x**2 + y**2) + 2 * (mu1 / r1 + mu2 / r2) - np.dot(v, v)


def calc_specific_energy(
    rv: np.ndarray, mu: float, j2: float | None = None, r_cb: float | None = None
) -> float | np.ndarray:
    """Specific mechanical energy, optionally including the J2 potential.

    Without `j2` this is the Keplerian specific orbital energy `v^2 / 2 - mu / r`
    (so `a = -mu / (2 E)`). With `j2` it adds the J2 term of the zonal potential,

        `E = v^2 / 2 - mu / r + mu J2 R^2 (3 z^2 / r^2 - 1) / (2 r^3)`,

    which is `-(mu / r) J2 (R / r)^2 P2(sin(lat))` with the Legendre polynomial
    `P2(x) = (3 x^2 - 1) / 2` and `sin(lat) = z / r`, written as potential energy
    (negative of the geodesy sign convention). Its negative gradient is the J2
    acceleration in `grav_accel` (checked by finite differences in the tests).

    Use the J2 form for conservation checks: the J2 field is axisymmetric about z
    and static in the inertial frame, so it and `h_z` are conserved on a J2 coast,
    while the Keplerian energy is not. Use the Keplerian form for orbit elements.

    HMMM

    Parameters
    ----------
    rv : np.ndarray (6,) or (N, 6)
        Position and velocity in an inertial frame whose z-axis is the body's spin
        axis (e.g. ECI, MCI) [m, m/s].
    mu : float
        Gravitational parameter of the central body [m3/s2].
    j2 : float, optional
        J2 coefficient. If None, only the point-mass potential is used.
    r_cb : float, optional
        Equatorial radius of the central body [m]. Required if `j2` is given.

    Returns
    -------
    float or np.ndarray (N,)
        Specific energy [m2/s2].

    References
    ----------
    Standard zonal-harmonic expansion of the gravity potential, truncated at J2
    (Vallado 4e, Ch. 8, gravitational potential; check your edition's equation).
    """
    if j2 is not None and r_cb is None:
        raise ValueError("r_cb is required when j2 is given")
    rv = np.asarray(rv, dtype=float)
    r, v = rv[..., 0:3], rv[..., 3:6]
    r_norm = np.linalg.norm(r, axis=-1)
    energy = 0.5 * np.sum(v * v, axis=-1) - mu / r_norm

    # LEGENDRE POLYNOMIALS
    if j2 is not None:
        sin2_lat = (r[..., 2] / r_norm) ** 2
        energy += mu * j2 * r_cb**2 / (2 * r_norm**3) * (3 * sin2_lat - 1)
    return energy
