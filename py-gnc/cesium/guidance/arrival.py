import numpy as np
from numpy.linalg import norm

from ..math.base import unit


def b_plane_frame(r_pci: np.ndarray, v_pci: np.ndarray, mu: float):
    """B-plane coordinates of the hyperbola a spacecraft state is on.

    Maps the trajectory the spacecraft is actually flying (not the target) onto
    the B-plane. Any state on the same two-body hyperbola gives the same result,
    since B depends only on the conserved h and e vectors. Targeting compares
    this against the desired aim point and corrects the difference.

    B-plane frame (Vallado), right-handed with `S x T = R`:
    - `S` - along the incoming asymptote (direction of v_inf)
    - `T` - perpendicular to S in the planet's equatorial plane, `unit(S x k)`
    - `R` - `S x T`, completing the frame (points "down" toward the south pole)

    B lies in the B-plane, so its S component is zero (Jah Eq. 8).

    Parameters
    ----------
    r_pci, v_pci : np.ndarray (3,)
        Spacecraft position and velocity relative to the planet, in
        planet-equatorial inertial axes (pole along +z, e.g. MCI) [m, m/s].
        Must be on a hyperbolic orbit (v_inf² > 0).
    mu : float
        Central-body gravitational parameter [m³/s²].

    Returns
    -------
    [B_T, B_R] : np.ndarray (2,)
        B-plane intersection point [m].

    References
    ----------
    Vallado 4e, Algorithm 78 (B-Plane I), p. 962.
    Jah, "Derivation of the B-Plane (Body Plane) and Its Associated Parameters",
        Eqs. 6 (S), 9 (B), 12 (|B|), 14-15 (B·R, B·T).
    """
    h_hat = unit(np.cross(r_pci, v_pci))
    r, v = norm(r_pci), norm(v_pci)
    e = ((v * v - mu / r) * r_pci - np.dot(r_pci, v_pci) * v_pci) / mu

    v_inf_sq = v * v - 2 * mu / r
    a = -mu / v_inf_sq
    b = -a * np.sqrt(np.dot(e, e) - 1)

    ta_asymptote = np.arccos(-1 / norm(e))

    S_hat = unit(e) / norm(e) + unit(np.cross(h_hat, e)) * np.sin(ta_asymptote)
    T_hat = unit(np.cross(S_hat, [0, 0, 1]))
    R_hat = unit(np.cross(S_hat, T_hat))

    B = b * np.cross(S_hat, h_hat)

    B_T = np.dot(B, T_hat)
    B_R = np.dot(B, R_hat)

    return np.array([B_T, B_R])


def b_plane_flyby(v_inf_arr: np.ndarray, v_inf_dep: np.ndarray, mu: float):
    """B-plane target based on flyby velocity vectors

    B-plane frame (Vallado), right-handed with `S x T = R`:
    - `S` - along the incoming asymptote (direction of v_inf)
    - `T` - perpendicular to S in the planet's equatorial plane, `unit(S x k)`
    - `R` - `S x T`, completing the frame (points "down" toward the south pole)

    Parameters
    ----------
    v_inf_arr, v_inf_dep : np.ndarray
        Arrival and departure excess velocities of spacecraft in PCI [m/s]
    mu : float
        Central-body gravitational parameter in m³/s².

    Returns
    -------
    B : np.ndarray (3,)
        B vector in the B-Plane [m]
    r_p : float
        Periapsis [m]

    References
    ----------
    Vallado 4e, Algorithm 79 (B-Plane II), p. 963.
    """

    S_hat = unit(v_inf_arr)
    h_hat = unit(np.cross(v_inf_arr, v_inf_dep))

    B_hat = unit(np.cross(S_hat, h_hat))

    # turning angle
    cos_phi = np.dot(v_inf_dep, v_inf_arr) / norm(v_inf_dep) / norm(v_inf_arr)
    phi = np.arccos(cos_phi)

    v_in2 = norm(v_inf_arr) ** 2
    r_p = mu / v_in2 * (1 / np.cos((np.pi - phi) / 2) - 1)
    B = mu / v_in2 * np.sqrt((1 + v_in2 * r_p / mu) ** 2 - 1)

    B_vec = B * B_hat

    return B_vec, r_p
