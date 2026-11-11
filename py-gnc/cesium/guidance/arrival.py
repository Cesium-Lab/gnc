from dataclasses import dataclass

import numpy as np
from numpy.linalg import norm

from ..constants import DEG_TO_RAD, KM_TO_M
from ..math.base import unit
from ..orbital.coes import coes_to_rv
from .interplanetary import interplanetary_circ_injection


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


@dataclass(frozen=True)
class ArrivalPlan:
    r_p: np.ndarray  # (3,)  periapsis position, MCI [m]
    v_p_hyp: np.ndarray  # (3,)  hyperbolic velocity at periapsis (pre-burn) [m/s]
    B_TR: np.ndarray  # (2,) aim point [B·T, B·R] [m]
    burns: list[tuple[np.ndarray, np.ndarray]]  # each: location/epoch, dv vector, |dv|
    dv_total: float


def plan_direct_injection(
    v_inf_hci_km: np.ndarray,
    target_periapse_m: float,
    target_raan: float,
    target_i: float,
    mu: float,
    deg=True,
    low_energy_path=True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Capture + plane change in one periapsis burn into a fixed (i, Ω) circular orbit."""
    # Everything in meters unless specified

    # Coplanar circular capture cost and hyperbola geometry
    _dv_periapse_m, ta_asymptote_rad, _B, e = interplanetary_circ_injection(
        norm(v_inf_hci_km * KM_TO_M), target_periapse_m, mu, deg=False
    )
    v_hat = unit(v_inf_hci_km)
    if deg:
        target_raan *= DEG_TO_RAD
        target_i *= DEG_TO_RAD

    # Use the existing element conversion at the ascending node.
    r_target, v_target = coes_to_rv(
        target_periapse_m, 0.0, target_i, target_raan, 0.0, 0.0, mu
    )
    orbit_normal = unit(np.cross(r_target, v_target))

    # Incoming velocity direction projected into the target plane.
    v_projected = v_hat - np.dot(v_hat, orbit_normal) * orbit_normal
    q = norm(v_projected)
    if q < 1 / e - 1e-12:
        raise ValueError(
            "Cannot enter target plane for the chosen v_inf and radius (usually because turning angle will not exist in the orbit)"
        )
    u = unit(v_projected)
    w = np.cross(orbit_normal, u)  # along track in target orbit

    # Arrival: dot(p_hat, v_inf_hat) = 1/e. Use basis vectors
    # Each p appears on both sides of the v_inf_vec when viewed from above
    theta_rad = np.pi - ta_asymptote_rad
    p_hats = np.array(
        [unit(np.cos(theta_rad) * u + sign * np.sin(theta_rad) * w) for sign in (1, -1)]
    )
    r_p_captures_mci = target_periapse_m * p_hats

    # Hyperbola things
    v_p = np.sqrt(norm(v_inf_hci_km * KM_TO_M) ** 2 + 2 * mu / target_periapse_m)
    v_hyp_hats = np.array([unit(np.cross(np.cross(p, v_hat), p)) for p in p_hats])
    v_p_captures_mci = v_p * v_hyp_hats

    # Capture orbit things
    v_captures_mci = np.sqrt(mu / target_periapse_m) * np.cross(orbit_normal, p_hats)
    dv_captures_mci = v_captures_mci - v_p_captures_mci

    # Keep both branches; select the smaller combined capture/plane-change burn.
    selected_index = int(np.argmin(norm(dv_captures_mci, axis=1)))
    if not low_energy_path:
        selected_index = 1 - selected_index  # switches off
    r_p_mci = r_p_captures_mci[selected_index]
    v_p_mci = v_p_captures_mci[selected_index]
    v_capture_mci = v_captures_mci[selected_index]
    dv_capture_mci = dv_captures_mci[selected_index]

    return r_p_mci, v_p_mci, v_capture_mci, dv_capture_mci
