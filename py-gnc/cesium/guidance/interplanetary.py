import numpy as np
from numpy.linalg import norm

from ..constants import RAD_TO_DEG
from ..math.base import unit
from .types import (
    FlybyStichScalarInput,
    FlybyStichScalarOutput,
    FlybyStichVectorInput,
    FlybyStichVectorOutput,
)


# TODO: test with more unit tests
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
    deg : bool
        Whether output `ta_asymptote` is in degrees (default False)

    Returns
    -------
    dv_p : float
        Delta-v at periapsis for circular capture
    ta_asymptote : float
        Arrival true anomaly angle to asymptote [rad or deg]
    b : float
        Aiming radius / offset distance between asymptote and periapsis [m]
    e : float
        Eccentricity of arrival hyperbola


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

    b = mu / (v_arrival_PCI**2) * np.sqrt(e * e - 1)

    if deg:
        ta_asymptote *= RAD_TO_DEG

    return dv_p, ta_asymptote, b, e


# eps=1e-8rad
def flyby_stich(
    inputs: FlybyStichScalarInput,
    delta_in_deg=False,
    eps=1e-8,
    d=10,
    verbose=False,
    iter=20,
) -> FlybyStichScalarOutput:
    """_summary_

    Parameters
    ----------
    inputs : FlybyStichScalarInput
        - v_arrival_PCI: float [m/s], 
        - v_departure_PCI: float [m/s]
        - delta_des: float [rad or deg]
        - mu: float [m3/s2]
    delta_in_deg : bool, optional
        Whether turning angle delta is in degrees , by default False
    eps : float, optional
        Angle error for convergence [rad], by default 1e-8
    d : int, optional
        Periapsis radius perturbation for iteration, by default 10
    verbose : bool, optional
        Whether error is printed every iteration, by default False
    iter : int, optional
        Iterations, by default 20

    Returns
    -------
    FlybyStichScalarOutput
        Too much to list just look at docstring

    Raises
    ------
    ValueError
        If it does not converge within `eps` parameter
    """
    
    v_arrival_PCI = inputs.v_arrival_PCI
    v_departure_PCI = inputs.v_departure_PCI
    delta_des = inputs.delta_des
    mu = inputs.mu

    delta_des = delta_des / RAD_TO_DEG if delta_in_deg else delta_des

    def delta_tot(r_p):
        return np.arcsin(1 / (1 + r_p * v_arrival_PCI**2 / mu)), np.arcsin(
            1 / (1 + r_p * v_departure_PCI**2 / mu)
        )

    # Guess if circular velocity with e=2
    e = 2
    r_p = mu * (e - 1) / (v_arrival_PCI**2)

    for i in range(iter):
        delta1, delta2 = delta_tot(r_p)

        d_delta = delta_des - (delta1 + delta2)

        if verbose:
            print(f"d_delta: {d_delta:.5f}m")

        if norm(d_delta) < eps:
            break

        J = (sum(delta_tot(r_p + d)) - sum(delta_tot(r_p - d))) / (2 * d)

        step = d_delta / J

        # Make sure periapsis is positive
        alpha = 1.0

        while r_p + alpha * step <= 0:
            alpha *= 0.5

        r_p += alpha * step

    else:
        raise ValueError(f"Did not converge in {iter} iterations. Got to {norm(d_delta) * RAD_TO_DEG}º error")

    # dv
    v_p1 = np.sqrt(v_arrival_PCI**2 + 2 * mu / r_p)
    v_p2 = np.sqrt(v_departure_PCI**2 + 2 * mu / r_p)
    dv = abs(v_p2 - v_p1)
    # e
    e1 = 1 + r_p * v_arrival_PCI**2 / mu
    e2 = 1 + r_p * v_departure_PCI**2 / mu
    # b
    b1 = mu / (v_arrival_PCI**2) * np.sqrt(e1 * e1 - 1)
    b2 = mu / (v_departure_PCI**2) * np.sqrt(e2 * e2 - 1)
    
    if delta_in_deg:
        delta1 *= RAD_TO_DEG
        delta2 *= RAD_TO_DEG

    return FlybyStichScalarOutput(dv, v_p1, v_p2, r_p, e1, e2, delta1, delta2, b1, b2)

def flyby_stich_vector(
    inputs: FlybyStichVectorInput,
    deg=False,
    eps=1e-8,
    d=10,
    verbose=False,
) -> FlybyStichVectorOutput:
    """_summary_

    Parameters
    ----------
    inputs : FlybyStichScalarInput
        - v_arrival_PCI: float [m/s], 
        - v_departure_PCI: float [m/s]
        - mu: float [m3/s2]
    delta_in_deg : bool, optional
        Whether turning angle delta is in degrees , by default False
    eps : float, optional
        Angle error for convergence [rad], by default 1e-8
    d : int, optional
        Periapsis radius perturbation for iteration, by default 10
    verbose : bool, optional
        Whether error is printed every iteration, by default False

    Returns
    -------
    FlybyStichVectorOutput
        Too much to list just look at docstring
    """
    v_arrival_PCI = inputs.v_arrival_PCI
    v_departure_PCI = inputs.v_departure_PCI
    mu = inputs.mu

    # Turning angle since we have vectors
    cos_delta = np.dot(v_arrival_PCI, v_departure_PCI) / (norm(v_arrival_PCI) * norm(v_departure_PCI))
    turning_angle_rad = np.arccos(np.clip(cos_delta, -1, 1))
    
    # Shooting using AA 279B method. deltas in RAD for this calc
    dv, v_p1, v_p2, r_p, e1, e2, delta1, delta2, b1, b2 = flyby_stich(
        norm(v_arrival_PCI), norm(v_departure_PCI),
        turning_angle_rad, mu, False, eps, d, verbose
    )
    assert np.isclose(delta1 + delta2, turning_angle_rad, atol=eps)
    
    # Normal to everything
    n_vec = unit(np.cross(v_arrival_PCI, v_departure_PCI))
    
    # b vectors
    b1_vec = b1 * unit(np.cross(v_arrival_PCI, n_vec))
    b2_vec = b2 * unit(np.cross(v_departure_PCI, n_vec))
    
    # v_inf vecs
    v_arr_proj = b1 * np.tan(delta1) * unit(v_arrival_PCI)
    v_dep_proj = b2 * np.tan(delta2) * unit(v_departure_PCI)
    
    # r_p vector (assert equal)
    r_p1 = r_p * unit(v_arr_proj + b1_vec)
    r_p2 = r_p * unit(-v_dep_proj + b2_vec)
    assert norm(r_p1 - r_p2) < 1e-3 # mm?
    
    v_p1_vec = v_p1 * unit(np.cross(n_vec, r_p1))
    v_p2_vec = v_p2 * unit(np.cross(n_vec, r_p1))
    dv_vec = dv_vec = (v_p2 - v_p1) * unit(np.cross(n_vec, r_p1))
    assert np.isclose(norm(dv_vec), dv)
    assert np.allclose(v_p2_vec - v_p1_vec, dv_vec)
    
    if deg:
        delta1 *= RAD_TO_DEG
        delta2 *= RAD_TO_DEG
    
    return FlybyStichVectorOutput(dv_vec, v_p1_vec, v_p2_vec, r_p1, e1, e2, delta1, delta2, b1_vec, b2_vec)
    
def flyby_
    