import numpy as np
from numpy.linalg import norm


def unit(vec: np.ndarray):
    """Normalize a vector or quaternion to unit length.

    Parameters
    ----------
    vec : np.ndarray (N,)
        Vector or quaternion to normalize.

    Returns
    -------
    np.ndarray (N,)
        Unit vector, or zeros if norm of is below 1e-6.
    """
    v_norm = np.linalg.norm(vec)
    if abs(v_norm) < 0.000001:
        return np.zeros(len(vec))

    return vec / v_norm


def center_of_curvature(r: np.ndarray, v: np.ndarray, a: np.ndarray):

    if norm(v) < 1e-9:
        raise ValueError("Center of curvature undefined for zero velocity")

    t_hat = unit(v)
    a_on_t = np.dot(a, t_hat) * t_hat
    a_on_n = a - a_on_t

    a_n_norm = norm(a_on_n)
    if a_n_norm < 1e-9:
        raise ValueError("Center of curvature undefined for zero normal acceleration")

    n_hat = a_on_n / a_n_norm
    rho = np.dot(v, v) / a_n_norm

    r_C = r + rho * n_hat

    return r_C
