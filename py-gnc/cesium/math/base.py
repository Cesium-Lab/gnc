from collections.abc import Callable

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
    if abs(v_norm) < 1e-6:
        return np.zeros(len(vec))

    return vec / v_norm


def center_of_curvature(r: np.ndarray, v: np.ndarray, a: np.ndarray):

    if norm(v) < 1e-6:
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


def angle_between(u, v):
    """Returns the angle between two vectors in radians."""
    u_norm = norm(u)
    v_norm = norm(v)

    # Avoid division by zero for zero vectors
    if u_norm < 1e-12 or v_norm < 1e-12:
        raise ValueError("Zero vectors do not have a defined angle.")

    dot_product = np.dot(u, v)
    cosine_similarity = dot_product / (u_norm * v_norm)

    # Avoid numerical overflow / floating point inaccuracies
    angle_in_radians = np.arccos(np.clip(cosine_similarity, -1.0, 1.0))

    return angle_in_radians


def jacobian(func: Callable, inputs: np.ndarray, out_dim: int, delta=0.001):

    if not isinstance(inputs, (np.ndarray, list)):
        inputs = np.array([inputs])
    

    # Helper
    def disturb(i: int, delta: float):
        ret = inputs.copy()
        ret[i] += delta
        return ret

    J = np.zeros((out_dim, len(inputs)))

    for i in range(len(inputs)):
        upper = func(disturb(i, delta))
        lower = func(disturb(i, -delta))

        J[:, i] = (upper - lower) / (2 * delta)
        
    # Back to 1 dimension
    if len(inputs) == 1:
        J = J[out_dim-1][0]

    return J
