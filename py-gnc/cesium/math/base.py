import numpy as np

def unit(vec: np.ndarray):
    """Normalize a vector or quaternion to unit length.

    Parameters
    ----------
    vec : np.ndarray (N,)
        Vector or quaternion to normalize.

    Returns
    -------
    np.ndarray (N,)
        Unit vector, or zeros if the norm of is below 1e-6.
    """
    v_norm = np.linalg.norm(vec)
    if abs(v_norm) < 0.000001:
        return np.zeros(len(vec))

    return vec / v_norm
