import numpy as np
from numpy.linalg import norm
from ..constants import RAD_TO_DEG, DEG_TO_RAD
from .base import unit

def conj(q: np.ndarray):
    """Conjugate of a scalar-first quaternion.

    Parameters
    ----------
    q : np.ndarray (4,)
        Quaternion `[w, x, y, z]`.

    Returns
    -------
    np.ndarray (4,)
        Conjugate quaternion `[w, -x, -y, -z]`.
    """
    q = -q
    q[0] *= -1
    return q


def hamilton_product(q: np.ndarray, w: np.ndarray | list):
    """Hamilton product of a quaternion with a pure-vector quaternion, `q * [0, w]`.

    Used for quaternion kinematics, `q_dot = 0.5 * hamilton_product(q, w)`.

    Parameters
    ----------
    q : np.ndarray (4,)
        Scalar-first attitude quaternion `[w, x, y, z]`.
    w : np.ndarray or list (3,)
        Angular velocity of the body expressed in the body frame [rad/s].

    Returns
    -------
    np.ndarray (4,)
        Quaternion product `q * [0, w]`.

    References
    ----------
    Schaub, Equation 3.112.
    """
    qw, qx, qy, qz = q
    wx, wy, wz = w

    # beta = np.array([
    #     [-qx, -qy, -qz],
    #     [qw, -qz, qy],
    #     [qz, qw, -qx],
    #     [-qy, qx, qw]
    # ])

    prod = np.empty(4, dtype=np.float64)
    prod[0] = -qx * wx - qy * wy - qz * wz
    prod[1] = qw * wx - qz * wy + qy * wz
    prod[2] = qw * wy + qz * wx - qx * wz
    prod[3] = qw * wz - qy * wx + qx * wy
    return prod
    # return beta @ w


# =========================================================================================================== #
#                                               Angle Axis                                                    #
# =========================================================================================================== #


# ----- Quaternion and Axis rotation -----#
def angle_axis_to_q(angle: float, axis: np.ndarray | list, degrees=False):
    """Convert a rotation angle and axis to a scalar-first quaternion.

    Parameters
    ----------
    angle : float
        Rotation angle about `axis` [deg if `degrees` is True, else rad].
    axis : np.ndarray or list (3,)
        Rotation axis. Does not need to be unit length.
    degrees : bool, optional
        If True, `angle` is in degrees; otherwise radians. Default is False.

    Returns
    -------
    np.ndarray (4,)
        Unit quaternion `[w, x, y, z]`.
    """

    angle_rad = angle * DEG_TO_RAD if degrees else angle
    unit_axis = unit(axis)

    w = np.cos(angle_rad / 2)
    if abs(w) < 0.1e-6:
        w = 0

    vector = np.sin(angle_rad / 2) * unit_axis

    return np.array([w, *vector])


def q_to_angle_axis(quat: np.ndarray | list, degrees=False):
    """Convert a scalar-first unit quaternion to a rotation angle and axis.

    Parameters
    ----------
    quat : np.ndarray or list (4,)
        Unit quaternion `[w, x, y, z]`.
    degrees : bool, optional
        If True, return `angle` in degrees; otherwise radians. Default is False.

    Returns
    -------
    angle : float
        Rotation angle [deg if `degrees` is True, else rad]. 0 for a near-identity
        rotation.
    axis : np.ndarray (3,)
        Unit rotation axis, or zeros for a near-identity rotation.
    """
    w, x, y, z = quat

    angle = 2 * np.arccos(w)

    if abs(angle) < 1e-6:
        return 0, np.zeros(3)

    i = x / np.sin(angle / 2)
    j = y / np.sin(angle / 2)
    k = z / np.sin(angle / 2)

    if degrees:
        angle *= RAD_TO_DEG

    return angle, np.array([i, j, k])


# ----- Quaternion and Rotation Matrix -----#
def q_to_DCM(q: np.ndarray):
    """Convert a scalar-first quaternion to a passive direction cosine matrix.

    Parameters
    ----------
    q : np.ndarray (4,)
        Quaternion `[w, x, y, z]`. Normalized internally.

    Returns
    -------
    DCM : np.ndarray (3, 3)
        Passive DCM equivalent to `q`. Zeros if `q` is the zero quaternion.

    Raises
    ------
    ValueError
        If `q` does not have 4 elements.

    References
    ----------
    https://motoq.github.io/doc/tnotes/dcm
    """
    if len(q) != 4:
        raise ValueError(f"Input quaternion should have 4 elements. Input was {q}")

    # 0 Quaternion for whatever reason
    q_norm = unit(q)

    if all(q_norm == 0):
        return np.zeros((3, 3))

    # In the PDF, it is notated as s,i,j,k, not w,x,y,z
    s, i, j, k = q_norm

    DCM = np.array(
        [
            [1 - 2 * (j**2 + k**2), 2 * (i * j + s * k), 2 * (i * k - s * j)],
            [2 * (i * j - s * k), 1 - 2 * (i**2 + k**2), 2 * (j * k + s * i)],
            [2 * (i * k + s * j), 2 * (j * k - s * i), 1 - 2 * (i**2 + j**2)],
        ]
    )
    return DCM


def DCM_to_q(DCM: np.ndarray):
    """Convert a passive direction cosine matrix to a scalar-first quaternion.

    Uses the largest-diagonal method for numerical stability.

    Parameters
    ----------
    DCM : np.ndarray (3, 3)
        Passive direction cosine matrix.

    Returns
    -------
    np.ndarray (4,)
        Quaternion `[w, x, y, z]`.

    References
    ----------
    https://motoq.github.io/doc/tnotes/dcm
    """

    assert np.shape(DCM) == (3, 3)

    c11, c12, c13 = DCM[0, :]
    c21, c22, c23 = DCM[1, :]
    c31, c32, c33 = DCM[2, :]

    tr = c11 + c22 + c33

    if tr > c11 and tr > c22 and tr > c33:
        w = np.sqrt((1 + c11 + c22 + c33) / 4)
        x = (c23 - c32) / 4 / w
        y = (c31 - c13) / 4 / w
        z = (c12 - c21) / 4 / w
    elif c11 > c22 and c11 > c33:
        x = np.sqrt((1 + c11 - c22 - c33) / 4)
        w = (c23 - c32) / 4 / x
        y = (c12 + c21) / 4 / x
        z = (c31 + c13) / 4 / x
    elif c22 > c33:
        y = np.sqrt((1 - c11 + c22 - c33) / 4)
        w = (c31 - c13) / 4 / y
        x = (c12 + c21) / 4 / y
        z = (c23 + c32) / 4 / y
    else:
        z = np.sqrt((1 - c11 - c22 + c33) / 4)
        w = (c12 - c21) / 4 / z
        x = (c31 + c13) / 4 / z
        y = (c23 + c32) / 4 / z

    return np.array([w, x, y, z])


# ----- Quaternion math! -----#
def quat_mult(q1: np.ndarray, q2: np.ndarray):
    """Hamilton product of two scalar-first quaternions, `q1 * q2`.

    Parameters
    ----------
    q1 : np.ndarray (4,)
        Left quaternion `[w, x, y, z]`.
    q2 : np.ndarray (4,)
        Right quaternion `[w, x, y, z]`.

    Returns
    -------
    np.ndarray (4,)
        Quaternion product `q1 * q2`.
    """

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


# ----- Applies Quaternion to Vector -----#
def quat_apply(quat: np.ndarray, v: np.ndarray, passive=True):
    """Rotate a vector, or re-express it in a rotated frame, using a quaternion.

    Parameters
    ----------
    quat : np.ndarray (4,)
        Quaternion `[w, x, y, z]`. Normalized internally.
    v : np.ndarray (3,)
        Vector to transform.
    passive : bool, optional
        If True, compute `q' * v * q`, expressing `v` in the frame rotated by
        `quat` (frame transformation). If False, compute `q * v * q'`, rotating
        `v` within the same frame. Default is True.

    Returns
    -------
    np.ndarray (3,)
        Transformed vector.
    """

    quat = unit(quat)
    v_quat = np.hstack(([0], v))

    if passive:
        """ q'*v*q """
        result = quat_mult(quat_mult(conj(quat), v_quat), quat)
    else:
        """ q*v*q' """
        result = quat_mult(quat_mult(quat, v_quat), conj(quat))

    if abs(result[0]) > 0.0001:
        print(f"Quanternion is not normalized. Result vector of {result}")

    # Discards
    return result[1:4]


# =========================================================================================================== #
#                                                   Angles                                                    #
# =========================================================================================================== #


# def angle_between(v1: np.ndarray | list, v2: np.ndarray | list):
#     """Returns the angle in radians between vectors 'v1' and 'v2'::

#     >>> angle_between((1, 0, 0), (0, 1, 0))
#     1.5707963267948966
#     >>> angle_between((1, 0, 0), (1, 0, 0))
#     0.0
#     >>> angle_between((1, 0, 0), (-1, 0, 0))
#     3.141592653589793
#     """

#     v1 = np.asarray(v1)
#     v2 = np.asarray(v2)


#     # Two vectors
#     if len(v1) == 3 and len(v2) == 3:

#         num = np.dot(v1, v2)
#         den = norm(v1) * norm(v2)

#         result = np.arccos(num / den)

#         return result

#     elif len(v1) == 4 and len(v2) == 4:
#         # Found with error quaternion
#         q_e = quat_mult(quat_inv(v1), v2)
#         angle, _ = axis_rot_from_quat(q_e)
#         return angle

#     raise RuntimeError("Check lengths of input vectors")
