import numpy as np
from numpy.linalg import norm
from ..constants import RAD_TO_DEG, DEG_TO_RAD


def unit(q: np.ndarray):
    q_norm = norm(q)
    if abs(q_norm) < 0.000001:
        return np.zeros(len(q))

    return q / q_norm


def conj(q: np.ndarray):
    q = -q
    q[0] *= -1
    return q


def hamilton_product(q: np.ndarray, w: np.ndarray | list):
    """(Schaub 3.112)"""
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

    angle_rad = angle * DEG_TO_RAD if degrees else angle
    unit_axis = unit(axis)

    w = np.cos(angle_rad / 2)
    if abs(w) < 0.1e-6:
        w = 0

    vector = np.sin(angle_rad / 2) * unit_axis

    return np.array([w, *vector])


def q_to_angle_axis(quat: np.ndarray | list, degrees=False):
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
    """https://motoq.github.io/doc/tnotes/dcm"""
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
    """
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
def mul(q1: np.ndarray, q2: np.ndarray):

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

    quat = unit(quat)
    v_quat = np.hstack(([0], v))

    if passive:
        """ q'*v*q """
        result = mul(mul(conj(quat), v_quat), quat)
    else:
        """ q*v*q' """
        result = mul(mul(quat, v_quat), conj(quat))

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
