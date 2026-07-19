from .constants import *
import numpy as np
from numpy.linalg import norm


#----- Quaternion and Axis rotation -----#
def quat_from_axis_rot(angle, axis):
    degrees = angle
    axis_norm = axis / norm(axis)

    w = np.cos(degrees/2)
    x,y,z = [np.sin(degrees / 2)*i for i in axis_norm]

    return np.array([w,x,y,z])

def axis_rot_from_quat(quat):
    w,x,y,z = quat
    
    angle = 2*np.arccos(w)

    if angle == 0:
        return np.zeros(4)
    
    i = x/np.sin(angle/2)
    j = y/np.sin(angle/2)
    k = z/np.sin(angle/2)


    return angle, np.array([i, j, k])

#----- Quaternion and Rotation Matrix -----#
def R_from_quat(q):
    if len(q) != 4:
        print(f"Input quaternion should have 4 elements. Input was {q}")
        return np.identity(3)
    
    # 0 Quaternion   
    if norm(q) == 0:

        return np.zeros((3,3))
    
    q_norm = unit(q)

    # if abs(q_norm[0] - 1) < 1e-10:
    #     print(f"Identity quaternion with Q={q_norm}")

    w, i, j, k = q_norm

    R00 = 1 - 2*(j*j + k*k)
    R01 = 2 * (i*j - k*w)
    R02 = 2 * (i*k + j*w)

    R10 = 2 * (i*j + k*w)
    R11 = 1 - 2 * (i*i + k*k)
    R12 = 2 * (j*k - i*w)

    R20 = 2 * (i*k - j*w)
    R21 = 2 * (j*k + i*w)
    R22 = 1 - 2*(i*i + j*j)

    R = np.array([[R00, R01, R02],
                  [R10, R11, R12],
                  [R20, R21, R22]])
    return R

def quat_from_R(R):

    # Makes sure the transpose can be taken
    if type(R) != np.ndarray:
        R = np.array(R)

    # Insomniac games formula, but taking transpose 
    # because they use scaler last convection
    row0, row1, row2 = R.T 
    m00, m01, m02 = row0
    m10, m11, m12 = row1
    m20, m21, m22 = row2

    if m22 < 0:
        if m00 > m11:
            t = 1 + m00 - m11 - m22
            q = np.array([t, m01+m10, m20+m02, m12-m21])    
        else:
            t = 1 - m00 + m11 - m22
            q = np.array([m01+m10, t, m12+m21, m20-m02])
    else:
        if m00 < -m11:
            t = 1 - m00 - m11 + m22
            q = np.array([m20+m02, m12+m21, t, m01-m10])
        else:
            t = 1 + m00 + m11 + m22
            q = np.array([m12-m21, m20-m02, m01-m10, t])

    q *= 1/2/np.sqrt(t)

    return q


#----- Quaternion math! -----#
def quat_mult(q1, q2):

    if len(q1) == 3:
        quat1 = np.array([0, *q1])
        # print("q1 is a vector")
    else:
        quat1 = q1

    if len(q2) == 3:
        quat2 = np.array([0, *q2])
        # print("q2 is a vector")
    else:
        quat2 = q2
    # print(quat1)
    # print(quat2)
    
    w1, x1, y1, z1 = quat1
    w2, x2, y2, z2 = quat2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

#----- Quaternion Operations -----#
def quat_inv(q):
    q = -q
    q[0] *= -1
    return q

def unit(q):
    if abs(norm(q)) < 0.000001:
        return np.zeros(len(q))
    
    return q / norm(q)

#----- Applies Quaternion to Vector -----#
def quat_apply(quat, vector):
    temp = quat_mult(quat, vector)
    rslt = quat_mult(temp, quat_inv(quat))

    if abs(rslt[0]) > 0.0001:
        print(f"Quanternion is not normalized. Result vector of {rslt}") 
    
    # Discards 
    return rslt[1:]
