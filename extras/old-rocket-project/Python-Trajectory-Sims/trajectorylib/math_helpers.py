import numpy as np


def vec(x, y, z):
    return np.array([x, y, z])


def quat(s, v):
    return np.array([s, v[0], v[1], v[2]])
