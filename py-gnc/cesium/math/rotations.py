import numpy as np

################################################################################
#               Angle
################################################################################


def wrap_to_pi(x):
  return (x + np.pi) % (2 * np.pi) - np.pi

def wrap_to_2pi(x):
  return x % (2 * np.pi)

def dcm_x(theta: float):
    """Passive DCM from frame 1 to frame 2, rotated by theta [rad] (x2 = dcm @ x1)"""
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), np.sin(theta)],
            [0, -np.sin(theta), np.cos(theta)],
        ]
    )


def dcm_y(theta: float):
    """Passive DCM from frame 1 to frame 2, rotated by theta [rad] (x2 = dcm @ x1)"""
    return np.array(
        [
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)],
        ]
    )


def dcm_z(theta: float):
    """Passive DCM from frame 1 to frame 2, rotated by theta [rad] (x2 = dcm @ x1)"""
    return np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )