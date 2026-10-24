import numpy as np

################################################################################
#               Angle
################################################################################


def wrap_to_pi(x):
  """Wrap angle(s) [rad] to `[-pi, pi)`"""
  return (x + np.pi) % (2 * np.pi) - np.pi

def wrap_to_2pi(x):
  """Wrap angle(s) [rad] to `[0, 2*pi)`"""
  return x % (2 * np.pi)

def dcm_x(theta: float):
    """Passive DCM for a rotation of frame 1 about its x-axis to frame 2.

    Parameters
    ----------
    theta : float
        Rotation angle of frame 2 with respect to frame 1 about x [rad].

    Returns
    -------
    np.ndarray (3, 3)
        DCM such that `x2 = dcm @ x1`.
    """
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), np.sin(theta)],
            [0, -np.sin(theta), np.cos(theta)],
        ]
    )


def dcm_y(theta: float):
    """Passive DCM for a rotation of frame 1 about its y-axis to frame 2.

    Parameters
    ----------
    theta : float
        Rotation angle of frame 2 with respect to frame 1 about y [rad].

    Returns
    -------
    np.ndarray (3, 3)
        DCM such that `x2 = dcm @ x1`.
    """
    return np.array(
        [
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)],
        ]
    )


def dcm_z(theta: float):
    """Passive DCM for a rotation of frame 1 about its z-axis to frame 2.

    Parameters
    ----------
    theta : float
        Rotation angle of frame 2 with respect to frame 1 about z [rad].

    Returns
    -------
    np.ndarray (3, 3)
        DCM such that `x2 = dcm @ x1`.
    """
    return np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    
################################################################################
#               Relative Motion
################################################################################

# TODO: what I did at Blue Origin
"""Curtis Example 1.10 and problems"""