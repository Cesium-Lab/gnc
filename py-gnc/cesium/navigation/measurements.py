"""Measurement models h(x) and their Jacobians H = dh/dx, for the estimator.

Pure functions of the (estimated or true) state; noise, bias and availability belong
to the simulated sensor, not here. State convention for relative navigation:
`x_rsw = [r_rsw, v_rsw]` (6,), the chaser relative to the target in RSW [m, m/s].
"""

import numpy as np
from numpy.linalg import norm

###########################################################################
#                   Relative
###########################################################################


def meas_rel_pos(x_rsw: np.ndarray) -> np.ndarray:
    """Relative position measurement `z = r_rsw` (3,) [m]."""
    return x_rsw[0:3]


def meas_rel_pos_jac(x_rsw: np.ndarray) -> np.ndarray:
    """Jacobian of `meas_rel_pos`: `H = [I 0]` (3, 6). Linear, so `h(x) = H x`."""
    return np.hstack([np.eye(3), np.zeros((3, 3))])


def meas_range_az_el(x_rsw: np.ndarray) -> np.ndarray:
    """Range, azimuth and elevation of the chaser as seen from the target, in RSW.

    Returns
    -------
    np.ndarray (3,)
        `[range, az, el]` [m, rad, rad].
    """
    x, y, z = x_rsw[0:3]
    r = norm(x_rsw[0:3])
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.hypot(x, y))
    return np.array([r, az, el])


def meas_range_az_el_jac(x_rsw: np.ndarray) -> np.ndarray:
    """Jacobian of `meas_range_az_el` with respect to `x_rsw` (3, 6).

    With `r = |r_rsw|` and `horiz = sqrt(x^2 + y^2)`:

    `d range = [ x/r,               y/r,               z/r        ]` \\
    `d az    = [-y/horiz^2,         x/horiz^2,         0          ]` \\
    `d el    = [-x z/(r^2 horiz),  -y z/(r^2 horiz),   horiz/r^2  ]`

    Velocity columns are zero. Singular at `horiz = 0` (directly above or below
    the target), where az and el are undefined.
    """
    x, y, z = x_rsw[0:3]
    horiz2 = x**2 + y**2
    horiz = np.sqrt(horiz2)
    r2 = horiz2 + z**2
    r = np.sqrt(r2)

    H = np.zeros((3, 6))
    H[0, 0:3] = [x / r, y / r, z / r]
    H[1, 0:3] = [-y / horiz2, x / horiz2, 0.0]
    H[2, 0:3] = [-x * z / (r2 * horiz), -y * z / (r2 * horiz), horiz / r2]
    return H
