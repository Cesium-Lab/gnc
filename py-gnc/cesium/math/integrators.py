from typing import Callable

import numpy as np

# @njit
# def euler_func(
#     t: float,
#     dt: float,
#     x_prev: np.ndarray,
#     x_dot: Callable[[float, np.ndarray], np.ndarray],
# ):
#     """Euler's Method using a function for the derivative"""
#     x_n = dt * x_dot(t, x_prev) + x_prev

#     return x_n


def rk4_func(
    t: float,
    dt: float,
    x_prev: float,
    x_dot: Callable[[float, np.ndarray], np.ndarray],
    params=None,
):
    """Advance a state one step with the classic 4th-order Runge-Kutta method.

    Parameters
    ----------
    t : float
        Time at the start of the step [s].
    dt : float
        Step size [s].
    x_prev : float or np.ndarray
        State at time `t`.
    x_dot : Callable
        Derivative function with signature `x_dot(t, x, params) -> dx/dt`.
    params : object, optional
        Extra parameters passed through to `x_dot` (e.g. `RigidBodyParams`).
        Default is None.

    Returns
    -------
    float or np.ndarray
        State at time `t + dt`.
    """
    k1 = x_dot(t, x_prev, params)
    k2 = x_dot(t + dt / 2, x_prev + dt * k1 / 2, params)
    k3 = x_dot(t + dt / 2, x_prev + dt * k2 / 2, params)
    k4 = x_dot(t + dt, x_prev + dt * k3, params)

    x_n = x_prev + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_n
