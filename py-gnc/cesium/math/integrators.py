from collections.abc import Callable

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
    x_prev: np.ndarray,
    x_dot: Callable[[float, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Advance a state one step with the classic 4th-order Runge-Kutta method.

    Parameters
    ----------
    t : float
        Time at the start of the step [s].
    dt : float
        Step size [s].
    x_prev : np.ndarray
        State at time `t`.
    x_dot : Callable
        Derivative function with signature `x_dot(t, x) -> dx/dt`, returning
        an array with the same shape as `x`.

    Returns
    -------
    np.ndarray
        State at time `t + dt`.
    """
    if not isinstance(x_prev, np.ndarray):
        raise TypeError("x_prev must be a NumPy array")

    k1 = x_dot(t, x_prev)

    if not isinstance(k1, np.ndarray):
        raise TypeError("Result of x_dot must be a NumPy array")

    k2 = x_dot(t + dt / 2, x_prev + dt * k1 / 2)
    k3 = x_dot(t + dt / 2, x_prev + dt * k2 / 2)
    k4 = x_dot(t + dt, x_prev + dt * k3)

    x_n = x_prev + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_n
