import sys
import numpy as np

sys.path.append(".")

from csim.math.integrators import rk4_func

#################################################################
#               Constant
#################################################################


def constant_deriv(t: float, x: np.ndarray | list, _):
    return np.zeros_like(x)


def test_constant():
    x0 = [1, -2, 3.5, 100.3, 12.3]
    dt = 1
    t = 0
    x1 = rk4_func(t, dt, x0, constant_deriv)

    assert np.array_equal(x1, x0)


#################################################################
#               Constant Slope
#################################################################


def slope_2(t: float, x: np.ndarray | list, _):
    return np.array([2, 4])


def test_constant_slope():
    x0 = [1, 2]
    dt = 2
    t = 0
    x1 = rk4_func(t, dt, x0, slope_2)

    # x1 = x0 + (dx/dt)*dt
    # Since dx/dt = [2,4], exact solution: x = [1,2] + [2,4]*2 = [5,10]
    assert np.array_equal(x1, [5, 10])


#################################################################
#               Exponential Decay
#################################################################


def e_decay_deriv(t: float, x: np.ndarray | list, _):
    return -x


def test_exponential_decay():
    y0 = np.array([4, 10])  # Starts at y=4
    dt = 0.2
    t = 0
    y1 = rk4_func(t, dt, y0, e_decay_deriv)

    # Simple exponential decay function where the derivative is y
    # y1 = y0 * exp(-dt)
    assert np.allclose(y1, y0 * np.exp(-dt), rtol=1e-3)


def test_exponential_decay_multistep():
    y0 = np.array([4, 10])  # Starts at y=4
    dt = 0.005
    t_max = 10
    t = 0

    y = y0
    while t < t_max:
        y = rk4_func(t, dt, y, e_decay_deriv)
        t += dt

    # Simple exponential decay function where the derivative is y
    # y1 = y0 * exp(-t_max)
    assert np.allclose(y, y0 * np.exp(-t_max), rtol=1e-3)


#################################################################
#               Sine!
#################################################################


def sine_deriv(t: float, x: np.ndarray | list, _):
    return np.cos(t)


def test_sine():
    y0 = np.array([4, 10])  # Starts at y=4
    dt = 0.1
    t = 0
    y1 = rk4_func(t, dt, y0, sine_deriv)

    # Basically the functions:
    # sin(x) + 4
    # sin(x) + 10
    assert np.allclose(y1, y0 + np.sin(dt), rtol=1e-2)


def test_sine_multistep():
    y0 = np.array([4, 10])  # Starts at y=4
    dt = 0.001
    t_max = 50
    t = 0

    y = y0
    while t < t_max:  # 50,000 steps
        y = rk4_func(t, dt, y, sine_deriv)
        t += dt

    # Basically the functions:
    # sin(x) + 4
    # sin(x) + 10
    assert np.allclose(y, y0 + np.sin(t_max), rtol=1e-3)


#################################################################
#               Logistic
#################################################################


def logistic_deriv(t: float, x: np.ndarray | list, _):
    rate = 1
    k = 10  # asymptote
    return rate * x * (1 - x / k)


def test_logistic():
    y0 = np.array([0.001, 0.01, 0.1])  # Starts at y=4
    dt = 0.001
    t_max = 15
    t = 0

    y = y0
    while t < t_max:  # 50,000 steps
        y = rk4_func(t, dt, y, logistic_deriv)
        t += dt

    # Logistic function reaches asymptote
    assert np.allclose(y, [10, 10, 10], rtol=1e-2)
