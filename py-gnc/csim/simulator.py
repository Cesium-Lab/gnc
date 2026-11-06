from collections.abc import Callable

import numpy as np

from cesium.math import rk4_func

# TODO: Different integrators for better ECI-ECEF OR interpolate every second or something


class Simulator:
    """
    Fixed-step RK4 simulator for an array state and derivative(t, state).
    """

    def __init__(
        self,
        state0: np.ndarray,
        t0: float,
        dt: float,
        max_steps: int,
        derivative: Callable[[float, np.ndarray], np.ndarray],
        stop_funcs: list = None,
    ):
        """
        stop_funcs: optional. List of functions checked against the new state after each step; if any
            returns True, simulate() stops early and trims self.X and self.t
        """
        if dt <= 0 or max_steps < 0:
            raise ValueError("dt must be positive and max_steps nonnegative")
        self.state0 = state0.copy()
        self.t0 = t0
        self.t_curr = t0
        self.dt = dt
        self.max_steps = max_steps
        self.derivative = derivative
        self.stop_funcs = stop_funcs if stop_funcs is not None else []

    def simulate(self, t_final: float | None = None):
        """Run from the initial state for t seconds, up to max_steps * dt.

        None runs the full duration. A shorter final step reaches t exactly.
        Each call replaces the previous history.
        """
        max_duration = self.max_steps * self.dt
        duration = max_duration if t_final is None else t_final
        if not np.isfinite(duration) or not 0 <= duration <= max_duration:
            raise ValueError("duration must be between 0 and max_steps * dt")

        offsets = np.append(
            np.arange(0.0, duration, self.dt), duration
        )  # includes final time just in case
        self.t = self.t0 + offsets

        self.X = np.zeros((len(self.t), len(self.state0)))
        self.X[0] = self.state0
        self.t_curr = self.t0

        for step in range(1, len(self.t)):
            self.X[step] = rk4_func(
                self.t[step - 1],
                offsets[step] - offsets[step - 1],
                self.X[step - 1],
                self.derivative,
            )
            self.t_curr = self.t[step]

            if any(stop_func(self.X[step]) for stop_func in self.stop_funcs):
                self.X = self.X[: step + 1]
                self.t = self.t[: step + 1]
                break
