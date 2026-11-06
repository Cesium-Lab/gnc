from collections.abc import Callable

import numpy as np

# TODO: Different integrators for better ECI-ECEF OR interpolate every second or something


class Simulator:
    """
    Generic fixed-step simulator for any state size and dynamics
    All dynamics-specific details (integrator, transformation, renormalization, etc.) are in
    `step_fn`
    """

    def __init__(
        self,
        state0: np.ndarray,
        t0: float,
        dt: float,
        max_steps: int,
        step_fn: Callable[[float, np.ndarray], np.ndarray],
        stop_funcs: list = None,
    ):
        """
        stop_funcs: optional. List of functions checked against the new state after each step; if any
            returns True, simulate() stops early and trims self.X and self.t
        """
        self.t_curr = t0
        self.dt = dt
        self.max_steps = max_steps
        self.step_fn = step_fn
        self.stop_funcs = stop_funcs if stop_funcs is not None else []

        n = len(state0)
        self.X = np.zeros((max_steps + 1, n))
        self.t = t0 + dt * np.arange(max_steps + 1)
        self.X[0] = state0

    def simulate(self, t: float):
        """Simulates length of t. If t is None then simulates all"""
        final_step = self.max_steps if t is None else int(t / self.dt)
        for step in range(1, self.max_steps + 1):
            self.X[step] = self.step_fn(self.t_curr, self.X[step - 1])
            self.t_curr += self.dt

            if any(stop_func(self.X[step]) for stop_func in self.stop_funcs):
                final_step = step
                break

        self.X = self.X[: final_step + 1]
        self.t = self.t[: final_step + 1]
