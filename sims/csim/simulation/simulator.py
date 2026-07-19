import numpy as np
from numpy.linalg import norm
from typing import Callable

from ..entities import Spacecraft
from ..math import rk4_func, quat_apply, unit
from ..physics.rigid_body import rigid_body_derivative, RigidBodyParams
from ..physics import grav_accel

# TODO: Different integrators for better ECI-ECEF OR interpolate every second or something

class Simulator:
    """
    Generic fixed-step simulator for any state size and dynamics
    All dynamics-specific details (integrator, transformation, renormalization, etc.) are in 
    `step_fn`
    """

    def __init__(self, state0: np.ndarray, t0: float, dt: float, n_steps: int,
                 step_fn: Callable[[float, np.ndarray], np.ndarray],
                 stop_fn: Callable[[np.ndarray], bool] = None):
        """
        stop_fn: optional. Checked against the new state after each step; if it
            returns True, simulate() stops early and trims self.X and self.t
        """
        self.t_curr = t0
        self.dt = dt
        self.n_steps = n_steps
        self.step_fn = step_fn
        self.stop_fn = stop_fn

        n = len(state0)
        self.X = np.zeros((n_steps+1, n))
        self.t = t0 + dt*np.arange(n_steps+1)
        self.X[0] = state0

    def simulate(self):
        final_step = self.n_steps
        for step in range(1, self.n_steps + 1):
            self.X[step] = self.step_fn(self.t_curr, self.X[step - 1])
            self.t_curr += self.dt

            if self.stop_fn is not None and self.stop_fn(self.X[step]):
                final_step = step
                break

        self.X = self.X[:final_step + 1]
        self.t = self.t[:final_step + 1]

# Example step functions

def rigid_body_step_fn(dt: float, spacecraft: Spacecraft) -> Callable[[float, np.ndarray], np.ndarray]:
    """Builds the `step_fn(t, state) -> next_state` for Simulator, matching the
    rigid-body-under-gravity-only behavior this file used to hardcode directly."""
    def step_fn(t, state):
        accel = grav_accel(state[:3])
        params = RigidBodyParams(spacecraft.mass, spacecraft.I, accel * spacecraft.mass, np.zeros(3))
        next_state = rk4_func(t, dt, state, rigid_body_derivative, params)
        next_state[6:10] = unit(next_state[6:10])
        return next_state
    return step_fn
