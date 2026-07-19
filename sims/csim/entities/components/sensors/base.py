"""Base class for RPO sensors.

All relative-navigation sensors here work off the truth relative state in the target's
Hill/RSW frame `rho = [x,y,z,xdot,ydot,zdot]` (see `csim.physics.relative_motion`), not
raw ECI states -- `Spacecraft.sense()` does that frame conversion once and hands each
sensor the same `rho`.

A `Sensor` bundles two things that must never drift apart:
- the truth-side noisy measurement generator (`measure`)
- the analytic observation model (`h`/`H`) an EKF (see `estimation.filtering.ekf`) uses
  to predict that same measurement from an *estimated* state
"""

import numpy as np


class Sensor:
    def __init__(self, R: np.ndarray, rng: np.random.Generator | None = None):
        """
        Args:
            R (np.ndarray): Measurement noise covariance [MxM]
            rng (np.random.Generator, optional): Defaults to `np.random.default_rng()`.
        """
        self.R = np.asarray(R)
        self.rng = rng if rng is not None else np.random.default_rng()

    def h(self, rho: np.ndarray) -> np.ndarray:
        """Observation model: expected measurement given a (estimated or truth) relative state [Mx1]"""
        raise NotImplementedError

    def H(self, rho: np.ndarray) -> np.ndarray:
        """Jacobian of `h` w.r.t. `rho` [MxN]"""
        raise NotImplementedError

    def in_view(self, rho: np.ndarray) -> bool:
        """Whether the target is within this sensor's range/FOV. Defaults to always true."""
        return True

    def measure(self, rho_truth: np.ndarray):
        """Truth-side: generate a noisy measurement from the true relative state.

        Args:
            rho_truth (np.ndarray): True relative state `[x,y,z,xdot,ydot,zdot]` (Hill frame) [m],[m/s]

        Returns:
            tuple: `(z, valid)` -- `z` (np.ndarray or None) is the noisy measurement,
                None when `valid` is False (out of FOV/range)
        """
        if not self.in_view(rho_truth):
            return None, False

        z_true = self.h(rho_truth)
        noise = self.rng.multivariate_normal(np.zeros(len(z_true)), self.R)
        return z_true + noise, True
