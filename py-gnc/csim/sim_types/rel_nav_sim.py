"""Minimal relative-navigation simulation: truth, sensor, and the loop around the
flight computer.

The simulator owns the truth. Each step it propagates the true relative state,
makes a measurement from it, and hands only `(t, z)` to the flight computer.
It reads the flight computer's estimate back as telemetry, never writes it.
"""

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from cesium.math import rk4_func
from cesium.navigation.measurements import meas_rel_pos
from cesium.orbital.hcw import calc_stm, hcw_dynamics
from cfsw.flight_computer import FlightComputer

################################################################################
#               Truth propagation
################################################################################


class TruthProp(Enum):
    """How the simulator propagates the true relative state."""

    STM = auto()
    """Closed-form exact HCW STM."""
    RK4 = auto()
    """Fixed-step RK4 on `hcw_dynamics`"""


def propagate_truth(
    x: np.ndarray, dt: float, n: float, method: TruthProp
) -> np.ndarray:
    """True relative state after `dt` [s], by the chosen method."""
    if method is TruthProp.STM:
        return calc_stm(dt, n) @ x
    if method is TruthProp.RK4:
        return rk4_func(0.0, dt, x, lambda _t, s: hcw_dynamics(s, np.zeros(3), n))
    raise ValueError(f"unknown truth propagator {method}")


################################################################################
#               Sensor
################################################################################


@dataclass
class RelPosSensor:
    """Relative position sensor: `z = meas_rel_pos(x_true) + noise`.

    `R_true=None` is a perfect sensor (no noise). Noise comes from an explicit,
    seedable generator so Monte Carlo runs are reproducible.
    """

    R_true: np.ndarray | None = None
    """True noise covariance (3, 3) [m^2], or None for no noise."""
    rng: np.random.Generator | None = None
    """Noise source; required when `R_true` is set (seed it for repeatable runs)."""

    def __post_init__(self) -> None:
        if self.R_true is not None and self.rng is None:
            raise ValueError("a noisy sensor needs an rng")

    def measure(self, x_true: np.ndarray) -> np.ndarray:
        z = meas_rel_pos(x_true).copy()
        if self.R_true is not None:
            z += self.rng.multivariate_normal(np.zeros(3), self.R_true)
        return z


################################################################################
#               Simulation
################################################################################


@dataclass(frozen=True)
class SimResult:
    """Time histories from `RelNavSim.run`"""

    t: np.ndarray
    """Time (K,) [s]."""
    x_true: np.ndarray
    """True relative states (K, 6) [m, m/s]."""
    z: np.ndarray
    """Measurements (K, 3) [m]; NaN on steps without one."""
    x_est: np.ndarray
    """Flight computer estimates after each step (K, 6)."""
    P: np.ndarray
    """Flight computer covariances after each step (K, 6, 6)."""

    @property
    def err(self) -> np.ndarray:
        """Estimation error `x_est - x_true` (K, 6)."""
        return self.x_est - self.x_true


class RelNavSim:
    """Fixed-step loop: propagate truth, measure, `fc.step(t, z)`, log."""

    def __init__(
        self,
        x0_true: np.ndarray,
        n: float,
        dt: float,
        fc: FlightComputer,
        sensor: RelPosSensor,
        *,
        truth_prop: TruthProp = TruthProp.STM,
        meas_interval: int = 1,
        t0: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        x0_true : np.ndarray (6,)
            True relative state at `t0` in RSW [m, m/s].
        n : float
            True target mean motion [rad/s].
        dt : float
            Step size [s].
        fc : FlightComputer
            Flight computer, already initialized at `t0`.
        sensor : RelPosSensor
            Measurement source.
        truth_prop : TruthProp
            Truth propagation method.
        meas_interval : int
            Measure every this many steps (1 = every step); other steps pass `None`.
        t0 : float
            Start time [s].
        """
        if dt <= 0 or meas_interval < 1:
            raise ValueError("need dt > 0 and meas_interval >= 1")
        self.x_true = np.asarray(x0_true, dtype=float).copy()
        self.n = n
        self.dt = dt
        self.fc = fc
        self.sensor = sensor
        self.truth_prop = truth_prop
        self.meas_interval = meas_interval
        self.t0 = t0

    def run(self, n_steps: int) -> SimResult:
        """Run `n_steps` steps after the initial (update-only) step at `t0`."""
        ts, xs, zs, x_ests, Ps = [], [], [], [], []
        x = self.x_true.copy()

        for k in range(n_steps + 1):
            t = self.t0 + k * self.dt
            if k > 0:
                x = propagate_truth(x, self.dt, self.n, self.truth_prop)

            z = self.sensor.measure(x) if k % self.meas_interval == 0 else None
            est = self.fc.step(t, z)  # the only contact with flight software

            ts.append(t)
            xs.append(x.copy())
            zs.append(z if z is not None else np.full(3, np.nan))
            x_ests.append(est.x)
            Ps.append(est.P)

        self.x_true = x
        return SimResult(
            np.array(ts), np.array(xs), np.array(zs), np.array(x_ests), np.array(Ps)
        )
