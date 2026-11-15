"""Minimal relative-navigation simulation: truth, sensor, and the loop around the
flight computer.

The simulator owns the truth. Each step it propagates the true relative state,
measures it with each sensor that fires, and hands only `(t, meas)` to the flight
computer. It reads the flight computer's estimate back as telemetry
"""

import numpy as np

from cesium.math import rk4_func
from cesium.orbital.hcw import calc_stm, hcw_dynamics
from cfsw.flight_computer import FlightComputer
from cfsw.measurements import Measurement

from .types import SimResult, SimSensor, TruthPropType

################################################################################
#               Truth propagation
################################################################################


def propagate_truth(
    x: np.ndarray, dt: float, n: float, method: TruthPropType
) -> np.ndarray:
    """True relative state after `dt` [s], by the chosen method."""
    if method is TruthPropType.STM:
        return calc_stm(dt, n) @ x
    if method is TruthPropType.RK4:
        return rk4_func(0.0, dt, x, lambda _t, s: hcw_dynamics(s, np.zeros(3), n))
    raise ValueError(f"unknown truth propagator {method}")


################################################################################
#               Simulation loop
################################################################################


class RelNavSim:
    """Fixed-step loop: propagate truth, measure, `fc.step(t, meas)`, log."""

    def __init__(
        self,
        x0_true: np.ndarray,
        n: float,
        dt: float,
        fc: FlightComputer,
        sensors: list[SimSensor],
        *,
        truth_prop: TruthPropType = TruthPropType.STM,
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
        sensors : list[SimSensor]
            Measurement sources, each with a unique `sensor_id` and its own rate.
        truth_prop : TruthPropType
            Truth propagation method.
        t0 : float
            Start time [s].
        """
        if dt <= 0:
            raise ValueError(f"need dt > 0, got {dt}")
        ids = [s.sensor_id for s in sensors]
        if len(set(ids)) != len(ids):
            raise ValueError(f"sensor ids must be unique, got {ids}")
        self.x_true = np.asarray(x0_true, dtype=float).copy()
        self.n = n
        self.dt = dt
        self.fc = fc
        self.sensors = sensors
        self.truth_prop = truth_prop
        self.t0 = t0

    def run(self, n_steps: int) -> SimResult:
        """Run `n_steps` steps after the initial (update-only) step at `t0`."""
        x = self.x_true.copy()
        ts, xs, x_ests, Ps = [], [], [], []
        zs: dict[str, list[np.ndarray]] = {s.sensor_id: [] for s in self.sensors}
        z_nan = {s.sensor_id: np.full(len(s.h(x)), np.nan) for s in self.sensors}

        for k in range(n_steps + 1):
            t = self.t0 + k * self.dt
            if k > 0:
                x = propagate_truth(x, self.dt, self.n, self.truth_prop)

            meas = [
                Measurement(s.sensor_id, s.measure(x))
                for s in self.sensors
                if s.fires(k)
            ]
            est = self.fc.step(t, meas or None)  # the only contact with flight software

            fired = {m.sensor_id: m.z for m in meas}
            for sensor_id, z_log in zs.items():
                z_log.append(fired.get(sensor_id, z_nan[sensor_id]))
            ts.append(t)
            xs.append(x.copy())
            x_ests.append(est.x)
            Ps.append(est.P)

        self.x_true = x
        return SimResult(
            np.array(ts),
            np.array(xs),
            {sensor_id: np.array(z) for sensor_id, z in zs.items()},
            np.array(x_ests),
            np.array(Ps),
        )
