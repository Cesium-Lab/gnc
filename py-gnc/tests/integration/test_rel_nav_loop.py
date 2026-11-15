"""Closed navigation loop: csim truth + sensors driving the cfsw flight computer.

Does the filter actually work in the loop? Convergence, consistency (NEES) and
measurement-rate behavior. Loop bookkeeping is tested in `tests/csim`.
"""

import numpy as np
import pytest
from scipy.stats import chi2

from cesium.navigation.kf import Estimate
from cesium.navigation.measurements import meas_rel_pos, meas_rel_pos_jac
from cesium.world.constants import MU_MARS, R_MARS
from cfsw.flight_computer import FlightComputer, PositionFcConfig
from cfsw.measurements import MeasModel
from csim.rel_nav_sim import RelNavSim
from csim.types import SimResult, SimSensor, TruthPropType

N = np.sqrt(MU_MARS / (R_MARS + 300e3) ** 3)  # 300 km circular Mars orbit [rad/s]
DT = 10.0  # s
X0_TRUE = np.array([100.0, -2000.0, 30.0, 0.05, 0.2, -0.01])
P0 = np.diag([100.0**2] * 3 + [0.2**2] * 3)
X0_ERR = np.array([50.0, -50.0, 20.0, 0.1, -0.1, 0.05])  # ~0.5 sigma per axis
ID = "rel-pos"


def rel_pos_sensor(
    R_true: np.ndarray | None = None, seed: int | None = None, every: int = 1
) -> SimSensor:
    """Truth-side relative position sensor; noisy when `R_true` is given."""
    rng = None if seed is None else np.random.default_rng(seed)
    return SimSensor(ID, meas_rel_pos, R_true, rng, every)


def run_test_sim_and_fc(
    sensor: SimSensor,
    R_filter: np.ndarray,
    n_steps: int,
    *,
    x0_est: np.ndarray | None = None,
    truth_prop: TruthPropType = TruthPropType.STM,
) -> SimResult:
    """One sensor in the sim; the flight computer models it with `R_filter`."""
    x0_est = X0_TRUE + X0_ERR if x0_est is None else x0_est
    models = {sensor.sensor_id: MeasModel(meas_rel_pos, meas_rel_pos_jac, R_filter)}
    fc = FlightComputer(0.0, Estimate(x0_est, P0.copy()), PositionFcConfig(N), models)
    sim = RelNavSim(X0_TRUE, N, DT, fc, [sensor], truth_prop=truth_prop)
    return sim.run(n_steps)


def nees(res: SimResult, k: int) -> float:
    """Normalized estimation error squared at step k: e^T P^-1 e (chi-square, 6 dof)."""
    e = res.err[k]
    return float(e @ np.linalg.solve(res.P[k], e))


################################################################################
#               Perfect sensor (sanity check)
################################################################################


class Test_perfect_sensor:
    R_FILTER = 1e-6 * np.eye(3)  # small but nonzero, so S stays invertible

    def test_stm_truth_estimate_converges_to_truth(self) -> None:
        """Same model on both sides and no noise: the estimate locks onto the truth."""
        res = run_test_sim_and_fc(rel_pos_sensor(), self.R_FILTER, 200)
        assert res.err[-1, 0:3] == pytest.approx(np.zeros(3), abs=1e-6)  # m
        assert res.err[-1, 3:6] == pytest.approx(np.zeros(3), abs=1e-9)  # m/s

    def test_rk4_truth_estimate_converges_within_integration_error(self) -> None:
        """RK4 truth vs STM filter: a tiny modeling/propagation mismatch, so a looser
        but still tight lock."""
        res = run_test_sim_and_fc(
            rel_pos_sensor(), self.R_FILTER, 2000, truth_prop=TruthPropType.RK4
        )
        assert res.err[-1, 0:3] == pytest.approx(np.zeros(3), abs=1e-4)
        assert res.err[-1, 3:6] == pytest.approx(np.zeros(3), abs=1e-7)

    def test_first_step_position_equals_measurement(self) -> None:
        """At t0 the flight computer only updates: position jumps to the (perfect) z."""
        res = run_test_sim_and_fc(rel_pos_sensor(), self.R_FILTER, 0)
        assert res.x_est[0, 0:3] == pytest.approx(X0_TRUE[0:3], abs=1e-3)


################################################################################
#               Noisy sensor (consistency)
################################################################################


class Test_noisy_sensor:
    R = 4.0 * np.eye(3)  # (2 m)^2, true and assumed

    def test_monte_carlo_nees_within_chi2_bounds(self) -> None:
        """M seeded Monte Carlo NEES tests.

        Mahalanobis Distance

        References
        ----------
        Navigation Best Practices, Section 10.1
        """
        M, checkpoints = 40, [0, 10, 100, 300]
        nees_runs = np.zeros((M, len(checkpoints)))
        for m in range(M):
            rng = np.random.default_rng(m)
            sensor = SimSensor(ID, meas_rel_pos, self.R, rng)
            x0_est = X0_TRUE + rng.multivariate_normal(np.zeros(6), P0)
            res = run_test_sim_and_fc(sensor, self.R, max(checkpoints), x0_est=x0_est)
            nees_runs[m] = [nees(res, k) for k in checkpoints]

        # within chi squared bounds
        lo, hi = chi2.ppf([0.0005, 0.9995], 6 * M) / M
        assert np.all((lo < nees_runs.mean(axis=0)) & (nees_runs.mean(axis=0) < hi))

    def test_covariance_converges(self) -> None:
        """With Q = 0, P keeps shrinking: after 300 measurements the position variance
        is far below that of the first update (~ one measurement's R per axis)."""
        res = run_test_sim_and_fc(rel_pos_sensor(self.R, seed=0), self.R, 300)
        tr = np.trace(res.P[:, 0:3, 0:3], axis1=1, axis2=2)
        assert tr[-1] < 0.1 * tr[0]
        assert np.all(np.diff(tr[100:]) <= 0)

    def test_estimate_beats_raw_measurement(self) -> None:
        """Filtering many measurements should be far better than any single one."""
        res = run_test_sim_and_fc(rel_pos_sensor(self.R, seed=3), self.R, 500)

        # Over 500 measurements
        end = slice(100, None)  # skip convergence from P0
        meas_err = res.z[ID][end] - res.x_true[end, 0:3]
        est_err = res.err[end, 0:3]

        rms_meas = np.sqrt(
            np.nanmean(np.sum(meas_err**2, axis=1))
        )  # ~ sqrt(trace R) = 3.46 m
        rms_est = np.sqrt(np.mean(np.sum(est_err**2, axis=1)))
        assert rms_est < 0.2 * rms_meas  # 5x better


################################################################################
#               Measurement rate (steps with no measurement)
################################################################################


class Test_measurement_rate:
    def test_uncertainty_sawtooth_between_measurements(self) -> None:
        """Measure every 5th step: along-track variance grows on predict-only steps
        and drops at each measurement."""
        R = 4.0 * np.eye(3)
        res = run_test_sim_and_fc(rel_pos_sensor(R, seed=0, every=5), R, 300)
        var_s = res.P[:, 1, 1]
        for k in range(25, 300):  # has grown significantly enough
            if k % 5 == 0:
                assert var_s[k] < var_s[k - 1]  # measurement step
            else:
                assert var_s[k] > var_s[k - 1]  # predict only


################################################################################
#               Multiple sensors
################################################################################


class Test_multiple_sensors:
    def test_two_perfect_sensors_at_different_rates_converge(self) -> None:
        R = 1e-6 * np.eye(3)
        fast = SimSensor("fast", meas_rel_pos)
        slow = SimSensor("slow", meas_rel_pos, every=3)
        models = {
            "fast": MeasModel(meas_rel_pos, meas_rel_pos_jac, R),
            "slow": MeasModel(meas_rel_pos, meas_rel_pos_jac, R),
        }
        fc = FlightComputer(
            0.0, Estimate(X0_TRUE + X0_ERR, P0), PositionFcConfig(N), models
        )
        res = RelNavSim(X0_TRUE, N, DT, fc, [fast, slow]).run(200)
        assert res.err[-1, 0:3] == pytest.approx(np.zeros(3), abs=1e-6)
