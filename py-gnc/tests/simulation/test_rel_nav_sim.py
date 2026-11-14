"""Closed navigation loop: csim truth + sensor driving the cfsw flight computer."""

import numpy as np
import pytest
from scipy.stats import chi2

from cesium.navigation.kf import Estimate
from cesium.world.constants import MU_MARS, R_MARS
from cfsw.flight_computer import FlightComputer, PositionFcConfig
from csim.sim_types.rel_nav_sim import (
    RelNavSim,
    RelPosSensor,
    SimResult,
    TruthProp,
    propagate_truth,
)

N = np.sqrt(MU_MARS / (R_MARS + 300e3) ** 3)  # 300 km circular Mars orbit [rad/s]
DT = 10.0  # s
X0_TRUE = np.array([100.0, -2000.0, 30.0, 0.05, 0.2, -0.01])
P0 = np.diag([100.0**2] * 3 + [0.2**2] * 3)
X0_ERR = np.array([50.0, -50.0, 20.0, 0.1, -0.1, 0.05])  # ~0.5 sigma per axis


def run_sim_and_fc(
    sensor: RelPosSensor,
    R_filter: np.ndarray,
    n_steps: int,
    *,
    x0_est: np.ndarray | None = None,
    truth_prop: TruthProp = TruthProp.STM,
    meas_every: int = 1,
) -> SimResult:
    x0_est = X0_TRUE + X0_ERR if x0_est is None else x0_est
    fc = FlightComputer(0.0, Estimate(x0_est, P0.copy()), PositionFcConfig(N, R_filter))
    sim = RelNavSim(
        X0_TRUE, N, DT, fc, sensor, truth_prop=truth_prop, meas_interval=meas_every
    )
    return sim.run(n_steps)


def nees(res: SimResult, k: int) -> float:
    """Normalized estimation error squared at step k: e^T P^-1 e (chi-square, 6 dof)."""
    e = res.err[k]
    return float(e @ np.linalg.solve(res.P[k], e))


################################################################################
#               Truth propagation
################################################################################


class Test_truth_propagation:
    def test_rk4_agrees_with_stm(self) -> None:
        """Two independent propagators of the same dynamics agree over one step."""
        x_stm = propagate_truth(X0_TRUE, DT, N, TruthProp.STM)
        x_rk4 = propagate_truth(X0_TRUE, DT, N, TruthProp.RK4)
        assert x_rk4 == pytest.approx(x_stm, rel=1e-10, abs=1e-10)


################################################################################
#               Perfect sensor (sanity check)
################################################################################


class Test_perfect_sensor:
    R_FILTER = 1e-6 * np.eye(3)  # small but nonzero, so S stays invertible

    def test_stm_truth_estimate_converges_to_truth(self) -> None:
        """Same model on both sides and no noise: the estimate locks onto the truth."""
        res = run_sim_and_fc(RelPosSensor(), self.R_FILTER, 200)
        assert res.err[-1, 0:3] == pytest.approx(np.zeros(3), abs=1e-6)  # m
        assert res.err[-1, 3:6] == pytest.approx(np.zeros(3), abs=1e-9)  # m/s

    def test_rk4_truth_estimate_converges_within_integration_error(self) -> None:
        """RK4 truth vs STM filter: a tiny modeling/propagation mismatch, so a looser but still tight lock."""
        res = run_sim_and_fc(
            RelPosSensor(), self.R_FILTER, 2000, truth_prop=TruthProp.RK4
        )
        assert res.err[-1, 0:3] == pytest.approx(np.zeros(3), abs=1e-4)
        assert res.err[-1, 3:6] == pytest.approx(np.zeros(3), abs=1e-7)

    def test_first_step_position_equals_measurement(self) -> None:
        """At t0 the flight computer only updates: position jumps to the (perfect) z."""
        res = run_sim_and_fc(RelPosSensor(), self.R_FILTER, 0)
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
            x0_est = X0_TRUE + rng.multivariate_normal(np.zeros(6), P0)
            res = run_sim_and_fc(
                RelPosSensor(self.R, rng), self.R, max(checkpoints), x0_est=x0_est
            )
            nees_runs[m] = [nees(res, k) for k in checkpoints]

        # within chi squared bounds
        lo, hi = chi2.ppf([0.0005, 0.9995], 6 * M) / M
        assert np.all((lo < nees_runs.mean(axis=0)) & (nees_runs.mean(axis=0) < hi))

    def test_covariance_converges(self) -> None:
        """With Q = 0, P keeps shrinking: after 300 measurements the position variance
        is far below that of the first update (~ one measurement's R per axis)."""
        res = run_sim_and_fc(
            RelPosSensor(self.R, np.random.default_rng(0)), self.R, 300
        )
        tr = np.trace(res.P[:, 0:3, 0:3], axis1=1, axis2=2)
        assert tr[-1] < 0.1 * tr[0]
        assert np.all(np.diff(tr[100:]) <= 0)

    def test_estimate_beats_raw_measurement(self) -> None:
        """Filtering many measurements should be far better than any single one."""
        sensor = RelPosSensor(self.R, np.random.default_rng(3))
        res = run_sim_and_fc(
            sensor, self.R, 500
        )

        # Over 500 measurements
        end = slice(100, None)  # skip convergence from P0
        meas_err = res.z[end] - res.x_true[end, 0:3]
        est_err = res.err[end, 0:3]

        rms_meas = np.sqrt(np.nanmean(np.sum(meas_err**2, axis=1)))  # ~ sqrt(trace R) = 3.46 m
        rms_est = np.sqrt(np.mean(np.sum(est_err**2, axis=1)))
        assert rms_est < 0.2 * rms_meas  # 5x better



################################################################################
#               Measurement rate (z = None steps)
################################################################################


class Test_measurement_rate:
    def test_uncertainty_sawtooth_between_measurements(self) -> None:
        """Measure every 5th step: along-track variance grows on predict-only steps
        and drops at each measurement."""
        R = 4.0 * np.eye(3)
        res = run_sim_and_fc(
            RelPosSensor(R, np.random.default_rng(0)), R, 300, meas_every=5
        )
        var_s = res.P[:, 1, 1]
        for k in range(25, 300):  # has grown significantly enough
            if k % 5 == 0:
                assert var_s[k] < var_s[k - 1]  # measurement step
            else:
                assert var_s[k] > var_s[k - 1]  # predict only

    def test_missing_measurements_logged_as_nan(self) -> None:
        res = run_sim_and_fc(RelPosSensor(), 1e-6 * np.eye(3), 10, meas_every=5)
        has_z = ~np.isnan(res.z).any(axis=1)
        assert has_z.tolist() == [k % 5 == 0 for k in range(11)]


################################################################################
#               Result format
################################################################################


class Test_result:
    def test_shapes_and_times(self) -> None:
        res = run_sim_and_fc(RelPosSensor(), 1e-6 * np.eye(3), 20)
        assert res.t == pytest.approx(DT * np.arange(21))
        assert res.x_true.shape == (21, 6)
        assert res.z.shape == (21, 3)
        assert res.x_est.shape == (21, 6)
        assert res.P.shape == (21, 6, 6)

    def test_noisy_sensor_requires_rng(self) -> None:
        with pytest.raises(ValueError, match="rng"):
            RelPosSensor(R_true=np.eye(3))
