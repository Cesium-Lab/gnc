"""Flight computer on its own: fed synthetic measurements, no simulator."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from cesium.navigation.kf import Estimate, KfUpdate
from cesium.orbital.hcw import hcw_dynamics
from cfsw.flight_computer import FlightComputer, PositionFcConfig

N = 1.1e-3  # rad/s
X0 = np.array([100.0, -2000.0, 30.0, 0.05, 0.2, -0.01])
P0 = np.diag([100.0**2] * 3 + [0.2**2] * 3)
CONFIG = PositionFcConfig(n=N, R=np.eye(3))


def make_fc(t0: float = 0.0, config: PositionFcConfig = CONFIG) -> FlightComputer:
    return FlightComputer(t0, Estimate(X0.copy(), P0.copy()), config)


################################################################################
#               First step (update only)
################################################################################


class Test_first_step:
    def test_step_at_t0_updates_without_propagating(self) -> None:
        """Same time as initialization: no propagation yet"""
        fc = make_fc(config=PositionFcConfig(n=N, R=1e-12 * np.eye(3)))
        z = X0[0:3] + np.array([5.0, -5.0, 2.0])
        est = fc.step(0.0, z)
        assert est.x[0:3] == pytest.approx(z, abs=1e-6)
        assert est.x[3:6] == pytest.approx(
            X0[3:6]
        )  # velocity estimate shouldn't change
        assert fc.t == 0.0

    def test_step_at_t0_without_measurement_changes_nothing(self) -> None:
        est = make_fc().step(0.0, None)
        assert est.x == pytest.approx(X0, abs=0)
        assert est.P == pytest.approx(P0, abs=0)


################################################################################
#               Predict only (z = None)
################################################################################


class Test_predict_only:
    def test_mean_matches_numerical_integration(self) -> None:
        """No measurement: the mean follows HCW, checked against solve_ivp on
        `hcw_dynamics` (independent of the closed-form STM)."""
        est = make_fc().step(600.0, None)
        sol = solve_ivp(
            lambda _t, s: hcw_dynamics(s, np.zeros(3), N),
            (0.0, 600.0),
            X0,
            rtol=1e-12,
            atol=1e-12,
        )
        assert est.x == pytest.approx(sol.y[:, -1], rel=1e-9, abs=1e-9)

    def test_two_short_steps_equal_one_long_step(self) -> None:
        """Without process noise, propagating 300 s twice equals 600 s once."""
        fc_a, fc_b = make_fc(), make_fc()
        fc_a.step(300.0, None)
        est_a = fc_a.step(600.0, None)
        est_b = fc_b.step(600.0, None)
        assert est_a.x == pytest.approx(est_b.x, rel=1e-10)
        assert est_a.P == pytest.approx(est_b.P, rel=1e-10)

    def test_uncertainty_grows_without_measurements(self) -> None:
        """Velocity uncertainty smears into along-track position as time passes."""
        est = make_fc().step(600.0, None)
        assert est.P[1, 1] > P0[1, 1]

    def test_process_noise_scales_with_dt(self) -> None:
        """First order integration does this"""
        Q_rate = 1e-6 * np.eye(6)
        cfg_q = PositionFcConfig(n=N, R=np.eye(3), Q_rate=Q_rate)
        P_with = make_fc(config=cfg_q).step(10.0, None).P
        P_without = make_fc().step(10.0, None).P
        assert P_with - P_without == pytest.approx(10.0 * Q_rate, rel=1e-9)


################################################################################
#               Update
################################################################################


class Test_update:
    def test_measurement_shrinks_position_uncertainty(self) -> None:
        fc = make_fc()
        P_pred = fc.step(60.0, None).P
        P_post = fc.step(60.0, X0[0:3]).P  # same time: update only
        assert np.all(np.diag(P_post)[0:3] < np.diag(P_pred)[0:3])

    def test_records_last_update_for_telemetry(self) -> None:
        fc = make_fc()
        assert fc.last_update is None
        z = X0[0:3] + 1.0
        fc.step(0.0, z)
        assert isinstance(fc.last_update, KfUpdate)
        assert fc.last_update.innovation == pytest.approx(np.ones(3))


################################################################################
#               Interface rules
################################################################################


class Test_interface:
    def test_rejects_time_going_backwards(self) -> None:
        fc = make_fc(t0=100.0)
        with pytest.raises(ValueError, match="backwards"):
            fc.step(99.0, None)

    def test_returned_estimate_is_a_copy(self) -> None:
        """Telemetry is read-only: editing what step() returns can't change onboard state."""
        fc = make_fc()
        est = fc.step(0.0, None)
        est.x[:] = 0.0
        est.P[:] = 0.0
        assert fc.estimate.x == pytest.approx(X0, abs=0)
        assert fc.estimate.P == pytest.approx(P0, abs=0)

        x0, p0 = X0.copy(), P0.copy()
        fc = FlightComputer(0.0, Estimate(x0, p0), CONFIG)
        x0[:] = 0.0
        assert fc.estimate.x == pytest.approx(X0, abs=0)
