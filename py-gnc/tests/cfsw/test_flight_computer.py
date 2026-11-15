"""Flight computer on its own: fed synthetic measurements, no simulator."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from cesium.navigation.kf import Estimate, KfUpdate
from cesium.navigation.measurements import meas_rel_pos, meas_rel_pos_jac
from cesium.orbital.hcw import hcw_dynamics
from cfsw.flight_computer import FlightComputer, PositionFcConfig
from cfsw.measurements import MeasModel, Measurement

N = 1.1e-3  # rad/s
X0 = np.array([100.0, -2000.0, 30.0, 0.05, 0.2, -0.01])
P0 = np.diag([100.0**2] * 3 + [0.2**2] * 3)
CONFIG = PositionFcConfig(n=N)


def rel_pos_models(R: np.ndarray = np.eye(3)) -> dict[str, MeasModel]:
    return {"rel-pos": MeasModel(meas_rel_pos, meas_rel_pos_jac, R)}


def make_fc(
    t0: float = 0.0,
    config: PositionFcConfig = CONFIG,
    models: dict[str, MeasModel] | None = None,
) -> FlightComputer:
    models = rel_pos_models() if models is None else models
    return FlightComputer(t0, Estimate(X0.copy(), P0.copy()), config, models)


################################################################################
#               First step (update only)
################################################################################


class Test_first_step:
    def test_step_at_t0_updates_without_propagating(self) -> None:
        """Same time as initialization: no propagation yet"""
        fc = make_fc(models=rel_pos_models(1e-12 * np.eye(3)))  # Low noise
        z = X0[0:3] + np.array([5.0, -5.0, 2.0])
        est = fc.step(0.0, [Measurement("rel-pos", z)])
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
#               Predict only (no measurements)
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
        cfg_q = PositionFcConfig(n=N, Q_rate=Q_rate)
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
        z = Measurement("rel-pos", X0[0:3])
        P_post = fc.step(60.0, [z]).P  # same time: update only
        assert np.all(np.diag(P_post)[0:3] < np.diag(P_pred)[0:3])

    def test_records_last_update_for_telemetry(self) -> None:
        fc = make_fc()
        assert fc.last_update is None
        z = X0[0:3] + 1.0
        fc.step(0.0, [Measurement("rel-pos", z)])
        assert isinstance(fc.last_update, KfUpdate)
        assert fc.last_update.innovation == pytest.approx(np.ones(3))

    def test_empty_list_is_predict_only(self) -> None:
        est_none = make_fc().step(60.0, None)
        est_empty = make_fc().step(60.0, [])
        assert est_empty.x == pytest.approx(est_none.x, abs=0)
        assert est_empty.P == pytest.approx(est_none.P, abs=0)

    def test_same_time_measurements_equal_sequential_steps(self) -> None:
        """Two measurements in one step = the same two fed in back-to-back steps."""
        models = {
            "lidar": MeasModel(meas_rel_pos, meas_rel_pos_jac, np.eye(3)),
            "radar": MeasModel(meas_rel_pos, meas_rel_pos_jac, 4.0 * np.eye(3)),
        }
        m1 = Measurement("lidar", X0[0:3] + 1.0)
        m2 = Measurement("radar", X0[0:3] - 2.0)

        fc_a = make_fc(models=models)
        est_a = fc_a.step(60.0, [m1, m2])

        fc_b = make_fc(models=models)
        fc_b.step(60.0, [m1])
        est_b = fc_b.step(60.0, [m2])

        assert est_a.x == pytest.approx(est_b.x, rel=1e-12)
        assert est_a.P == pytest.approx(est_b.P, rel=1e-12)

        # just updating m1 != just updating m2
        assert est_a.x != pytest.approx(make_fc(models=models).step(60.0, [m2]).x)

    def test_each_measurement_uses_its_own_R(self) -> None:
        """Same z through a precise and a noisy model: the precise one moves x more."""
        models = {
            "precise": MeasModel(meas_rel_pos, meas_rel_pos_jac, 1e-2 * np.eye(3)),
            "noisy": MeasModel(meas_rel_pos, meas_rel_pos_jac, 1e4 * np.eye(3)),
        }
        z = X0[0:3] + 50.0
        x_precise = make_fc(models=models).step(0.0, [Measurement("precise", z)]).x
        x_noisy = make_fc(models=models).step(0.0, [Measurement("noisy", z)]).x
        assert np.all(np.abs(x_precise[0:3] - X0[0:3]) > np.abs(x_noisy[0:3] - X0[0:3]))

    def test_uses_model_residual(self) -> None:
        """A custom residual function (and therefore innovation) of 0 replaces z - z_hat (e.g. angle wrapping)."""
        models = {
            "rel-pos": MeasModel(
                meas_rel_pos, meas_rel_pos_jac, np.eye(3), residual=lambda z, zh: 0 * z
            )
        }
        fc = make_fc(models=models)
        fc.step(0.0, [Measurement("rel-pos", X0[0:3] + 100.0)])
        assert fc.last_update is not None
        assert fc.last_update.innovation == pytest.approx(np.zeros(3), abs=0)


################################################################################
#               Interface rules
################################################################################


class Test_interface:
    def test_rejects_time_going_backwards(self) -> None:
        fc = make_fc(t0=100.0)
        with pytest.raises(ValueError, match="backwards"):
            fc.step(99.0, None)

    def test_rejects_unknown_sensor(self) -> None:
        with pytest.raises(KeyError, match="UNKNOWN"):
            make_fc().step(0.0, [Measurement("UNKNOWN", np.zeros(3))])

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
