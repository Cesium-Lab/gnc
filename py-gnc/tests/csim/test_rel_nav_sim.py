"""`csim.rel_nav_sim`: truth propagation and the loop's bookkeeping.

Whether the filter converges in the loop is tested in
`tests/integration/test_rel_nav_loop.py`.
"""

import numpy as np
import pytest

from cesium.navigation.kf import Estimate
from cesium.navigation.measurements import meas_rel_pos, meas_rel_pos_jac
from cfsw.flight_computer import FlightComputer, PositionFcConfig
from cfsw.measurements import MeasModel
from csim.rel_nav_sim import RelNavSim, propagate_truth
from csim.types import SimResult, SimSensor, TruthPropType

N = 1.1e-3  # rad/s
DT = 10.0  # s
X0_TRUE = np.array([100.0, -2000.0, 30.0, 0.05, 0.2, -0.01])
P0 = np.diag([100.0**2] * 3 + [0.2**2] * 3)
R = 1e-6 * np.eye(3)


def run_sim(sensors: list[SimSensor], n_steps: int) -> SimResult:
    """Run the loop with a flight computer that models every sensor as rel-pos."""
    models = {
        s.sensor_id: MeasModel(meas_rel_pos, meas_rel_pos_jac, R) for s in sensors
    }
    fc = FlightComputer(0.0, Estimate(X0_TRUE, P0), PositionFcConfig(N), models)
    return RelNavSim(X0_TRUE, N, DT, fc, sensors).run(n_steps)


################################################################################
#               Truth propagation
################################################################################


class Test_propagate_truth:
    def test_rk4_agrees_with_stm(self) -> None:
        """Two independent propagators of the same dynamics agree over one step."""
        x_stm = propagate_truth(X0_TRUE, DT, N, TruthPropType.STM)
        x_rk4 = propagate_truth(X0_TRUE, DT, N, TruthPropType.RK4)
        assert x_rk4 == pytest.approx(x_stm, rel=1e-10, abs=1e-10)


################################################################################
#               Result format
################################################################################


class Test_result:
    def test_shapes_and_times(self) -> None:
        res = run_sim([SimSensor("rel-pos", meas_rel_pos)], 20)
        assert res.t == pytest.approx(DT * np.arange(21))
        assert res.x_true.shape == (21, 6)
        assert res.z["rel-pos"].shape == (21, 3)
        assert res.x_est.shape == (21, 6)
        assert res.P.shape == (21, 6, 6)

    def test_logged_truth_matches_propagation(self) -> None:
        res = run_sim([SimSensor("rel-pos", meas_rel_pos)], 3)
        expected = propagate_truth(X0_TRUE, 3 * DT, N, TruthPropType.STM)
        assert res.x_true[-1] == pytest.approx(expected, rel=1e-12)

    def test_err_is_estimate_minus_truth(self) -> None:
        res = run_sim([SimSensor("rel-pos", meas_rel_pos)], 5)
        assert res.err == pytest.approx(res.x_est - res.x_true, abs=0)


################################################################################
#               Measurement logging
################################################################################


class Test_measurement_logging:
    def test_missing_measurements_logged_as_nan(self) -> None:
        res = run_sim([SimSensor("rel-pos", meas_rel_pos, every=5)], 10)
        has_z = ~np.isnan(res.z["rel-pos"]).any(axis=1)
        assert has_z.tolist() == [k % 5 == 0 for k in range(11)]

    def test_each_sensor_logged_at_its_own_rate(self) -> None:
        fast = SimSensor("fast", meas_rel_pos)
        slow = SimSensor("slow", meas_rel_pos, every=3)
        res = run_sim([fast, slow], 9)
        assert set(res.z) == {"fast", "slow"}
        assert (~np.isnan(res.z["fast"]).any(axis=1)).all()
        assert (~np.isnan(res.z["slow"]).any(axis=1)).tolist() == [
            k % 3 == 0 for k in range(10)
        ]

    def test_perfect_measurement_logged_equals_truth(self) -> None:
        res = run_sim([SimSensor("rel-pos", meas_rel_pos)], 5)
        assert res.z["rel-pos"] == pytest.approx(res.x_true[:, 0:3], abs=0)


################################################################################
#               Construction
################################################################################


class Test_construction:
    def test_sensor_ids_must_be_unique(self) -> None:
        fc = FlightComputer(0.0, Estimate(X0_TRUE, P0), PositionFcConfig(N))
        sensors = [SimSensor("a", meas_rel_pos), SimSensor("a", meas_rel_pos)]
        with pytest.raises(ValueError, match="unique"):
            RelNavSim(X0_TRUE, N, DT, fc, sensors)

    @pytest.mark.parametrize("dt", [0.0, -1.0])
    def test_rejects_nonpositive_dt(self, dt: float) -> None:
        fc = FlightComputer(0.0, Estimate(X0_TRUE, P0), PositionFcConfig(N))
        with pytest.raises(ValueError, match="dt"):
            RelNavSim(X0_TRUE, N, dt, fc, [])
