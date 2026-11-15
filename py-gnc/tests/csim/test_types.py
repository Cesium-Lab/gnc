"""`csim.types`: the truth-side sensor on its own."""

import numpy as np
import pytest

from cesium.navigation.measurements import meas_range_az_el, meas_rel_pos
from csim.types import SimSensor

X = np.array([100.0, -2000.0, 30.0, 0.05, 0.2, -0.01])


################################################################################
#               Measuring
################################################################################


class Test_measure:
    def test_perfect_sensor_returns_h(self) -> None:
        sensor = SimSensor("rel-pos", meas_rel_pos)
        assert sensor.measure(X) == pytest.approx(X[0:3], abs=0)

    def test_does_not_modify_truth(self) -> None:
        x = X.copy()
        sensor = SimSensor("rel-pos", meas_rel_pos, np.eye(3), np.random.default_rng(0))
        sensor.measure(x)
        assert x == pytest.approx(X, abs=0)

    def test_noise_has_true_covariance(self) -> None:
        """Sample mean ~ 0 and sample covariance ~ R_true over many draws."""
        R_true = np.diag([1.0, 4.0, 9.0])
        sensor = SimSensor("rel-pos", meas_rel_pos, R_true, np.random.default_rng(0))
        noise = np.array([sensor.measure(X) - X[0:3] for _ in range(20000)])
        assert noise.mean(axis=0) == pytest.approx(np.zeros(3), abs=0.05)
        assert np.cov(noise.T) == pytest.approx(R_true, rel=0.05, abs=0.05)

    def test_same_seed_same_noise(self) -> None:
        a = SimSensor("rel-pos", meas_rel_pos, np.eye(3), np.random.default_rng(7))
        b = SimSensor("rel-pos", meas_rel_pos, np.eye(3), np.random.default_rng(7))
        assert a.measure(X) == pytest.approx(b.measure(X), abs=0)

    def test_noise_size_follows_measurement_size(self) -> None:
        """Not hard-coded to 3: works for any M (here range/az/el, M = 3 but a
        different h) and a 1-D measurement."""
        rae = SimSensor(
            "rae", meas_range_az_el, 1e-6 * np.eye(3), np.random.default_rng(0)
        )
        assert rae.measure(X).shape == (3,)
        rng_only = SimSensor(
            "range",
            lambda x: np.array([np.linalg.norm(x[0:3])]),
            np.eye(1),
            np.random.default_rng(0),
        )
        assert rng_only.measure(X).shape == (1,)


################################################################################
#               Rate
################################################################################


class Test_fires:
    def test_every_step_by_default(self) -> None:
        sensor = SimSensor("rel-pos", meas_rel_pos)
        assert all(sensor.fires(k) for k in range(10))

    def test_every_nth_step(self) -> None:
        sensor = SimSensor("rel-pos", meas_rel_pos, every=4)
        assert [sensor.fires(k) for k in range(17)] == [k % 4 == 0 for k in range(17)]


################################################################################
#               Validation
################################################################################


class Test_validation:
    def test_noisy_sensor_requires_rng(self) -> None:
        with pytest.raises(ValueError, match="rng"):
            SimSensor("rel-pos", meas_rel_pos, R_true=np.eye(3))

    @pytest.mark.parametrize("every", [0, -1])
    def test_rejects_nonpositive_every(self, every: int) -> None:
        with pytest.raises(ValueError, match="every"):
            SimSensor("rel-pos", meas_rel_pos, every=every)
