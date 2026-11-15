"""`cfsw.measurements`: the flight computer's sensor models and tagged measurements."""

import dataclasses

import numpy as np
import pytest

from cesium.navigation.measurements import meas_rel_pos, meas_rel_pos_jac
from cfsw.measurements import MeasModel, Measurement

################################################################################
#               MeasModel
################################################################################


class Test_MeasModel:
    def test_default_residual_is_subtraction(self) -> None:
        model = MeasModel(meas_rel_pos, meas_rel_pos_jac, np.eye(3))
        z, z_hat = np.array([3.0, -1.0, 2.0]), np.array([1.0, 1.0, 1.0])
        assert model.residual(z, z_hat) == pytest.approx([2.0, -2.0, 1.0])

    def test_custom_residual_is_kept(self) -> None:
        def wrap(z: np.ndarray, z_hat: np.ndarray) -> np.ndarray:
            return (z - z_hat + 180.0) % 360.0 - 180.0

        model = MeasModel(meas_rel_pos, meas_rel_pos_jac, np.eye(1), residual=wrap)
        assert model.residual(np.array([179.0]), np.array([-179.0])) == pytest.approx(
            [-2.0]
        )

    def test_is_frozen(self) -> None:
        """Onboard config can't be swapped out after construction."""
        model = MeasModel(meas_rel_pos, meas_rel_pos_jac, np.eye(3))
        with pytest.raises(dataclasses.FrozenInstanceError):
            model.R = 2 * np.eye(3)  # type: ignore[misc]


################################################################################
#               Measurement
################################################################################


class Test_Measurement:
    def test_is_frozen(self) -> None:
        m = Measurement("rel-pos", np.zeros(3))
        with pytest.raises(dataclasses.FrozenInstanceError):
            m.sensor_id = "radar"  # type: ignore[misc]
