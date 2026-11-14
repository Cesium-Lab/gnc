"""Kalman filter math (`cesium.navigation.kf`): hand calculations and algebraic identities."""

from collections.abc import Callable

import numpy as np
import pytest

from cesium.navigation.kf import Estimate, kf_propagate_cov, kf_update
from cesium.navigation.measurements import meas_rel_pos, meas_rel_pos_jac


def linear_h(
    H: np.ndarray,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Measurement model and Jacobian for a constant matrix `H`."""

    def meas(x):
        return H @ x

    def jac(x):
        return H

    return meas, jac


################################################################################
#               Covariance propagation
################################################################################


class Test_kf_propagate_cov:
    def test_hand_calc_constant_velocity(self) -> None:
        """P = I, Phi = [[1, 1], [0, 1]] (1 s of constant velocity). Matlab evaluated"""
        P = np.eye(2)
        stm = np.array([[1.0, 1.0], [0.0, 1.0]])  # position and velocity
        Q = np.zeros((2, 2))
        P = kf_propagate_cov(P, stm, Q)
        assert P == pytest.approx(np.array([[2.0, 1.0], [1.0, 1.0]]))

    def test_adds_process_noise(self) -> None:
        P = stm = np.eye(2)
        Q = np.diag([0.5, 0.25])
        P_next = kf_propagate_cov(stm, stm, Q)
        assert P_next == pytest.approx(P + Q)

    def test_result_is_exactly_symmetric(self) -> None:
        seed = 0
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(6, 6))
        P = kf_propagate_cov(A @ A.T, rng.normal(size=(6, 6)), np.zeros((6, 6)))
        assert np.array_equal(P, P.T)


################################################################################
#               Measurement update
################################################################################


class Test_kf_update:
    def test_scalar_hand_calc(self) -> None:
        """x = 0, P = 4, z = 1, R = 1:
        S = P + R = 5, K = P / S = 0.8, x = 0 + 0.8 * 1 = 0.8,
        P = (1 - K)^2 P + K^2 R = 0.04 * 4 + 0.64 * 1 = 0.8."""
        h, H = linear_h(np.eye(1))

        up = kf_update(
            Estimate(np.array([0.0]), np.array([[4.0]])),
            np.eye(1),
            h,
            H,
            np.eye(1),
        )
        assert up.innovation == pytest.approx([1.0])
        assert up.S == pytest.approx(np.array([[5.0]]))
        assert up.K == pytest.approx(np.array([[0.8]]))
        assert up.est.x == pytest.approx([0.8])
        assert up.est.P == pytest.approx(np.array([[0.8]]))

    def test_correlation_carries_update_to_unmeasured_state(self) -> None:
        """Position-velocity pair, measure position only. x = [0, 0], z = 2, R = 1,
        P = [[1, 0.5], [0.5, 1]]:
        S = 1 + 1 = 2, K = P H^T / S = [1, 0.5] / 2 = [0.5, 0.25]
        x = K * 2 = [1, 0.5]          (velocity moves through the correlation)
        P = P - K H P = [[1, 0.5], [0.5, 1]] - [[0.5, 0.25], [0.25, 0.125]]
          = [[0.5, 0.25], [0.25, 0.875]]
        """
        h, H = linear_h(np.array([[1.0, 0.0]]))
        est = Estimate(np.zeros(2), np.array([[1.0, 0.5], [0.5, 1.0]]))
        up = kf_update(est, np.array([2.0]), h, H, np.array([[1.0]]))
        assert up.est.x == pytest.approx([1.0, 0.5])
        assert up.est.P == pytest.approx(np.array([[0.5, 0.25], [0.25, 0.875]]))

    def test_tiny_R_snaps_measured_states_to_measurement(self) -> None:
        est = Estimate(np.zeros(6), np.diag([100.0, 100.0, 100.0, 1.0, 1.0, 1.0]))
        z = np.array([10.0, -20.0, 5.0])
        up = kf_update(est, z, meas_rel_pos, meas_rel_pos_jac, 1e-12 * np.eye(3))
        assert up.est.x[0:3] == pytest.approx(z, abs=1e-9)
        assert up.est.x[3:6] == pytest.approx(np.zeros(3))  # uncorrelated: untouched
        assert np.diag(up.est.P)[0:3] == pytest.approx(np.zeros(3), abs=1e-9)

    def test_huge_R_ignores_measurement(self) -> None:
        est = Estimate(np.ones(6), np.eye(6))
        up = kf_update(
            est,
            np.array([1e3, 1e3, 1e3]),  # big measurement
            meas_rel_pos,
            meas_rel_pos_jac,
            1e12 * np.eye(3),  # but also big covariance
        )
        assert up.est.x == pytest.approx(est.x, abs=1e-6)
        assert up.est.P == pytest.approx(est.P, abs=1e-9)

    def test_joseph_matches_simple_form_for_optimal_gain(self) -> None:
        """With the optimal K, Joseph form equals (I - K H) P"""
        rng = np.random.default_rng(1)
        A = rng.normal(size=(6, 6))
        est = Estimate(
            rng.normal(size=6), A @ A.T + np.eye(6)
        )  # emulates an update already
        R = np.diag([2.0, 3.0, 4.0])
        up = kf_update(est, rng.normal(size=3), meas_rel_pos, meas_rel_pos_jac, R)
        H = meas_rel_pos_jac(est.x)
        assert up.est.P == pytest.approx((np.eye(6) - up.K @ H) @ est.P, rel=1e-10)

    def test_P_next_is_symmetric_psd_and_improves(self) -> None:
        rng = np.random.default_rng(2)
        A = rng.normal(size=(6, 6))
        est = Estimate(
            np.zeros(6), A @ A.T + 0.1 * np.eye(6)
        )  # emulates an update already
        update = kf_update(
            est, rng.normal(size=3), meas_rel_pos, meas_rel_pos_jac, np.eye(3)
        )
        P = update.est.P
        assert np.array_equal(P, P.T)
        assert np.linalg.eigvalsh(P).min() >= 0
        assert np.linalg.eigvalsh(est.P - P).min() >= -1e-12  # Should get better

    def test_custom_residual_replaces_subtraction(self) -> None:
        """Angle wrapping: z = 179 deg, z_hat = -179 deg is a -2 deg innovation, not 358."""
        h, H = linear_h(np.eye(1))

        def wrap_resid_fn(z: np.ndarray, z_hat: np.ndarray) -> np.ndarray:
            return (z - z_hat + 180.0) % 360.0 - 180.0

        est = Estimate(np.array([-179.0]), np.array([[1.0]]))
        up = kf_update(est, np.array([179.0]), h, H, np.eye(1), wrap_resid_fn)
        assert up.innovation == pytest.approx([-2.0])
        assert up.est.x == pytest.approx([-180.0])  # K = 0.5, moves -1 deg
