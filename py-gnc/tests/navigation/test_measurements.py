"""Measurement models: hand-calculated values, geometry, inversion, and Jacobians
checked against finite differences (independent of the analytic derivation)."""

from collections.abc import Callable

import numpy as np
import pytest

from cesium.constants import DEG_TO_RAD
from cesium.navigation.measurements import (
    meas_range_az_el,
    meas_range_az_el_jac,
    meas_rel_pos,
    meas_rel_pos_jac,
)

# Generic relative state (no zero components): 2 km behind, 300 m below, 50 m out of plane
X_GENERIC = np.array([-300.0, -2000.0, 50.0, 0.1, 0.65, -0.02])


def jac_fd(
    h: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Central-difference Jacobian of `h` at `x` [step eps in m or m/s]."""
    cols = []
    for k in range(x.size):
        dx = np.zeros_like(x)
        dx[k] = eps
        cols.append((h(x + dx) - h(x - dx)) / (2 * eps))
    return np.column_stack(cols)


class Test_meas_rel_pos:
    def test_returns_position_only(self) -> None:
        z = meas_rel_pos(X_GENERIC)
        assert z.shape == (3,)
        assert z == pytest.approx(X_GENERIC[0:3])

    def test_jacobian_selects_position(self) -> None:
        H = meas_rel_pos_jac(X_GENERIC)
        assert H.shape == (3, 6)
        assert H == pytest.approx(np.hstack([np.eye(3), np.zeros((3, 3))]))
        assert H @ X_GENERIC == pytest.approx(
            meas_rel_pos(X_GENERIC)
        )  # linear: h(x) = H x


class Test_meas_range_az_el:
    def test_hand_calc_3_4_12(self) -> None:
        """r = (3, 4, 12) m: range = 13, rho = 5,
        az = atan2(4, 3) = 53.130102 deg, el = atan2(12, 5) = 67.380135 deg."""
        z = meas_range_az_el(np.array([3.0, 4.0, 12.0, 0.0, 0.0, 0.0]))
        assert z.shape == (3,)
        assert z == pytest.approx(
            [13.0, 53.130102 * DEG_TO_RAD, 67.380135 * DEG_TO_RAD]
        )

    @pytest.mark.parametrize(
        ("r", "az_deg", "el_deg"),
        [
            ([1000.0, 0.0, 0.0], 0.0, 0.0),  # radially above (+R)
            ([0.0, 1000.0, 0.0], 90.0, 0.0),  # ahead along-track (+S)
            ([0.0, -1000.0, 0.0], -90.0, 0.0),  # behind along-track
            ([-1000.0, 0.0, 0.0], 180.0, 0.0),  # radially below
            ([1000.0, 0.0, 1000.0], 0.0, 45.0),  # 45 deg toward +W
            ([0.0, 1000.0, -1000.0], 90.0, -45.0),
        ],
    )
    def test_axis_geometry(self, r: list[float], az_deg: float, el_deg: float) -> None:
        z = meas_range_az_el(np.array([*r, 0.0, 0.0, 0.0]))
        assert z[1:3] == pytest.approx(
            [az_deg * DEG_TO_RAD, el_deg * DEG_TO_RAD], abs=1e-12
        )

    def test_inverts_to_position(self) -> None:
        """Spherical -> Cartesian recovers r (an independent inverse of the model)."""
        rng, az, el = meas_range_az_el(X_GENERIC)
        r = rng * np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
        )
        assert r == pytest.approx(X_GENERIC[0:3], rel=1e-12)

    def test_independent_of_velocity(self) -> None:
        x2 = X_GENERIC.copy()
        x2[3:6] = [5.0, -3.0, 1.0]
        assert meas_range_az_el(x2) == pytest.approx(meas_range_az_el(X_GENERIC), abs=0)


class Test_meas_range_az_el_jac:
    def test_hand_calc_3_4_12(self) -> None:
        """r = (3, 4, 12): r = 13, rho = 5, r^2 rho = 845.
        d range = [3, 4, 12] / 13
        d az    = [-4, 3, 0] / 25
        d el    = [-36/845, -48/845, 5/169]
        """
        H = meas_range_az_el_jac(np.array([3.0, 4.0, 12.0, 0.0, 0.0, 0.0]))
        expected = np.zeros((3, 6))
        expected[0, 0:3] = [3 / 13, 4 / 13, 12 / 13]
        expected[1, 0:3] = [-0.16, 0.12, 0.0]
        expected[2, 0:3] = [-36 / 845, -48 / 845, 5 / 169]
        assert H == pytest.approx(expected, abs=1e-15)

    @pytest.mark.parametrize(
        "x",
        [
            X_GENERIC,
            np.array([20e3, 500.0, -800.0, 0.0, 0.0, 0.0]),  # far, mostly radial
            np.array([-5.0, 3.0, 40.0, 0.0, 0.0, 0.0]),  # close, steep elevation
        ],
        ids=["generic", "far_radial", "close_steep"],
    )
    def test_matches_finite_differences(self, x: np.ndarray) -> None:
        H = meas_range_az_el_jac(x)
        assert H.shape == (3, 6)
        assert H == pytest.approx(
            jac_fd(meas_range_az_el, x, eps=1e-4), rel=1e-6, abs=1e-12
        )

    def test_range_row_is_line_of_sight_unit_vector(self) -> None:
        H = meas_range_az_el_jac(X_GENERIC)
        los = X_GENERIC[0:3] / np.linalg.norm(X_GENERIC[0:3])
        assert H[0, 0:3] == pytest.approx(los)

    def test_rows_orthogonal(self) -> None:
        """Range, az and el gradients point along the orthogonal spherical unit vectors."""
        G = meas_range_az_el_jac(X_GENERIC)[:, 0:3]
        assert G[0] @ G[1] == pytest.approx(0.0, abs=1e-15)
        assert G[0] @ G[2] == pytest.approx(0.0, abs=1e-15)
        assert G[1] @ G[2] == pytest.approx(0.0, abs=1e-15)
