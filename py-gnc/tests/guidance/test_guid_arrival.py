import numpy as np
import pytest
from numpy.linalg import norm
from scipy.integrate import solve_ivp

from cesium.constants import DAY_TO_SEC, DEG_TO_RAD
from cesium.guidance.arrival import b_plane_flyby, b_plane_frame
from cesium.math.base import unit
from cesium.physics.dynamics import two_body
from cesium.world.constants import MU_MARS, R_MARS

EXAMPLE_MARS_R_P = R_MARS + 300e3  # m
EXAMPLE_V_INF = 2820.0  # m/s

"""
References
----------
    Jah, "Derivation of the B-Plane (Body Plane) and Its Associated Parameters",

"""


def _periapsis_state(tilt_rad: float):
    """Periapsis state of an arrival hyperbola on +x, plane tilted about x."""
    v_p = np.sqrt(EXAMPLE_V_INF**2 + 2 * MU_MARS / EXAMPLE_MARS_R_P)
    r0 = np.array([EXAMPLE_MARS_R_P, 0.0, 0.0])
    v0 = v_p * np.array([0.0, np.cos(tilt_rad), np.sin(tilt_rad)])
    return r0, v0


def _propagated_b_vector(r0, v0, days=200):
    """Brute force to get B vector. Propagates from periapsis to infinity.

    Mars-equatorial axes assumed (pole along +z), matching `b_plane_frame`.
    """
    sol = solve_ivp(
        lambda _, s: two_body(s, MU_MARS),
        (0, -days * DAY_TO_SEC),
        np.concatenate((r0, v0)),
        rtol=1e-12,
        atol=1e-6,
    )
    assert sol.success

    r_far, v_far = sol.y[:3, -1], sol.y[3:, -1]
    S_hat = unit(v_far)
    T_hat = unit(np.cross(S_hat, [0, 0, 1]))
    R_hat = np.cross(S_hat, T_hat)
    B = r_far - np.dot(r_far, S_hat) * S_hat
    return np.array([np.dot(B, T_hat), np.dot(B, R_hat)])


def _asymptotes(r0, v0):
    """Analytic incoming/outgoing v_inf vectors of the hyperbola through periapsis (r0, v0).

    Velocity far away is along ±r_hat at true anomaly ±ta_inf, cos(ta_inf) = -1/e.
    """
    e = 1 + norm(r0) * EXAMPLE_V_INF**2 / MU_MARS
    P = unit(r0)
    Q = np.cross(unit(np.cross(r0, v0)), P)
    s = np.sqrt(1 - 1 / e**2)
    return EXAMPLE_V_INF * (P / e + s * Q), EXAMPLE_V_INF * (-P / e + s * Q)


#########################################################################################################
#               B-plane Algorithm 1
#########################################################################################################


class Test_b_plane_frame:
    @pytest.mark.parametrize("tilt_deg", [10.0, 28.6, 60.0, 135.0])
    def test_b_plane_frame_matches_propagated_asymptote(self, tilt_deg: float):
        """B.T and B.R agree with the asymptote recovered by backwards propagation.

        200 days out the trajectory is still slightly curved, so allow 0.05%.
        """
        r0, v0 = _periapsis_state(tilt_deg * DEG_TO_RAD)

        B_T, B_R = b_plane_frame(r0, v0, MU_MARS)
        B_T_prop, B_R_prop = _propagated_b_vector(r0, v0)

        B_mag = np.hypot(B_T_prop, B_R_prop)
        assert B_T == pytest.approx(B_T_prop, abs=5e-4 * B_mag)
        assert B_R == pytest.approx(B_R_prop, abs=5e-4 * B_mag)

    def test_b_plane_magnitude_closed_form(self):
        """Jah Eq. 12."""
        r0, v0 = _periapsis_state(28 * DEG_TO_RAD)

        B_T, B_R = b_plane_frame(r0, v0, MU_MARS)

        expected = EXAMPLE_MARS_R_P * np.sqrt(
            1 + 2 * MU_MARS / (EXAMPLE_V_INF**2 * EXAMPLE_MARS_R_P)
        )
        assert norm([B_T, B_R]) == pytest.approx(expected, rel=1e-9)

    def test_b_plane_equatorial_arrival_has_no_B_R(self):
        """A hyperbola in Mars's equator has B entirely along T (B.R = 0)."""
        r0, v0 = _periapsis_state(0.0)

        B_T, B_R = b_plane_frame(r0, v0, MU_MARS)

        assert B_R == pytest.approx(0.0, abs=1e-6)
        assert abs(B_T) == pytest.approx(
            EXAMPLE_MARS_R_P
            * np.sqrt(1 + 2 * MU_MARS / (EXAMPLE_V_INF**2 * EXAMPLE_MARS_R_P)),
            rel=1e-9,
        )


#########################################################################################################
#               B-plane Algorithm 2
#########################################################################################################


class Test_b_plane_flyby:
    @pytest.mark.parametrize("tilt_deg", [10.0, 28.6, 60.0, 135.0])
    def test_matches_b_plane_frame(self, tilt_deg: float):
        """Same hyperbola from v_inf in/out (Alg. 79) and from a state (Alg. 78)."""
        r0, v0 = _periapsis_state(tilt_deg * DEG_TO_RAD)
        v_in, v_out = _asymptotes(r0, v0)

        B_vec, r_p = b_plane_flyby(v_in, v_out, MU_MARS)

        S_hat = unit(v_in)
        T_hat = unit(np.cross(S_hat, [0, 0, 1]))
        R_hat = np.cross(S_hat, T_hat)
        B_T, B_R = b_plane_frame(r0, v0, MU_MARS)

        assert r_p == pytest.approx(EXAMPLE_MARS_R_P, rel=1e-9)
        assert np.dot(B_vec, S_hat) == pytest.approx(0.0, abs=1e-6)
        assert np.dot(B_vec, T_hat) == pytest.approx(B_T, rel=1e-9)
        assert np.dot(B_vec, R_hat) == pytest.approx(B_R, rel=1e-9, abs=1e-6)

    def test_magnitude_from_turning_angle(self):
        """|B| = mu / v_inf^2 / tan(phi / 2), with phi the turning angle.
        Simply the flyby relation between both infinity vectors"""
        r0, v0 = _periapsis_state(28 * DEG_TO_RAD)
        v_in, v_out = _asymptotes(r0, v0)

        B_vec, _ = b_plane_flyby(v_in, v_out, MU_MARS)

        phi = np.arccos(np.dot(unit(v_in), unit(v_out)))
        assert norm(B_vec) == pytest.approx(
            MU_MARS / EXAMPLE_V_INF**2 / np.tan(phi / 2), rel=1e-9
        )
