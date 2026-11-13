import numpy as np
import pytest
from numpy.linalg import norm

from cesium.constants import DAY_TO_SEC, DEG_TO_RAD, RAD_TO_DEG
from cesium.guidance.arrival import b_plane_flyby, b_plane_frame, plan_direct_injection
from cesium.math.base import angle_between, unit
from cesium.orbital.coes import rv_to_coes
from cesium.physics.dynamics import propagate_two_body
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
    rv_far = propagate_two_body(
        np.concatenate((r0, v0)),
        [0.0, -days * DAY_TO_SEC],
        MU_MARS,
        rtol=1e-12,
        atol=1e-6,
    )[-1]
    r_far, v_far = rv_far[:3], rv_far[3:]
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


class Test_plan_direct_injection:
    # Arbitrary arrival direction in Mars-equatorial axes
    V_INF_MCI = EXAMPLE_V_INF * unit(np.array([0.3, -0.8, 0.2]))
    E = 1 + EXAMPLE_MARS_R_P * EXAMPLE_V_INF**2 / MU_MARS
    V_P = np.sqrt(EXAMPLE_V_INF**2 + 2 * MU_MARS / EXAMPLE_MARS_R_P)
    V_CIRC = np.sqrt(MU_MARS / EXAMPLE_MARS_R_P)

    def _plan(self, i_deg=45.0, raan_deg=90.0, **kwargs):  # beta ~ 24 deg
        return plan_direct_injection(
            self.V_INF_MCI, EXAMPLE_MARS_R_P, raan_deg, i_deg, MU_MARS, **kwargs
        )

    def test_periapsis_is_on_hyperbola_and_in_target_plane(self):
        plan = self._plan()
        S_hat = unit(self.V_INF_MCI)

        assert norm(plan.r_p) == pytest.approx(EXAMPLE_MARS_R_P, rel=1e-12)
        assert np.dot(unit(plan.r_p), S_hat) == pytest.approx(1 / self.E, rel=1e-12)
        assert np.dot(plan.r_p, plan.v_p_hyp) == pytest.approx(0.0, abs=1e-3)
        assert norm(plan.v_p_hyp) == pytest.approx(self.V_P, rel=1e-12)

    def test_backward_propagation_recovers_v_inf(self):
        """The pre-burn periapsis state flies back out along the incoming v_inf."""
        plan = self._plan()
        rv_far = propagate_two_body(
            np.concatenate((plan.r_p, plan.v_p_hyp)),
            [0.0, -200 * DAY_TO_SEC],
            MU_MARS,
            rtol=1e-12,
            atol=1e-6,
        )[-1]

        angle_deg = angle_between(rv_far[3:], self.V_INF_MCI) * RAD_TO_DEG
        assert angle_deg < 0.05
        assert norm(rv_far[3:]) == pytest.approx(EXAMPLE_V_INF, rel=1e-3)

    def test_aim_point_matches_propagated_b_vector(self):
        plan = self._plan()
        B_TR_prop = _propagated_b_vector(plan.r_p, plan.v_p_hyp)
        assert plan.B_TR == pytest.approx(B_TR_prop, abs=5e-4 * norm(B_TR_prop))

    @pytest.mark.parametrize(
        "i_deg, raan_deg", [(45.0, 90.0), (30.0, 210.0), (45.0, 180.0)]
    )
    def test_post_burn_state_is_target_orbit(self, i_deg: float, raan_deg: float):
        plan = self._plan(i_deg, raan_deg)
        _, e, i, raan, _, _ = rv_to_coes(plan.r_p, plan.v_p_capture, MU_MARS)

        assert e == pytest.approx(0.0, abs=1e-9)
        assert i == pytest.approx(i_deg * DEG_TO_RAD, abs=1e-9)
        assert raan == pytest.approx(raan_deg * DEG_TO_RAD, abs=1e-9)
        (burn,) = plan.burns
        assert burn.dv == pytest.approx(plan.v_p_capture - plan.v_p_hyp)
        assert plan.dv_total == pytest.approx(burn.dv_mag)

    def test_target_plane_containing_v_inf(self):
        """beta = 0: no plane change, so dv = v_p - v_circ."""
        S_hat = unit(self.V_INF_MCI)
        n_hat = unit(np.cross(S_hat, [0.0, 0.0, 1.0]))  # any normal perpendicular to S
        if n_hat[2] < 0:
            n_hat = -n_hat
        i_deg = np.arccos(n_hat[2]) * RAD_TO_DEG
        raan_deg = np.arctan2(n_hat[0], -n_hat[1]) * RAD_TO_DEG % 360

        plan = self._plan(i_deg, raan_deg)

        assert plan.beta == pytest.approx(0.0, abs=1e-7)
        assert plan.dv_total == pytest.approx(self.V_P - self.V_CIRC, rel=1e-9)

    def test_other_branch_costs_more(self):
        cheap, other = self._plan(), self._plan(low_energy_path=False)
        assert cheap.dv_total < other.dv_total
        assert np.dot(cheap.r_p, other.r_p) < EXAMPLE_MARS_R_P**2  # different periapsis

    def test_unreachable_plane_raises(self):
        """beta > arccos(1/e): the target plane is too far from the asymptote."""
        S_hat = unit(self.V_INF_MCI)
        # Normal nearly along S -> plane almost perpendicular to the asymptote
        n_hat = unit(S_hat + 0.05 * np.array([0.0, 0.0, 1.0]))
        i_deg = np.arccos(n_hat[2]) * RAD_TO_DEG
        raan_deg = np.arctan2(n_hat[0], -n_hat[1]) * RAD_TO_DEG % 360
        with pytest.raises(ValueError, match="Cannot enter target plane"):
            self._plan(i_deg, raan_deg)

    def test_single_burn_at_periapsis(self):
        plan = self._plan()
        (burn,) = plan.burns
        assert burn.t_s == 0.0
        assert burn.frame == "PCI"
        assert np.array_equal(burn.r, plan.r_p)
