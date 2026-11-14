"""Unit tests for the pure RPO planners in `cesium.guidance.hcw`.

Each planner returns numbers only; the tests apply the plan with an independent
two-body propagation and check the resulting geometry (on the plane, at the apsis,
on the radius, at the target) rather than re-deriving the planner's formula.
"""

from dataclasses import replace

import numpy as np
import pytest
from numpy.linalg import norm

from cesium.constants import DEG_TO_RAD, KM_TO_M, MIN_TO_SEC
from cesium.guidance.hcw import (
    hcw_rendezvous,
    plan_along_track_to_sma,
    plan_coast_to_nearer_apsis,
    plan_enter_phasing,
    plan_hohmann_from_apsis,
    plan_plane_change,
    shape_matches,
)
from cesium.guidance.timeline import RpoTimeline
from cesium.math.base import unit
from cesium.orbital.coes import COES, orbit_sweep
from cesium.physics.dynamics import propagate_two_body
from cesium.world.constants import MU_MARS, R_MARS

MU = MU_MARS
A_TARGET = 3696 * KM_TO_M  # 300 km altitude
TARGET = COES(a=A_TARGET, e=0.0, i=45 * DEG_TO_RAD, raan=0.5, aop=0.0, ta=0.3)


def coast(rv: np.ndarray, dt: float) -> np.ndarray:
    """Two-body state after `dt` [s] (independent of the planners' Kepler math)."""
    return propagate_two_body(
        rv, np.array([0.0, dt]), mu=MU, j2=None, rtol=1e-12, atol=1e-6
    )[-1]


def apply_dv(rv: np.ndarray, dv: np.ndarray) -> np.ndarray:
    out = rv.copy()
    out[3:6] += dv
    return out


class Test_plan_along_track_to_sma:
    RV = TARGET.to_rv(MU)  # circular, r = A_TARGET

    def test_same_sma_needs_no_burn(self) -> None:
        dv = plan_along_track_to_sma(self.RV, A_TARGET, MU)
        assert dv == pytest.approx(np.zeros(3), abs=1e-9)

    def test_raise_sma_hand_calc(self) -> None:
        """Circular at r = 3696 km to a = 3800 km (burn point becomes periapsis):
        v_c   = sqrt(mu / r)              = sqrt(42828e9 / 3696e3)  = 3404.0656 m/s
        v_req = sqrt(mu (2 / r - 1 / a))                            = 3450.3331 m/s
        dv    = 46.2675 m/s, prograde
        """
        dv = plan_along_track_to_sma(self.RV, 3800 * KM_TO_M, MU)
        assert norm(dv) == pytest.approx(46.26752, abs=1e-5)  # magnitude
        assert unit(dv) == pytest.approx(unit(self.RV[3:6]), abs=1e-5)  # direction
        assert norm(np.cross(dv, self.RV[3:6])) == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("a_new_km", [3500.0, 3696.0, 4200.0])
    def test_resulting_orbit_has_requested_sma(self, a_new_km: float) -> None:
        rv = COES(a=3750e3, e=0.02, i=0.6, raan=0.1, aop=0.4, ta=1.0).to_rv(MU)
        rv_new = apply_dv(rv, plan_along_track_to_sma(rv, a_new_km * KM_TO_M, MU))
        assert COES.from_rv(rv_new, MU).a == pytest.approx(
            a_new_km * KM_TO_M, rel=1e-12
        )


class Test_plan_coast_to_nearer_apsis:
    ORBIT = COES(a=3800e3, e=0.05, i=0.6, raan=0.1, aop=0.4, ta=0.0)

    @pytest.mark.parametrize(
        ("ta_deg", "expect_periapsis"),
        [(200.0, True), (350.0, True), (10.0, False), (179.9, False), (180.1, True)],
    )
    def test_lands_on_nearer_apsis(self, ta_deg: float, expect_periapsis: bool) -> None:
        c = self.ORBIT.with_ta(ta_deg * DEG_TO_RAD)
        dt, at_periapsis = plan_coast_to_nearer_apsis(c, MU)

        assert at_periapsis == expect_periapsis
        assert 0 <= dt < c.period(MU) / 2  # nearer apsis is always within half a rev
        r_expected = c.r_p if expect_periapsis else c.r_a
        assert norm(coast(c.to_rv(MU), dt)[0:3]) == pytest.approx(r_expected, rel=1e-9)


class Test_plan_plane_change:
    CHASER = COES(a=A_TARGET, e=0.0, i=50 * DEG_TO_RAD, raan=0.6, aop=0.0, ta=1.0)

    def test_coplanar_returns_none(self) -> None:
        chaser = TARGET.with_ta(2.0)  # same plane
        assert plan_plane_change(chaser.to_rv(MU), TARGET, MU) is None

    def test_raan_bias_can_make_planes_coplanar(self) -> None:
        """Same inclination, RAANs 10 rad apart: biasing the target by 10 aligns them."""
        chaser = TARGET.bias_raan(10).with_ta(2.0)
        assert plan_plane_change(chaser.to_rv(MU), TARGET, MU) is not None
        assert plan_plane_change(chaser.to_rv(MU), TARGET, MU, raan_bias=10) is None

    def test_coast_reaches_node_on_target_plane(self) -> None:
        n_hat, d_ta = plan_plane_change(self.CHASER.to_rv(MU), TARGET, MU)
        assert 0 <= d_ta < 2 * np.pi

        _, dt = orbit_sweep(self.CHASER, d_ta, MU)
        r_node = coast(self.CHASER.to_rv(MU), dt)[0:3]

        # At the node the chaser lies in both planes, along n_hat
        assert unit(r_node) @ unit(TARGET.h(MU)) == pytest.approx(
            0.0, abs=1e-9
        )  # on target plane
        assert unit(r_node) @ unit(self.CHASER.h(MU)) == pytest.approx(
            0.0, abs=1e-9
        )  # on chaser plane

        assert unit(r_node) == pytest.approx(n_hat, abs=1e-9)


class Test_shape_matches:
    def test_circular_on_radius(self) -> None:
        assert shape_matches(TARGET, A_TARGET)

    def test_within_tolerance(self) -> None:
        assert shape_matches(TARGET, A_TARGET + 0.5, tol=0.5000001)
        assert not shape_matches(TARGET, A_TARGET + 0.5, tol=0.5)

        assert shape_matches(TARGET, A_TARGET + 2.0, tol=2.000001)
        assert not shape_matches(TARGET, A_TARGET + 2.0, tol=2)
        assert not shape_matches(TARGET, A_TARGET + 2.0, tol=1.9)

    def test_eccentric_orbit_with_matching_sma_does_not_match(self) -> None:
        """Same a, but e = 0.01 puts the apses ~37 km off the target radius."""
        assert not shape_matches(replace(TARGET, e=0.01), A_TARGET, tol=5)


class Test_plan_hohmann_from_apsis:
    @pytest.mark.parametrize("ta", [0.0, np.pi], ids=["periapsis", "apoapsis"])
    def test_burns_circularize_on_target_radius(self, ta: float) -> None:
        chaser = COES(a=3780e3, e=0.01, i=0.6, raan=0.1, aop=0.4, ta=ta)
        at_periapsis = ta == 0.0
        dv_a, dt_trans, dv_b = plan_hohmann_from_apsis(
            chaser, A_TARGET, at_periapsis, MU
        )

        # Burn at apsis it is supposed to
        rv = chaser.to_rv(MU)
        rv = apply_dv(rv, dv_a * unit(rv[3:6]))
        rv = coast(rv, dt_trans)
        assert norm(rv[0:3]) == pytest.approx(A_TARGET, rel=1e-9)

        # Circularize and compare to target orbit
        rv = apply_dv(rv, dv_b * unit(rv[3:6]))
        c_final = COES.from_rv(rv, MU)
        assert c_final.a == pytest.approx(A_TARGET, rel=1e-9)
        assert c_final.e == pytest.approx(0.0, abs=1e-8)


class Test_plan_enter_phasing:
    @pytest.mark.parametrize("lead_deg", [-20.0, 15.0], ids=["behind", "ahead"])
    @pytest.mark.parametrize("k_phase", [1, 3])
    def test_meets_target_after_phasing(self, lead_deg: float, k_phase: int) -> None:
        """Scenario

        Same circular orbit, chaser offset in anomaly: after the entry burn and
        `dt_ph` of coasting, the chaser is at the target's (independently propagated)
        position."""
        rv_c = TARGET.with_ta(TARGET.ta + lead_deg * DEG_TO_RAD).to_rv(MU)
        rv_t = TARGET.to_rv(MU)

        dv, a_ph, dt_ph = plan_enter_phasing(
            rv_c, TARGET, mu=MU, k_phase=k_phase, r_cb=R_MARS
        )

        rv_c = apply_dv(rv_c, dv)
        assert COES.from_rv(rv_c, MU).a == pytest.approx(a_ph, rel=1e-9)

        # Behind: drop to a faster, lower orbit; ahead: wait on a higher one
        assert np.sign(a_ph - A_TARGET) == np.sign(lead_deg)

        rv_c_final = coast(rv_c, dt_ph)[0:3]
        rv_t_final = coast(rv_t, dt_ph)[0:3]
        miss = norm(rv_c_final - rv_t_final)
        assert miss < 1.0  # m

    def test_dv_circ_is_merged_along_track(self) -> None:
        """Burn-to-go from Burn 2 is folded into the same along-track burn."""
        rv_c = TARGET.with_ta(TARGET.ta - 0.2).to_rv(MU)
        dv, _, _ = plan_enter_phasing(rv_c, TARGET, mu=MU, k_phase=2, dv_circ=5.0)
        assert norm(np.cross(dv, rv_c[3:6])) == pytest.approx(0.0, abs=1e-6)

    def test_rejects_phasing_orbit_inside_planet(self) -> None:
        rv_c = TARGET.with_ta(TARGET.ta - 170 * DEG_TO_RAD).to_rv(MU)
        with pytest.raises(ValueError, match="hits the planet"):
            plan_enter_phasing(rv_c, TARGET, mu=MU, k_phase=1, r_cb=R_MARS)


class Test_hcw_rendezvous:
    @pytest.mark.parametrize("max_iter", [0, -1])
    def test_no_iterations_is_a_no_op(self, max_iter: int) -> None:
        """No passes: no burns (not even "HCW stop"), state untouched, returns 0."""
        rv_c = TARGET.with_ta(TARGET.ta - 1e-3).to_rv(MU)  # ~3.7 km behind
        tl = RpoTimeline(rv_c, TARGET.to_rv(MU), MU, frame="MCI")
        chaser_before = tl.chaser.copy()

        passes = hcw_rendezvous(
            tl, 40 * MIN_TO_SEC, dock_thresh=50.0, max_iter=max_iter
        )

        assert passes == 0
        assert tl.burns == []
        assert tl.dv_total == 0.0
        assert tl.chaser == pytest.approx(chaser_before, abs=0)
