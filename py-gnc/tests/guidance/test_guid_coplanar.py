import numpy as np
import pytest
from numpy.linalg import norm

from cesium.constants import DEG_TO_RAD, KM_TO_M, SEC_TO_MIN
from cesium.guidance.coplanar import (
    circularize,
    guid_coaxial_hohmann,
    guid_hohmann,
    guid_one_tangent,
    guid_phasing_different_orbits,
    guid_phasing_same_orbit,
)
from cesium.math.base import unit
from cesium.orbital.coes import sma_from_rv
from cesium.physics.dynamics import propagate_two_body
from cesium.world import MU_EARTH

R_LEO = 7000 * KM_TO_M
R_MEO = 12000 * KM_TO_M


def _state_at_apsis(r_p, r_a, at_periapsis, periapsis_dir=1.0):
    """Prograde state at an apsis"""
    a = (r_p + r_a) / 2
    r = r_p if at_periapsis else r_a
    v = np.sqrt(MU_EARTH * (2 / r - 1 / a))
    sign = periapsis_dir if at_periapsis else -periapsis_dir
    return np.array([sign * r, 0.0, 0.0]), np.array([0.0, sign * v, 0.0])


def _apsides(r, v):
    """for a given orbit with (r,v) state this gives `r_p`, `r_a`, `e`"""
    a = sma_from_rv(norm(r), norm(v), MU_EARTH)
    e_vec = ((norm(v) ** 2 - MU_EARTH / norm(r)) * r - np.dot(r, v) * v) / MU_EARTH
    e = norm(e_vec)
    return a * (1 - e), a * (1 + e), e_vec


class Test_circularize:
    def test_circular_orbit_needs_no_burn(self):
        assert circularize(R_LEO, 0.0, True, MU_EARTH) == pytest.approx(0.0, abs=1e-9)
        assert circularize(R_LEO, 0.0, False, MU_EARTH) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("at_periapsis", [True, False])
    def test_burn_leaves_circular_orbit(self, at_periapsis):
        a, e = 8000 * KM_TO_M, 0.1
        r, v = _state_at_apsis(a * (1 - e), a * (1 + e), at_periapsis)

        dv = circularize(a, e, at_periapsis, MU_EARTH)
        r_p, r_a, _ = _apsides(r, v + dv * unit(v))

        assert r_p == pytest.approx(norm(r), rel=1e-12)
        assert r_a == pytest.approx(norm(r), rel=1e-12)
        assert np.sign(dv) == (
            -1 if at_periapsis else 1
        )  # slow at periapsis, speed up at apoapsis


class Test_guid_coaxial_hohmann:
    @pytest.mark.parametrize("depart_at_periapsis", [True, False])
    def test_circular_orbits_match_hohmann(self, depart_at_periapsis):
        expected = guid_hohmann(R_LEO, R_MEO, MU_EARTH)
        result = guid_coaxial_hohmann(
            R_LEO, R_LEO, R_MEO, R_MEO, depart_at_periapsis, mu=MU_EARTH
        )
        assert result == pytest.approx(expected, rel=1e-12)

    def test_same_orbit_needs_no_burns(self):
        r_p, r_a = 7000 * KM_TO_M, 9000 * KM_TO_M
        _, t_trans, dv_a, dv_b = guid_coaxial_hohmann(
            r_p,
            r_a,
            r_p,
            r_a,
            depart_at_periapsis=True,
            arrive_at_periapsis=False,
            mu=MU_EARTH,
        )
        assert (dv_a, dv_b) == pytest.approx((0.0, 0.0), abs=1e-9)
        # Hand calc: a = (r_p + r_a) / 2 = 8000 km
        #   t = pi * sqrt(a^3 / mu) = pi * sqrt(5.12e20 / 3.986004418e14) = pi * 1133.36 s
        assert t_trans == pytest.approx(3560.541, abs=1e-3)

    @pytest.mark.parametrize("depart_at_periapsis", [True, False])
    @pytest.mark.parametrize("arrive_at_periapsis", [True, False])
    def test_propagated_transfer_reaches_final_orbit(
        self, depart_at_periapsis, arrive_at_periapsis
    ):
        """Burn, coast t_trans, burn: final orbit has the target apsides and orientation.

        Orbit 1's periapsis is along +x. Orbit 2 is oriented so the chosen
        arrival apsis sits 180 deg from the departure point.
        """
        r_p1, r_a1 = 7000 * KM_TO_M, 9000 * KM_TO_M
        r_p2, r_a2 = 12000 * KM_TO_M, 20000 * KM_TO_M
        _, t_trans, dv_a, dv_b = guid_coaxial_hohmann(
            r_p1, r_a1, r_p2, r_a2, depart_at_periapsis, arrive_at_periapsis, MU_EARTH
        )

        r0, v0 = _state_at_apsis(r_p1, r_a1, depart_at_periapsis)
        rv_end = propagate_two_body(
            np.concatenate((r0, v0 + dv_a * unit(v0))),
            [0.0, t_trans],
            MU_EARTH,
            rtol=1e-12,
            atol=1e-6,
        )[-1]
        r1, v1 = rv_end[:3], rv_end[3:]

        assert np.dot(unit(r1), unit(r0)) == pytest.approx(-1.0, abs=1e-9)  # 180 deg
        r_p, r_a, e_vec = _apsides(r1, v1 + dv_b * unit(v1))
        assert r_p == pytest.approx(r_p2, rel=1e-8)
        assert r_a == pytest.approx(r_a2, rel=1e-8)
        # Orbit 2's periapsis points at the arrival point if arriving at periapsis,
        # away from it if arriving at apoapsis.
        expected_dir = unit(r1) if arrive_at_periapsis else -unit(r1)
        assert np.dot(unit(e_vec), expected_dir) == pytest.approx(1.0, abs=1e-9)

    def test_elliptic_to_circular_shape_correction(self):
        """RPO-style: slightly elliptical chaser to a circular target radius."""
        r_p1, r_a1, r_f = 3660 * KM_TO_M, 3723 * KM_TO_M, 3696 * KM_TO_M
        costs = {}
        for depart_at_periapsis in (True, False):
            _, _, dv_a, dv_b = guid_coaxial_hohmann(
                r_p1, r_a1, r_f, r_f, depart_at_periapsis, mu=MU_EARTH
            )
            costs[depart_at_periapsis] = abs(dv_a) + abs(dv_b)
            # target between the apsides: leaving periapsis lowers r_a (retrograde);
            # leaving apoapsis raises r_p (prograde)
            assert np.sign(dv_a) == (-1 if depart_at_periapsis else 1)
        assert all(c > 0 for c in costs.values())

    def test_rejects_swapped_apsides(self):
        with pytest.raises(ValueError):
            guid_coaxial_hohmann(9000e3, 7000e3, R_MEO, R_MEO, mu=MU_EARTH)


class Test_guid_one_tangent:
    def test_180_deg_transfer_matches_hohmann(self):
        """With the second burn at nu = 180 deg the transfer is a Hohmann."""
        a_h, t_h, dv_a_h, dv_b_h = guid_hohmann(R_LEO, R_MEO, MU_EARTH)
        a, t, dv_a, dv_b = guid_one_tangent(
            R_LEO, R_MEO, np.pi * (1 - 1e-9), True, MU_EARTH
        )
        assert a == pytest.approx(a_h, rel=1e-6)
        assert t == pytest.approx(t_h, rel=1e-6)
        assert dv_a == pytest.approx(dv_a_h, rel=1e-6)
        assert dv_b == pytest.approx(abs(dv_b_h), rel=1e-6)

    def test_propagated_transfer_crosses_final_orbit(self):
        """Burn tangentially, coast t_trans: arrive at r_f and nu_b; dv_b matches."""
        ta_b = 160 * DEG_TO_RAD
        _, t_trans, dv_a, dv_b = guid_one_tangent(R_LEO, R_MEO, ta_b, True, MU_EARTH)

        r0, v0 = _state_at_apsis(R_LEO, R_LEO, True)
        rv_end = propagate_two_body(
            np.concatenate((r0, v0 + dv_a * unit(v0))),
            [0.0, t_trans],
            MU_EARTH,
            rtol=1e-12,
            atol=1e-6,
        )[-1]
        r1, v1 = rv_end[:3], rv_end[3:]

        assert norm(r1) == pytest.approx(R_MEO, rel=1e-8)
        assert np.arctan2(r1[1], r1[0]) == pytest.approx(ta_b, abs=1e-8)
        v_circ = np.sqrt(MU_EARTH / R_MEO) * unit(np.cross([0, 0, 1], r1))
        assert norm(v_circ - v1) == pytest.approx(dv_b, rel=1e-8)


class Test_guid_phasing:
    def test_same_orbit_no_change(self):
        """Period depends on target orbit"""
        a = 12756.274 * KM_TO_M
        period_expected = 2 * np.pi * np.sqrt(a**3 / MU_EARTH)

        dv, t_phase, a_phase = guid_phasing_same_orbit(
            a, 0, MU_EARTH, k_target=1, k_chaser=1, deg=True
        )
        assert dv == pytest.approx(0, abs=1e-3)  # retrograde
        assert t_phase == pytest.approx(period_expected, abs=1e-3)
        assert a_phase == pytest.approx(a, abs=1e-1)

        dv, t_phase, a_phase = guid_phasing_same_orbit(
            a, 0, MU_EARTH, k_target=2, k_chaser=2, deg=True
        )
        assert dv == pytest.approx(0, abs=1e-3)  # retrograde
        assert t_phase == pytest.approx(2 * period_expected, abs=1e-3)
        assert a_phase == pytest.approx(a, abs=1e-1)

        dv, t_phase, a_phase = guid_phasing_same_orbit(
            a, 0, MU_EARTH, k_target=3, k_chaser=2, deg=True
        )
        assert t_phase == pytest.approx(3 * period_expected, abs=1e-3)

        dv, t_phase, a_phase = guid_phasing_same_orbit(
            a, 0, MU_EARTH, k_target=4, k_chaser=2, deg=True
        )
        assert t_phase == pytest.approx(4 * period_expected, abs=1e-3)

    def test_same_orbit_vallado_example_6_8(self):
        a = 12756.274 * KM_TO_M
        d_ta = -20

        dv, t_phase, a_phase = guid_phasing_same_orbit(
            a, d_ta, MU_EARTH, k_target=1, k_chaser=1, deg=True
        )
        assert dv == pytest.approx(-219.268, abs=1e-3)  # retrograde
        assert t_phase * SEC_TO_MIN == pytest.approx(225.6950, abs=1e-3)
        assert a_phase == pytest.approx(12279333.2, abs=1e-1)

        dv, t_phase, a_phase = guid_phasing_same_orbit(
            a, d_ta, MU_EARTH, k_target=2, k_chaser=1, deg=True
        )
        assert t_phase * SEC_TO_MIN == pytest.approx(464.6661, abs=1e-4)

        dv, t_phase, a_phase = guid_phasing_same_orbit(
            a, d_ta, MU_EARTH, k_target=2, k_chaser=2, deg=True
        )
        assert dv == pytest.approx(-106.481, abs=1e-3)
        assert t_phase * SEC_TO_MIN == pytest.approx(
            464.6661, abs=1e-4
        )  # lots of minutes
        assert a_phase == pytest.approx(12518939.3, abs=1e-1)

    def test_same_orbit_vallado_example_6_9(self):
        """Period depends on target orbit"""
        a_chaser = 12756.274 * KM_TO_M
        a_target = 42159.48 * KM_TO_M
        d_ta = -20
        dv, t_trans, t_wait, a_trans = guid_phasing_different_orbits(
            a_chaser, a_target, d_ta, mu=MU_EARTH, k=1, deg=True
        )
        assert dv == pytest.approx(2315.71, abs=1e-3)
        assert t_trans * SEC_TO_MIN == pytest.approx(377.9537, abs=1.0)
        # assert t_wait * SEC_TO_MIN == pytest.approx(377.9537, abs=1.0)  # book value is wrong
        assert a_trans == pytest.approx(27457877, abs=1)

        # Regression pin (book's t_wait is wrong); fly_vallado_6_9 checks it physically
        assert t_wait * SEC_TO_MIN == pytest.approx(234.6110, abs=1e-4)

    def test_vallado_example_6_9_plan_meets_target(self):
        """Test that Vallado is actaually correct.

        Circular coplanar orbits; the target starts 20 deg ahead of the chaser.
        """
        a_c, a_t, d_ta = 12756.274 * KM_TO_M, 42159.48 * KM_TO_M, -20.0
        _, t_trans, t_wait, _ = guid_phasing_different_orbits(
            a_c, a_t, d_ta, mu=MU_EARTH, k=1, deg=True
        )

        def circular(a, angle):
            v = np.sqrt(MU_EARTH / a)
            c, s = np.cos(angle), np.sin(angle)
            return np.array([a * c, a * s, 0.0, -v * s, v * c, 0.0])

        chaser = propagate_two_body(circular(a_c, 0.0), [0.0, t_wait], MU_EARTH)[-1]
        a_trans = (a_c + a_t) / 2
        v_peri = np.sqrt(MU_EARTH * (2 / a_c - 1 / a_trans))  # vis-viva at departure
        chaser[3:6] = v_peri * unit(chaser[3:6])
        chaser = propagate_two_body(chaser, [0.0, t_trans], MU_EARTH)[-1]

        target0 = circular(a_t, -d_ta * DEG_TO_RAD)
        target = propagate_two_body(target0, [0.0, t_wait + t_trans], MU_EARTH)[-1]

        assert norm(chaser[0:3] - target[0:3]) < 1 * KM_TO_M  # vs 42,000 km orbit
