import numpy as np
import pytest
from numpy.linalg import norm

from cesium.constants import DAY_TO_SEC, DEG_TO_RAD, KM_TO_M, MIN_TO_SEC, RAD_TO_DEG
from cesium.math.base import angle_between
from cesium.math.rotations import dcm_z, wrap_to_pi
from cesium.orbital.coes import (
    COES,
    E_from_ta,
    M_from_ta,
    coes_to_rv,
    kepler_eq,
    # kepler_eq_h,  # commented out in coes.py for now
    kepler_eq_p,
    orbit_sweep,
    pkepler,
    rv_to_coes,
    sma_from_rv,
    ta_from_M,
    vis_viva,
)
from cesium.physics.dynamics import propagate_two_body
from cesium.world import J2_EARTH, MU_EARTH, MU_EARTH_KM, R_EARTH, W_EARTH


def test_kepler():
    """Vallado 4e Example 2-1 p. 66"""
    M = 235.4 * DEG_TO_RAD
    e = 0.4
    E = kepler_eq(M, e)

    assert E * RAD_TO_DEG == pytest.approx(220.512074767522)


def test_E_from_ta_curtis():
    """Curtis Example 3.1 p. 116"""
    rp = 9600 * KM_TO_M
    ra = 21000 * KM_TO_M
    e = (ra - rp) / (ra + rp)

    a = rp / (1 - e)
    T = 2 * np.pi * np.sqrt(a**3 / MU_EARTH)

    # want to get to 120 deg
    ta = 120 * DEG_TO_RAD
    E = E_from_ta(ta, e)
    M = E - e * np.sin(E)

    t = M / 2 / np.pi * T
    assert t == pytest.approx(4077, abs=0.1)


def test_kepler_eq_p_vallado():
    """Vallado Example 2-2, p. 69"""
    rp = 25512 * KM_TO_M
    dt = 53.7874 * MIN_TO_SEC
    B = kepler_eq_p(dt, rp)
    assert B == pytest.approx(0.817751, abs=1e-6)


# def test_kepler_eq_h_vallado():
#     """Vallado Example 2-3, p. 71"""
#     M = 235.4 * DEG_TO_RAD
#     e = 2.4
#     H = kepler_eq_h(M, e)
#
#     assert H == pytest.approx(1.601376144, abs=1e-9)


class TestRvToCoes:
    def test_vallado(self):
        """Vallado 4e p. 115"""
        r = np.array([6524.834, 6862.875, 6448.296])  # km
        v = np.array([4.901327, 5.533756, -1.976341])  # km/s

        a, e, i, raan, aop, ta = rv_to_coes(r, v, MU_EARTH_KM)

        assert a == pytest.approx(36127.343, rel=1e-6)
        assert e == pytest.approx(0.832853, rel=1e-6)
        assert i * RAD_TO_DEG == pytest.approx(87.870, rel=1e-4)
        assert raan * RAD_TO_DEG == pytest.approx(227.898, rel=1e-5)
        assert aop * RAD_TO_DEG == pytest.approx(53.38, rel=1e-4)
        assert ta * RAD_TO_DEG == pytest.approx(92.335, rel=1e-5)

        """Vallado 4e p. 115 but in meters"""
        r = np.array([6524.834, 6862.875, 6448.296]) * 1000  # m
        v = np.array([4.901327, 5.533756, -1.976341]) * 1000  # m/s

        a, e, i, raan, aop, ta = rv_to_coes(r, v, MU_EARTH)

        assert a == pytest.approx(36127343.0, rel=1e-6)
        assert e == pytest.approx(0.832853, rel=1e-6)
        assert i * RAD_TO_DEG == pytest.approx(87.870, rel=1e-4)
        assert raan * RAD_TO_DEG == pytest.approx(227.898, rel=1e-5)
        assert aop * RAD_TO_DEG == pytest.approx(53.38, rel=1e-4)
        assert ta * RAD_TO_DEG == pytest.approx(92.335, rel=1e-5)

    def test_random(self):
        """http://orbitsimulator.com/formulas/OrbitalElements.html"""

        r = np.array([100, 1300, -6000])  # km
        v = np.array([0.1, -1.0, 2.1])  # km/s

        a, e, i, raan, aop, ta = rv_to_coes(r, v, MU_EARTH_KM)

        assert a == pytest.approx(3203.754267903962, rel=1e-6)
        assert e == pytest.approx(0.9955257908191099, rel=1e-6)
        assert i * RAD_TO_DEG == pytest.approx(93.90569432920545, rel=1e-4)
        assert raan * RAD_TO_DEG == pytest.approx(283.91249467651966, rel=1e-5)
        assert aop * RAD_TO_DEG == pytest.approx(77.26657593515651, rel=1e-4)
        assert ta * RAD_TO_DEG == pytest.approx(181.10299292301673, rel=1e-5)

    def test_circular(self):
        """http://orbitsimulator.com/formulas/OrbitalElements.html"""

        r = np.array([6000, 4713.675910949817, 1000])  # km
        v = np.array([4.446, -5.658, 0])  # km/s (circular orbit)

        a, e, i, raan, _, _ = rv_to_coes(r, v, MU_EARTH_KM)

        assert a == pytest.approx(7692.648, rel=1e-4)
        assert e == pytest.approx(0, abs=1e-3)
        assert i * RAD_TO_DEG == pytest.approx(172.53, rel=1e-4)
        assert raan * RAD_TO_DEG == pytest.approx(128.1599, rel=1e-5)
        # argument of perigee and true anomaly are undefined for circular orbits (can be any value)

    def test_0_inclination(self):
        """http://orbitsimulator.com/formulas/OrbitalElements.html"""

        r = np.array([6800, 0, 0])  # km
        v = np.array([0, 8.656278815328648, 0])  # km/s

        a, e, i, _, _, _ = rv_to_coes(r, v, MU_EARTH_KM)

        assert a == pytest.approx(9421.974578532005, rel=1e-4)
        assert e == pytest.approx(0.2782829179465395, abs=1e-4)
        assert i * RAD_TO_DEG == pytest.approx(0, rel=1e-4)
        # raan, aop and ta are undefined for circular orbits (can be any value)

    def test_90_inclination(self):
        """http://orbitsimulator.com/formulas/OrbitalElements.html"""

        r = np.array([6686, 0, 968.35])  # km
        v = np.array([0, 0, -5.899127972862638])  # km/s

        a, e, i, raan, aop, ta = rv_to_coes(r, v, MU_EARTH_KM)

        assert a == pytest.approx(4790.6431, rel=1e-5)
        assert e == pytest.approx(0.4305248975978197, rel=1e-4)
        assert i * RAD_TO_DEG == pytest.approx(90, rel=1e-5)
        assert raan * RAD_TO_DEG == pytest.approx(
            180, rel=1e-5
        )  # Since going down initially, ascends on other side
        assert aop * RAD_TO_DEG == pytest.approx(340.552, rel=1e-5)
        assert ta * RAD_TO_DEG == pytest.approx(191.206, rel=1e-5)

    def test_hyperbolic_a(self):
        """hyperbolic orbit semi-major axis should be negative

        Hand calc: a = -mu / (v^2 - 2 mu / r) = -398600.4418 / (225 - 113.88584)
                     = -398600.4418 / 111.11416 = -3587.3056 km
        """
        r = np.array([7000.0, 0.0, 0.0])  # km
        v = np.array([0.0, 15.0, 0.0])  # km/s

        a, e, _i, _raan, _aop, _ta = rv_to_coes(r, v, MU_EARTH_KM)

        assert e > 1  # hyperbolic orbit
        assert a < 0
        assert a == pytest.approx(-3587.3055571396117, rel=1e-6)

    def test_180_inclination(self):
        """http://orbitsimulator.com/formulas/OrbitalElements.html"""

        r = np.array([7000, 2, 0])  # km
        v = np.array([1.2, -2, 0])  # km/s

        a, e, i, _, _, _ = rv_to_coes(r, v, MU_EARTH_KM)

        assert a == pytest.approx(3675.56885, rel=1e-5)
        assert e == pytest.approx(0.930685, rel=1e-4)
        assert i * RAD_TO_DEG == pytest.approx(180, rel=1e-5)
        # raan, aop, and ta are undefined for equitorial orbits (can be any value)


class TestCoesToRv:
    def test_vallado(self):
        """Vallado 4e Example 2-6 p. 119"""

        # p given as 11067.79 but p = a(1-e**2)
        a = 36126.64  # km
        e = 0.83285
        i = 87.87 * DEG_TO_RAD
        raan = 227.89 * DEG_TO_RAD
        aop = 53.38 * DEG_TO_RAD
        ta = 92.335 * DEG_TO_RAD

        r, v = coes_to_rv(a, e, i, raan, aop, ta, MU_EARTH_KM)

        assert np.allclose(r, [6525.344, 6861.535, 6449.125], atol=0.1)
        assert np.allclose(v, [4.902276, 5.533124, -1.975709], atol=0.0001)

        # Convert to m now
        a = a * 1000
        r, v = coes_to_rv(a, e, i, raan, aop, ta)  # use m3/s2 constant
        assert np.allclose(r, [6525344, 6861535, 6449125], atol=0.1)
        assert np.allclose(v, [4902.276, 5533.124, -1975.709], atol=0.0001)

    def test_random(self):
        """http://orbitsimulator.com/formulas/OrbitalElements.html"""

        a = 3203.754267903962  # km
        e = 0.9955257908191099
        i = 93.90569432920545 * DEG_TO_RAD
        raan = 283.91249467651966 * DEG_TO_RAD
        aop = 77.26657593515651 * DEG_TO_RAD
        ta = 181.10299292301673 * DEG_TO_RAD

        r, v = coes_to_rv(a, e, i, raan, aop, ta, MU_EARTH_KM)

        assert np.allclose(r, [100.0, 1300.0, -6000.0], atol=0.00001)
        assert np.allclose(v, [0.1, -1.0, 2.1], atol=0.0001)

    def test_90_inclination(self):
        """http://orbitsimulator.com/formulas/OrbitalElements.html"""

        a = 4848.9197268518665  # km
        e = 0.3938736352565609
        i = 90.0 * DEG_TO_RAD
        raan = 180.0 * DEG_TO_RAD
        aop = 353.8847334059262 * DEG_TO_RAD
        ta = 177.87427980631205 * DEG_TO_RAD

        r, v = coes_to_rv(a, e, i, raan, aop, ta, MU_EARTH_KM)

        assert np.allclose(r, [6686, 0.0, 968.35], atol=1e-6)
        assert np.allclose(v, [1.0, 0.0, -5.899127972862638], atol=1e-4)


class TestRoundTrip:
    def test_coes_roundtrip(self):
        test_cases = [
            (7000e3, 0.0, 0.0, 0.0, 0.0, 1.0),
            (7000e3, 0.0, 0.5, 0.3, 0.0, 1.0),
            (7000e3, 0.2, 0.0, 0.0, 0.4, 1.0),
            (7000e3, 0.2, 0.5, 0.3, 0.4, 1.0),
            (7000e3, 0.0, np.pi, 0.0, 0.0, 1.0),
            (7000e3, 0.2, np.pi, 0.0, 0.4, 1.0),
        ]

        for case in test_cases:
            r0, v0 = coes_to_rv(*case, mu=MU_EARTH)
            coes = rv_to_coes(r0, v0, MU_EARTH)
            r1, v1 = coes_to_rv(*coes, mu=MU_EARTH)

            assert np.allclose(r0, r1, rtol=1e-10, atol=1e-6)
            assert np.allclose(v0, v1, rtol=1e-10, atol=1e-9)


class Test_vis_viva:
    R = 7000 * KM_TO_M

    def test_circular_speed_hand_calc(self) -> None:
        """r = a = 7000 km: sqrt(3.986004418e14 / 7e6) = 7546.053 m/s (the usual ~7.5 km/s LEO)."""
        assert vis_viva(self.R, self.R, MU_EARTH) == pytest.approx(7546.053, abs=1e-3)

    def test_matches_rv_to_coes_on_hyperbola(self) -> None:
        """Independent route: `coes_to_rv` builds v from sqrt(mu / p), not vis-viva."""
        r, v = coes_to_rv(-25000e3, 1.3, 0.3, 0.4, 0.5, 0.7, MU_EARTH)
        assert norm(v) == pytest.approx(
            vis_viva(norm(r), -25000e3, MU_EARTH), rel=1e-12
        )

    def test_escape_speed_is_parabolic(self):
        v_esc = np.sqrt(2 * MU_EARTH / self.R)
        assert sma_from_rv(self.R, v_esc, MU_EARTH) == np.inf

    @pytest.mark.parametrize("a", [9000e3, -25000e3])  # ellipse and hyperbola
    def test_round_trip(self, a):
        v = vis_viva(self.R, a, MU_EARTH)
        assert sma_from_rv(self.R, v, MU_EARTH) == pytest.approx(a, rel=1e-12)

    def test_matches_rv_to_coes(self):
        """Same speed as long as the `r` and `a` are the same for an arbitrary orbit"""
        r, v = coes_to_rv(9000e3, 0.2, 0.3, 0.4, 0.5, 1.1, MU_EARTH)
        assert np.linalg.norm(v) == pytest.approx(
            vis_viva(np.linalg.norm(r), 9000e3, MU_EARTH), rel=1e-12
        )
        r, v = coes_to_rv(9000e3, 0.4, 12.3, 4.4, 4.5, 4.1, MU_EARTH)
        assert np.linalg.norm(v) == pytest.approx(
            vis_viva(np.linalg.norm(r), 9000e3, MU_EARTH), rel=1e-12
        )


class Test_COES:
    ELEMENTS = np.array([9000e3, 0.2, 0.3, 0.4, 0.5, 1.1])

    def test_elements_and_unpack_round_trip(self):
        coes = COES.from_array(self.ELEMENTS)
        assert np.array_equal(coes.elements(), self.ELEMENTS)
        a, e, i, raan, aop, ta = coes
        assert (a, e, i, raan, aop, ta) == tuple(self.ELEMENTS)

    def test_to_rv_matches_coes_to_rv(self):
        coes = COES.from_array(self.ELEMENTS)
        rv = coes.to_rv(MU_EARTH)
        r_ref, v_ref = coes_to_rv(*self.ELEMENTS, mu=MU_EARTH)
        assert rv.shape == (6,)
        assert np.array_equal(rv, np.concatenate((r_ref, v_ref)))

    def test_from_rv_round_trip(self):
        r, v = coes_to_rv(*self.ELEMENTS, mu=MU_EARTH)
        coes = COES.from_r_and_v(r, v, MU_EARTH)
        assert coes.elements() == pytest.approx(self.ELEMENTS, rel=1e-9)
        coes = COES.from_rv(np.concatenate((r, v)), MU_EARTH)
        assert coes.elements() == pytest.approx(self.ELEMENTS, rel=1e-9)

    def test_with_ta_changes_only_true_anomaly(self):
        coes = COES.from_array(self.ELEMENTS)
        moved = coes.with_ta(2.5)
        assert moved.ta == 2.5
        assert moved.elements()[:5] == pytest.approx(coes.elements()[:5])
        assert coes.ta == 1.1  # original unchanged (frozen)
        with pytest.raises(AttributeError):
            coes.ta = 0.0

    def test_delta_wraps_angles(self):
        before = COES(7000e3, 0.01, 0.5, 6.2, 0.1, 6.25)
        after = COES(7005e3, 0.02, 0.6, 0.1, 6.2, 0.05)
        d = after.delta(before)
        assert d.a == pytest.approx(5e3)
        assert d.e == pytest.approx(0.01)
        assert d.i == pytest.approx(0.1)
        assert d.raan == pytest.approx(
            0.1 - 6.2 + 2 * np.pi
        )  # small positive, not ~-6.1
        assert d.aop == pytest.approx(6.2 - 0.1 - 2 * np.pi)  # small negative
        assert d.ta == pytest.approx(0.05 - 6.25 + 2 * np.pi)

    def test_apsides_and_mean_motion(self):
        coes = COES(8000e3, 0.1, 0.0, 0.0, 0.0, 0.0)
        assert coes.r_p == pytest.approx(7200e3)
        assert coes.r_a == pytest.approx(8800e3)

    def test_geo_mean_motion_is_earth_rotation_rate(self):
        """Geostationary radius 42,164.17 km orbits once per sidereal day."""
        geo = COES(42164.17e3, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert geo.n(MU_EARTH) == pytest.approx(W_EARTH, rel=1e-7)


@pytest.mark.parametrize("e", [0.0, 0.1, 0.7])
@pytest.mark.parametrize("ta", [0.0, 0.5, 2.0, 3.5, 6.0])
def test_mean_anomaly_round_trip(ta, e):
    assert ta_from_M(M_from_ta(ta, e), e) == pytest.approx(ta, abs=1e-9)


class Test_orbit_sweep:
    COES0 = COES(8000e3, 0.2, 0.3, 0.4, 0.5, 0.1)

    def test_half_orbit_from_periapsis_is_half_period(self):
        start = self.COES0.with_ta(0.0)
        moved, dt = orbit_sweep(start, np.pi, MU_EARTH)
        assert moved.ta == pytest.approx(np.pi)
        assert moved.r_a == pytest.approx(
            norm(moved.to_r_and_v(MU_EARTH)[0])
        )  # at apoapsis
        assert dt == pytest.approx(start.period(MU_EARTH) / 2)

    def test_periapsis_to_90_deg_hand_calc(self):
        """a = 8000 km, e = 0.2, ta 0 -> 90 deg, by hand:

        tan(E/2) = sqrt((1-e)/(1+e)) tan(ta/2) = sqrt(0.8/1.2) * 1  ->  E = 1.369438 rad
        sin(E) = sqrt(1-e^2) sin(ta) / (1 + e cos(ta)) = sqrt(0.96)  = 0.979796
        M = E - e sin(E) = 1.369438 - 0.195959                       = 1.173479 rad
        n = sqrt(mu / a^3) = sqrt(3.986004418e14 / 5.12e20)           = 8.823358e-4 rad/s
        dt = M / n                                                     = 1329.969 s
        """
        _, dt = orbit_sweep(self.COES0.with_ta(0.0), np.pi / 2, MU_EARTH)
        assert dt == pytest.approx(1329.969, abs=1e-3)

    def test_propagating_dt_reaches_swept_anomaly(self):
        """Independent method: numerically coast for dt and read the true anomaly back."""
        moved, dt = orbit_sweep(self.COES0, 1.0, MU_EARTH)
        rv = propagate_two_body(self.COES0.to_rv(MU_EARTH), [0.0, dt], MU_EARTH)[-1]
        ta_reached = COES.from_rv(rv, MU_EARTH).ta
        assert wrap_to_pi(ta_reached - moved.ta) == pytest.approx(0.0, abs=1e-8)

    def test_negative_sweep_wraps_forward(self):
        _, dt_back = orbit_sweep(self.COES0, -1.0, MU_EARTH)
        assert 0 < dt_back < self.COES0.period(MU_EARTH)


def test_with_dv_prograde_raises_apoapsis_from_periapsis():
    coes = COES(7000e3, 0.01, 0.2, 0.3, 0.4, 0.0)
    v_hat = coes.to_rv(MU_EARTH)[3:6] / np.linalg.norm(coes.to_rv(MU_EARTH)[3:6])
    after = coes.with_dv(10.0 * v_hat, MU_EARTH)
    assert after.r_p == pytest.approx(coes.r_p, rel=1e-9)
    assert after.r_a > coes.r_a
    assert after.i == pytest.approx(coes.i) and after.raan == pytest.approx(coes.raan)
    unchanged = coes.with_dv(np.zeros(3), MU_EARTH).delta(coes)  # wraps ta = 0 vs 2pi
    assert unchanged.elements() == pytest.approx(np.zeros(6), abs=1e-6)


################################################################################
#               Kepler solvers
################################################################################


class Test_kepler_eq:
    def test_rejects_non_elliptical(self) -> None:
        for e in (-0.1, 1.0, 1.5):
            with pytest.raises(ValueError):
                kepler_eq(1.0, e)

    def test_raises_when_not_converged(self) -> None:
        with pytest.raises(ValueError):
            kepler_eq(1.0, 0.9, max_iter=1)


# class Test_kepler_eq_h:
#     @pytest.mark.parametrize(
#         "e, H",
#         [
#             (1.2, 0.8),  # e < 1.6, small M
#             (1.2, -0.8),  # negative: H is not an angle, must not wrap
#             (1.5, 3.0),  # e < 1.6, M > pi
#             (2.0, 0.5),  # 1.6 <= e < 3.6, |M| < pi
#             (2.4, -2.0),  # 1.6 <= e < 3.6, |M| > pi
#             (5.0, 0.5),  # e >= 3.6
#             # (1.1, 7.0),  # H > 2 pi: needs the large-|M| guess (TODO in kepler_eq_h)
#         ],
#     )
#     def test_inverts_hyperbolic_kepler_equation(self, e: float, H: float) -> None:
#         """Round trip through the defining equation M = e sinh(H) - H, every guess branch."""
#         M = e * np.sinh(H) - H
#         assert kepler_eq_h(M, e) == pytest.approx(H, abs=1e-9)
#
#     def test_rejects_non_hyperbolic(self) -> None:
#         with pytest.raises(ValueError):
#             kepler_eq_h(1.0, 1.0)
#
#     def test_raises_when_not_converged(self) -> None:
#         with pytest.raises(ValueError):
#             kepler_eq_h(10.0, 1.5, max_iter=1)


def test_kepler_eq_p_inverts_barker_equation() -> None:
    """Barker: dt = (1/2) sqrt(p^3 / mu) (B + B^3 / 3), with B = tan(ta / 2).

    Hand calc at ta = 60 deg: B = tan(30 deg) = 0.577350.
    """
    p = 10000e3
    B = np.tan(30 * DEG_TO_RAD)
    dt = 0.5 * np.sqrt(p**3 / MU_EARTH) * (B + B**3 / 3)
    assert kepler_eq_p(dt, p, MU_EARTH) == pytest.approx(0.577350, abs=1e-6)


class Test_pkepler:
    RV0 = coes_to_rv(7000e3, 0.01, 0.9, 0.5, 0.3, 0.2, MU_EARTH)

    def test_without_j2_matches_numerical_two_body(self) -> None:
        dt = 5000.0
        r, v = pkepler(*self.RV0, dt, MU_EARTH)
        rv = propagate_two_body(
            np.concatenate(self.RV0), [0.0, dt], MU_EARTH, rtol=1e-12, atol=1e-6
        )[-1]
        assert norm(r - rv[0:3]) < 1e-3  # m
        assert norm(v - rv[3:6]) < 1e-6  # m/s

    def test_j2_secular_drift_matches_numerical_j2(self) -> None:
        """10 days: pkepler has only secular J2, so it matches the numerical (osculating)
        RAAN drift to ~0.3% and aop drift to ~2% (short-period terms make up the rest)."""
        T = 10 * DAY_TO_SEC
        r, v = pkepler(*self.RV0, T, MU_EARTH, R_EARTH, J2_EARTH)
        rv = propagate_two_body(
            np.concatenate(self.RV0),
            [0.0, T],
            MU_EARTH,
            J2_EARTH,
            R_EARTH,
            rtol=1e-12,
            atol=1e-6,
        )[-1]
        c0, c_pk, c_num = (
            rv_to_coes(*self.RV0, MU_EARTH),
            rv_to_coes(r, v, MU_EARTH),
            rv_to_coes(rv[0:3], rv[3:6], MU_EARTH),
        )
        d_raan_pk, d_raan_num = (
            wrap_to_pi(c_pk[3] - c0[3]),
            wrap_to_pi(c_num[3] - c0[3]),
        )
        d_aop_pk, d_aop_num = wrap_to_pi(c_pk[4] - c0[4]), wrap_to_pi(c_num[4] - c0[4])
        assert d_raan_pk == pytest.approx(d_raan_num, rel=1e-2)
        assert d_aop_pk == pytest.approx(d_aop_num, rel=5e-2)
        assert c_pk[0] == pytest.approx(c0[0], rel=1e-12)  # no secular change in a

    def test_rejects_hyperbolic(self) -> None:
        r = np.array([7000e3, 0.0, 0.0])
        v = np.array([0.0, 1.2 * np.sqrt(2 * MU_EARTH / 7000e3), 0.0])
        with pytest.raises(ValueError):
            pkepler(r, v, 100.0, MU_EARTH)


def test_rv_to_coes_parabolic_has_infinite_a() -> None:
    """Escape speed sqrt(2 mu / r) perpendicular to r: e = 1 and a undefined (inf)."""
    r = np.array([7000e3, 0.0, 0.0])
    v = np.array([0.0, np.sqrt(2 * MU_EARTH / 7000e3), 0.0])
    a, e, *_ = rv_to_coes(r, v, MU_EARTH)
    assert a == np.inf
    assert e == pytest.approx(1.0)


################################################################################
#               COES helpers
################################################################################


class Test_COES_helpers:
    C = COES(7000e3, 0.1, 0.9, 0.5, 0.3, 0.2)

    def test_copy_is_equal_but_separate(self) -> None:
        c = self.C.copy()
        assert c == self.C
        assert c is not self.C

    def test_to_r_and_v_splits_to_rv(self) -> None:
        r, v = self.C.to_r_and_v(MU_EARTH)
        assert np.concatenate((r, v)) == pytest.approx(self.C.to_rv(MU_EARTH))

    def test_h_magnitude_and_direction(self) -> None:
        """|h| = sqrt(mu a (1 - e^2)); h_hat = [sin i sin raan, -sin i cos raan, cos i]."""
        a, e, i, raan, _, _ = self.C
        h = self.C.h(MU_EARTH)
        assert norm(h) == pytest.approx(np.sqrt(MU_EARTH * a * (1 - e**2)), rel=1e-12)
        h_hat = [np.sin(i) * np.sin(raan), -np.sin(i) * np.cos(raan), np.cos(i)]
        assert h / norm(h) == pytest.approx(h_hat, abs=1e-12)

    def test_E_sp_hand_calc(self) -> None:
        """a = 7000 km: -mu / (2a) = -3.986004418e14 / 1.4e7 = -2.8471460e7 m2/s2."""
        assert self.C.E_sp(MU_EARTH) == pytest.approx(-2.8471460e7, rel=1e-7)
        assert COES(-25000e3, 1.3, 0, 0, 0, 0).E_sp(MU_EARTH) > 0  # hyperbola: unbound

    def test_aol_is_angle_from_node_to_position(self) -> None:
        """Geometric: angle from the ascending node to r, measured along the motion.

        `angle_between` is unsigned (0 to pi), so this orbit keeps aol = 0.5 rad < pi
        and the second assert checks r is ahead of the node, not behind it.
        """
        r = self.C.to_rv(MU_EARTH)[0:3]
        node = np.array([np.cos(self.C.raan), np.sin(self.C.raan), 0.0])
        assert angle_between(node, r) == pytest.approx(self.C.aol, abs=1e-9)
        assert np.cross(node, r) @ self.C.h(MU_EARTH) > 0

    def test_bias_raan_rotates_orbit_about_z(self) -> None:
        d = 0.25
        rv = self.C.to_rv(MU_EARTH)
        rv_biased = self.C.bias_raan(d).to_rv(MU_EARTH)
        rot = dcm_z(d).T  # active rotation by +d about z
        assert rv_biased[0:3] == pytest.approx(rot @ rv[0:3], rel=1e-12)
        assert rv_biased[3:6] == pytest.approx(rot @ rv[3:6], rel=1e-12)
        assert self.C.bias_raan(d).raan == pytest.approx(self.C.raan + d)

    def test_str_in_km_and_deg(self) -> None:
        c = COES(7000e3, 0.01, 45 * DEG_TO_RAD, 90 * DEG_TO_RAD, 0.0, 180 * DEG_TO_RAD)
        assert str(c) == (
            "a = 7000.000 km, e = 0.01000, i = 45.000 deg, raan = 90.000 deg, "
            "aop = 0.000 deg, ta = 180.000 deg"
        )

    def test_print_with_and_without_energy(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Hand calc: E_sp = -mu / (2a) = -3.986004418e14 / 14e6 = -28.471 km2/s2."""
        c = COES(7000e3, 0.0, 0.0, 0.0, 0.0, 0.0)
        c.print("leo")
        c.print("leo", MU_EARTH)
        out = capsys.readouterr().out.splitlines()
        assert out[0] == f"leo: {c}"
        assert out[1] == f"leo: {c} (E_sp = -28.471 km2/s2)"
