import numpy as np
import pytest

from cesium.constants import DEG_TO_RAD, KM_TO_M, MIN_TO_SEC, RAD_TO_DEG
from cesium.orbital.coes import (
    E_from_ta,
    coes_to_rv,
    kepler_eq,
    kepler_eq_h,
    kepler_eq_p,
    rv_to_coes,
)
from cesium.world import MU_EARTH, MU_EARTH_KM


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


def test_kepler_eq_h_vallado():
    """Vallado Example 2-3, p. 71"""
    M = 235.4 * DEG_TO_RAD
    e = 2.4
    H = kepler_eq_h(M, e)

    assert H == pytest.approx(1.601376144, abs=1e-9)


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
        """hyperbolic orbit semi-major axis should be negative"""
        r = np.array([7000.0, 0.0, 0.0])  # km
        v = np.array([0.0, 15.0, 0.0])  # km/s

        a, e, i, raan, aop, ta = rv_to_coes(r, v, MU_EARTH_KM)

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
