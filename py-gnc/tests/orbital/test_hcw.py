import numpy as np
import pytest
from tqdm import tqdm

from cesium.constants import HOUR_TO_SEC, KM_TO_M, MIN_TO_SEC
from cesium.math.integrators import rk4_func
from cesium.orbital.coes import coes_to_rv
from cesium.orbital.hcw import (
    calc_Phi,
    hcw_dynamics,
    hcw_pci_to_rsw,
    hcw_rsw_to_pci,
    rv_hcw_3d,
)
from cesium.physics.dynamics import propagate_two_body
from cesium.world import MU_EARTH, R_EARTH, W_EARTH


def test_vallado_hubble():
    """Vallado 4e Example 6-14 p. 397"""
    r0_m = np.zeros(3)
    v0_m = np.array([-0.1, -0.04, -0.02])
    alt = 590 * KM_TO_M  # km
    n = np.sqrt(MU_EARTH / (R_EARTH + alt) ** 3)

    r_m, v_m = rv_hcw_3d(5 * MIN_TO_SEC, n, r0_m, v0_m)
    assert np.allclose(r_m, [-33.345, -1.473, -5.894], atol=0.001)
    assert np.allclose(v_m, [-0.12, 0.03, -0.02], atol=0.01)

    r_m, v_m = rv_hcw_3d(20 * MIN_TO_SEC, n, r0_m, v0_m)
    assert np.allclose(r_m, [-143.000, 137.279, -17.766], atol=0.001)
    assert np.allclose(v_m, [-0.10, 0.27, -0.01], atol=0.01)

# problems 1,2,5
def test_curtis_problem_7_7():
    r0 = np.array([1, 0, 0]) * KM_TO_M
    v0 = np.array([0, 10, 0])

    n = 2 * np.pi / (90 * MIN_TO_SEC)

    r, _v = rv_hcw_3d(15 * MIN_TO_SEC, n, r0, v0)

    assert np.linalg.norm(r) == pytest.approx(11.2 * KM_TO_M, abs=0.05 * KM_TO_M)


def test_curtis_problem_7_11():
    """
    Spacecraft A and B are in the same circular earth orbit with a period of 2 hours. B
    is 6 km ahead of A. At t = 0, B applies an in-track delta-v (retrofire) of 3 m/s. Using
    a Clohessy-Wiltshire frame attached to A, determine the distance between A and B at
    t = 30 minutes and the velocity of B relative to A.
    {Ans.: r= 10.9 km, v= 10.8 m/s}
    """
    r0 = np.array([0, 6, 0]) * KM_TO_M  # Book had it radially I guess
    v0 = np.array([0, -3, 0])

    n = 2 * np.pi / (2 * HOUR_TO_SEC)

    r, v = rv_hcw_3d(30 * MIN_TO_SEC, n, r0, v0)

    assert np.linalg.norm(r) == pytest.approx(10.9 * KM_TO_M, abs=0.05 * KM_TO_M)
    assert np.linalg.norm(v) == pytest.approx(10.8, abs=0.05)


class TestHcwPropagation:
    @staticmethod
    def hcw_integrated(rv0: np.ndarray, n: float, tspan: float, dt: float):
        rv = rv0.copy()
        t = 0

        def hcw_f(t, x):
            return hcw_dynamics(x, np.zeros(3), n)

        print(tspan // dt)
        for _ in tqdm(range(int(tspan / dt) + 1)):
            h = min(dt, tspan - t)
            rv = rk4_func(t, h, rv, hcw_f)
            t += h

        return rv

    @classmethod
    def hcw_integrated_equal(cls, rv0, rvf, n, tspan, dt):
        rv_calc = cls.hcw_integrated(rv0, n, tspan, dt)
        assert rv_calc == pytest.approx(rvf, rel=1e-6, abs=1e-6)

    # ==================================================
    #               Test cases
    # ==================================================

    @staticmethod
    @pytest.fixture(
        params=[
            # Stationary
            (
                "origin",
                W_EARTH,
                np.zeros(6),
                np.zeros(6),
                60 * MIN_TO_SEC,
                0.1,
            ),
            (
                "along_track",
                W_EARTH,
                np.array([0, 1e3, 0, 0, 0, 0]),
                np.array([0, 1e3, 0, 0, 0, 0]),
                60 * MIN_TO_SEC,
                0.1,
            ),
            (
                "retro_along_track",
                W_EARTH,
                np.array([0, -1e3, 0, 0, 0, 0]),
                np.array([0, -1e3, 0, 0, 0, 0]),
                60 * MIN_TO_SEC,
                0.1,
            ),
            # Periodic relative orbit, one full period
            (
                "periodic_in_plane",
                W_EARTH,
                np.array([1e3, 0, 0, 0, -2 * W_EARTH * 1e3, 0]),
                np.array([1e3, 0, 0, 0, -2 * W_EARTH * 1e3, 0]),
                2 * np.pi / W_EARTH,
                1,
            ),
            # Periodic cross-track oscillation
            (
                "periodic_cross_track",
                W_EARTH,
                np.array([0, 0, 1e3, 0, 0, 0]),
                np.array([0, 0, -1e3, 0, 0, 0]),
                np.pi / W_EARTH,
                1,
            ),
            (
                "different_orbit_rate",
                0.001,
                np.array([500, 0, 0, 0, -1.0, 0]),
                np.array([500, 0, 0, 0, -1.0, 0]),
                2 * np.pi / 0.001,
                0.1,
            ),
        ],
        ids=lambda x: x[0],
    )
    def hcw_case(request):
        _, n, rv0, rvf, tspan, dt = request.param
        return rv0, rvf, n, tspan, dt

    @pytest.mark.slow
    def test_hcw_integrated(self, hcw_case):
        rv0, rvf, n, tspan, dt = hcw_case
        self.hcw_integrated_equal(rv0, rvf, n, tspan, dt)


# TODO: verify that it works by propagating in ECI and then transforming too
class TestHcwWithEci:
    pass


class Test_rsw_pci:
    R0 = R_EARTH + 500e3
    RV_TARGET = np.concatenate(coes_to_rv(R0, 0.0, 0.9, 0.4, 0.0, 1.3, MU_EARTH))

    def test_round_trips(self):
        rv_rel = np.array([1200.0, -3400.0, 250.0, 0.8, -1.1, 0.3])
        rv_chaser = hcw_rsw_to_pci(self.RV_TARGET, rv_rel)
        assert hcw_pci_to_rsw(self.RV_TARGET, rv_chaser) == pytest.approx(
            rv_rel, abs=1e-9
        )

        rv_chaser_pci = self.RV_TARGET + np.array(
            [900.0, -400.0, 300.0, 0.5, 0.2, -0.4]
        )
        rv_back = hcw_rsw_to_pci(
            self.RV_TARGET, hcw_pci_to_rsw(self.RV_TARGET, rv_chaser_pci)
        )
        assert rv_back == pytest.approx(rv_chaser_pci, abs=1e-9)

    def test_co_rotating_point_has_no_relative_velocity(self):
        """A point 1 km radially above the target, turning with it, sits still in RSW."""
        r_t, v_t = self.RV_TARGET[:3], self.RV_TARGET[3:]
        scale = (self.R0 + 1000.0) / self.R0
        rv_rel = hcw_pci_to_rsw(
            self.RV_TARGET, np.concatenate((r_t * scale, v_t * scale))
        )
        assert rv_rel == pytest.approx([1000.0, 0, 0, 0, 0, 0], abs=1e-9)

    def test_matches_hcw_propagation_for_nearby_orbits(self):
        """Two-body relative motion of a close chaser follows the HCW STM."""
        rv_rel0 = np.array([200.0, -500.0, 100.0, 0.1, -0.05, 0.02])
        rv_chaser0 = hcw_rsw_to_pci(self.RV_TARGET, rv_rel0)
        t_s = 20 * MIN_TO_SEC
        n = np.sqrt(MU_EARTH / self.R0**3)

        rv_t = propagate_two_body(
            self.RV_TARGET, [0.0, t_s], MU_EARTH, rtol=1e-12, atol=1e-9
        )[-1]
        rv_c = propagate_two_body(
            rv_chaser0, [0.0, t_s], MU_EARTH, rtol=1e-12, atol=1e-9
        )[-1]
        rv_rel = hcw_pci_to_rsw(rv_t, rv_c)

        Phi_rr, Phi_rv, Phi_vr, Phi_vv = calc_Phi(t_s, n)
        r_hcw = Phi_rr @ rv_rel0[:3] + Phi_rv @ rv_rel0[3:]
        v_hcw = Phi_vr @ rv_rel0[:3] + Phi_vv @ rv_rel0[3:]
        assert rv_rel[:3] == pytest.approx(
            r_hcw, abs=1.0
        )  # linearization error ~m at km scale
        assert rv_rel[3:] == pytest.approx(v_hcw, abs=1e-3)
