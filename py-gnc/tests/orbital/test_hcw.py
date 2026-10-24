import numpy as np
import pytest
from tqdm import tqdm

from cesium.orbital.hcw import rv_hcw_3d, hcw_dynamics
from cesium.math.integrators import rk4_func

from cesium.world import MU_EARTH, R_EARTH, W_EARTH
from cesium.constants import KM_TO_M, MIN_TO_SEC, HOUR_TO_SEC

def test_vallado_hubble():
    """Vallado 4e Example 6-14 p. 397"""
    r0_m = np.zeros(3)
    v0_m = np.array([-0.1, -0.04, -0.02])
    alt = 590 * KM_TO_M  # km
    n = np.sqrt(MU_EARTH / (R_EARTH+alt)**3)
    
    r_m, v_m = rv_hcw_3d(5 * MIN_TO_SEC, n, r0_m, v0_m)
    assert np.allclose(r_m, [-33.345, -1.473, -5.894], atol=0.001)
    assert np.allclose(v_m, [-0.12, 0.03, -0.02], atol=0.01)
    
    r_m, v_m = rv_hcw_3d(20 * MIN_TO_SEC, n, r0_m, v0_m)
    assert np.allclose(r_m, [-143.000, 137.279, -17.766], atol=0.001)
    assert np.allclose(v_m, [-0.10, 0.27, -0.01], atol=0.01)
    
def test_curtis_example_7_2():
    """Targeting"""
    pass
    
def test_curtis_example_7_3():
    """Targeting"""
    pass
    
# problems 1,2,5
def test_curtis_problem_7_7():
    r0 = np.array([1,0,0]) * KM_TO_M
    v0 = np.array([0, 10, 0])
    
    n = 2*np.pi / (90 * MIN_TO_SEC)
    
    r,v = rv_hcw_3d(15 * MIN_TO_SEC, n, r0, v0)
    
    assert np.linalg.norm(r) == pytest.approx(11.2 * KM_TO_M, abs=0.05 * KM_TO_M)
    
def test_curtis_problem_7_11():
    """
    Spacecraft A and B are in the same circular earth orbit with a period of 2 hours. B
    is 6 km ahead of A. At t = 0, B applies an in-track delta-v (retrofire) of 3 m/s. Using
    a Clohessy–Wiltshire frame attached to A, determine the distance between A and B at
    t = 30 minutes and the velocity of B relative to A.
    {Ans.: r= 10.9 km, v= 10.8 m/s}
    """
    r0 = np.array([0,6,0]) * KM_TO_M # Book had it radially I guess
    v0 = np.array([0,-3,0])
    
    n = 2*np.pi / (2 * HOUR_TO_SEC)
    
    r,v = rv_hcw_3d(30 * MIN_TO_SEC, n, r0, v0)
    
    assert np.linalg.norm(r) == pytest.approx(10.9 * KM_TO_M, abs=0.05 * KM_TO_M)
    assert np.linalg.norm(v) == pytest.approx(10.8, abs=0.05)

class TestHcwPropagation:

    @staticmethod
    def hcw_integrated(rv0: np.ndarray, n: float, tspan: float, dt: float):
        rv = rv0.copy()
        t = 0

        def hcw_f(t, x, p):
            return hcw_dynamics(x, np.zeros(3), n)

        print(tspan//dt)
        for _ in tqdm(range(0, int(tspan/dt) + 1)):
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
    @pytest.fixture(params=[
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
    ], ids=lambda x: x[0])
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