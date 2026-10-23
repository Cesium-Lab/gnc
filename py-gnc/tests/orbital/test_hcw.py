import sys
import numpy as np
import pytest


from cesium.orbital.hcw import rv_hcw_3d
from cesium.orbital.coes import rv_to_coes, coes_to_rv

from cesium.world import MU_EARTH, R_EARTH
from cesium.constants import KM_TO_M, MIN_TO_SEC, DEG_TO_RAD


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
    
def test_curtis_example_7_1():
    """Motion"""
    h_A, e_A= 52059*KM_TO_M*KM_TO_M, 0.025724
    r_A, v_A = coes_to_rv(
        h_A*h_A/MU_EARTH / (1 - e_A*e_A),
        e_A,
        60*DEG_TO_RAD,
        40*DEG_TO_RAD,
        30*DEG_TO_RAD,
        40*DEG_TO_RAD)
    
    h_B, e_B= 52362*KM_TO_M*KM_TO_M, 0.0072696
    r_B, v_B = coes_to_rv(
        h_B*h_B/MU_EARTH / (1 - e_B*e_B),
        e_B,
        50*DEG_TO_RAD,
        40*DEG_TO_RAD,
        120*DEG_TO_RAD,
        40*DEG_TO_RAD)
    
    print(r_A, v_A)
    print(r_B, v_B)

    pass
    
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
    
    
def test_curtis_problem_7_8():
    """
    A space station is in a circular earth orbit of radius 6600 km. An approaching spacecraft executes a delta-v burn when its position vector relative to the space station is
    r=[1,1,1] (km). Just before the burn the relative velocity of the spacecraft
    was v=[0,0,5] (m/s). Calculate the total delta-v required for the space shuttle
    to rendezvous with the station in one-third period of the space station orbit.
    {Ans.: 6.21 m/s}
    """
    r_t = 6600*KM_TO_M
    r0 = np.array([1,0,0]) * KM_TO_M
    v0 = np.array([0, 10, 0])
    
    n = np.sqrt(MU_EARTH/r_t**3)
    period = 2*np.pi/n
    
    r,v = rv_hcw_3d(period / 3, n, r0, v0)
    
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
    
