import numpy as np
from numpy.linalg import norm
import pytest


from cesium.orbital.transformations import PCI_to_RSW
from cesium.orbital.hcw import calc_Phi, rv_hcw_3d, rv_hcw_2d
from cesium.guidance.hcw import guid_hcw_two_impulse
from cesium.constants import KM_TO_M, HOUR_TO_SEC, MIN_TO_SEC, DAY_TO_SEC
from cesium.world import MU_EARTH
from cesium.physics.gravity import grav_accel
            
####################################################################################################
#               Curtis Book Problems
####################################################################################################

def test_example_7_1():
    """Motion"""
    
    # rv from book
    r_A_ECI = np.array([-266.74, 3865.4, 5425.7]) * KM_TO_M
    v_A_ECI = np.array([-6.4842, -3.6201, 2.4159]) * KM_TO_M

    r_B_ECI = np.array([-5890.0, -2979.4, 1792.0]) * KM_TO_M
    v_B_ECI = np.array([0.93594, -5.2409, -5.5016]) * KM_TO_M
    r_B_A_ECI = r_B_ECI - r_A_ECI
    
    # accel (mm precision)
    a_A_ECI = grav_accel(r_A_ECI)
    a_B_ECI = grav_accel(r_B_ECI)

    # w = r0 x v0 / r0^2
    w_A_ECI = np.cross(r_A_ECI, v_A_ECI) / norm(r_A_ECI)**2
    
    # w_dot = -2*dot(r,v)/r0^2 * w
    w_dot_A_ECI = -2*np.dot(r_A_ECI, v_A_ECI) * w_A_ECI / norm(r_A_ECI)**2

    # mm/s accuracy
    v_B_A_ECI = v_B_ECI - v_A_ECI - np.cross(w_A_ECI, r_B_A_ECI)

    a_B_A_ECI = a_B_ECI -(
        a_A_ECI
        + np.cross(w_dot_A_ECI, r_B_A_ECI)
        + np.cross(w_A_ECI, np.cross(w_A_ECI, r_B_A_ECI))
        + 2*np.cross(w_A_ECI, v_B_A_ECI)
    )
    
    dcm_RSW_wrt_ECI = PCI_to_RSW(np.concat((r_A_ECI, v_A_ECI)))

    r_B_A_RSW = dcm_RSW_wrt_ECI @ r_B_A_ECI
    v_B_A_RSW = dcm_RSW_wrt_ECI @ v_B_A_ECI
    a_B_A_RSW = dcm_RSW_wrt_ECI @ a_B_A_ECI
    
    assert a_A_ECI == pytest.approx([0.35876, -5.1989, -7.2975], rel=1e-3)
    assert a_B_ECI == pytest.approx([7.3377, 3.7117, -2.2325], rel=1e-3)
    
    assert v_B_A_ECI == pytest.approx([588.65, -696.92, 914.14], rel=1e-3)
    assert a_B_A_ECI == pytest.approx([0.43984, -0.38019, 0.017988], rel=1e-3)
    
    assert r_B_A_RSW == pytest.approx([-6700500, 6827400, -406220], rel=1e-4)
    assert v_B_A_RSW == pytest.approx([316.32, 111.99, 1247.1], rel=1e-4)
    assert a_B_A_RSW == pytest.approx([-0.22338, -0.17980, 0.50607], rel=1e-4)

def test_example_7_2():
    """Curtis Example 7.2 p. 331"""
    
    # From book
    # r_t = np.array([1622.39, 5305.10, 3717.44]) * KM_TO_M
    # v_t = np.array([-7.29977, 0.492357, 2.48318]) * KM_TO_M
    # r_c = np.array([ 1612.75, 5310.19, 3750.33]) * KM_TO_M
    # v_c = np.array([-7.35321, 0.463856, 2.46920]) * KM_TO_M

    
    
    
    # dcm_RSW_wrt_ECI = PCI_to_RSW(np.concat((r_t, v_t)))

    # w_t_ECI = np.cross(r_t, v_t) / norm(r_t)**2
    # w_t_RSW = dcm_RSW_wrt_ECI @ w_t_ECI # also from norm
    
    # r0_c_ECI = r_c - r_t
    # r0_c_RSW = dcm_RSW_wrt_ECI @ r0_c_ECI
    # v0_c_ECI = (v_c - v_t) - np.cross(w_t_ECI, r0_c_ECI)
    # v0_c_RSW = dcm_RSW_wrt_ECI @ v0_c_ECI
    
    r0 = np.array([20, 20, 20]) * KM_TO_M
    v0 = np.array([-0.02, 0.02, -0.005]) * KM_TO_M
    t_trans = 8*HOUR_TO_SEC
    w = 0.00115697 
    
    # HCW!
    Phi_rr, Phi_rv, Phi_vr, Phi_vv = calc_Phi(t_trans, w)
    
    # Solve
    v0_init_RSW = -np.linalg.inv(Phi_rv) @ Phi_rr @ r0
    v0_final_RSW = Phi_vr @ r0 + Phi_vv @ v0_init_RSW
    
    dv_tot = np.linalg.norm(v0_init_RSW - v0) + np.linalg.norm( -v0_final_RSW)
    
    # Seems like not very good approx but I sort of got the right answer after lots of rounding
    assert dv_tot == pytest.approx(109.7, rel=1e-3)
        

def test_problem_7_8():
    """
    A space station is in a circular earth orbit of radius 6600 km. An approaching spacecraft executes a
    delta-v burn when its position vector relative to the space station is
    r=[1,1,1] (km). Just before the burn the relative velocity of the spacecraft
    was v=[0,0,5] (m/s). Calculate the total delta-v required for the space shuttle
    to rendezvous with the station in one-third period of the space station orbit.
    {Ans.: 6.21 m/s}
    
    Curtis, Problem 7.8. p. 343
    """
    r_t = 6600*KM_TO_M
    r0 = np.array([1,1,1]) * KM_TO_M
    v0 = np.array([0,0,5])

    n = np.sqrt(MU_EARTH/r_t**3)
    period = 2*np.pi/n
    
    dv1, dv2, _ = guid_hcw_two_impulse(r0, v0, period / 3, n)
    dv = norm(dv1) + norm(dv2)

    assert dv == pytest.approx(6.21, abs=0.01)

def test_problem_7_11():
    """
    Spacecraft A and B are in the same circular earth orbit with a period of 2 hours. B
    is 6 km ahead of A. At t = 0, B applies an in-track delta-v (retrofire) of 3 m/s. Using
    a Clohessy–Wiltshire frame attached to A, determine the distance between A and B at
    t = 30 minutes and the velocity of B relative to A.
    {Ans.: ||r|| = 10.9 km, ||v|| = 10.8 m/s}
    
    Curtis, Problem 7.11. p. 344
    """
    r0 = np.array([0,6,0]) * KM_TO_M
    v0 = np.array([0,-3,0])

    period = 2*HOUR_TO_SEC
    n = 2*np.pi/period
    
    r,v = rv_hcw_3d(30*MIN_TO_SEC, n, r0, v0)

    assert norm(r)/KM_TO_M == pytest.approx(10.9, abs=0.1)
    assert norm(v) == pytest.approx(10.8, abs=0.1)
    
def test_problem_7_12():
    """
    A GEO satellite strikes some orbiting debris and is found 2 hours afterwards to have
    drifted to the position r=[-10 10 0] km relative to its original location. At that
    time the only slightly damaged satellite initiates a two-impulse maneuver to return to its
    original location in 6 hours. Find the total delta-v for this maneuver.
    {Ans.: 3.5 m/s}
    
    Curtis, Problem 7.12. p. 344
    """
    rf = np.array([-10,10,0]) * KM_TO_M

    n = 2*np.pi/DAY_TO_SEC
    
    # Finding velocity after 2 hours
    _, Phi_rv, _, Phi_vv = calc_Phi(2*HOUR_TO_SEC, n)
    v0 = np.linalg.inv(Phi_rv) @ rf
    vf = Phi_vv @ v0
    
    # Getting back!!
    dv1, dv2, _ = guid_hcw_two_impulse(rf, vf, 6*HOUR_TO_SEC, n)
    dv = norm(dv1) + norm(dv2)
    
    assert dv == pytest.approx(3.5, abs=0.1)

def test_problem_7_14():
    """
    The space station is in a circular earth orbit of radius 6600 km. The space shuttle is also
    in a circular orbit in the same plane as the space station’s. At the instant that the position
    of the shuttle relative to the space station, in Clohessy–Wiltshire coordinates, is (5 km,
    0, 0). What is the relative velocity of the space shuttle in meters/s?
    {Ans.: 8.83 m/s}
    
    Curtis, Problem 7.14. p. 344
    """
    r_t = 6600 * KM_TO_M
    r_c_t = 5*KM_TO_M
    
    w_t = np.sqrt(MU_EARTH/r_t**3)
    
    v_t = np.sqrt(MU_EARTH/r_t)
    v_c = np.sqrt(MU_EARTH/(r_c_t + r_t))
    
    v_rel = norm(v_c - v_t - w_t*r_c_t)
    assert v_rel == pytest.approx(8.83, abs=0.1)
        
def test_problem_7_15():
    """
    The space station is in a circular earth orbit of radius 6600 km. The space shuttle is also
    in a circular orbit in the same plane as the space station’s. At the instant that the position
    of the shuttle relative to the space station, in Clohessy–Wiltshire coordinates, is (5 km,
    0, 0). What is the relative velocity of the space shuttle in meters/s?
    {Ans.: 8.83 m/s}
    
    Curtis, Problem 7.15. p. 345
    """
    # only doing 2D
    
    # random
    dr = 12
    n = 5
    
    r0 = np.array([dr, np.pi*dr])
    v0 = np.array([np.pi/16*n*dr, -7/4*n*dr])
    t_trans = np.pi/2/n
    
    r,_ = rv_hcw_2d(t_trans, n, r0, v0)
    
    assert norm(r) == pytest.approx(0.900*dr, abs=0.001*dr)
        

        
        

