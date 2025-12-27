import sys
import numpy as np
import pytest
from numpy.linalg import norm
sys.path.append(".")

import csim.math.transformations as Trans
from csim.world import W_EARTH
from csim.constants import DEG_TO_RAD, ARCSEC_TO_RAD, RAD_TO_ARCSEC, RAD_TO_DEG
from csim.math.time import greg_to_jd

#########################################################################################################
#               LLA to R
#########################################################################################################

def test_lla_to_r_example_3_1():
    """Vallado 4e Example 3.1 (p. 141)"""
    lat_geodetic = 39.586667
    lon_geodetic = -105.640
    h_MSL = 4347.667
    r = Trans.surface_lla_to_ecef(lat_geodetic, lon_geodetic, h_MSL)

    r_horizontal = np.sqrt(r[0]**2 + r[1]**2)
    r_vert = r[2]

    # Book values are given in km
    assert r_horizontal == pytest.approx(4925.4298026 * 1e3) # m
    assert r_vert == pytest.approx(4045.4937426 * 1e3) # m
    assert np.arctan(r[1] / r[0]) == pytest.approx(DEG_TO_RAD * (lon_geodetic + 180))

def test_lla_to_r_example_3_2():
    """Vallado 4e Example 3.2 (p. 144)"""
    lat_geodetic = -(7 + 54/60 + 23.886/3600)
    lon_geodetic = 345 + 35/60 + 51/3600
    h_MSL = 56
    r = Trans.surface_lla_to_ecef(lat_geodetic, lon_geodetic, h_MSL)

    # Book values are given in km
    assert r[0] == pytest.approx(6119.400269 * 1e3)
    assert r[1] == pytest.approx(-1571.47955545 * 1e3)
    assert r[2] == pytest.approx(-871.56118090 * 1e3)

def test_r_to_lla_example_3_3():

    # Book values are given in km
    r = np.array([6524.834, 6862.875, 6448.296]) * 1e3

    lat, lon, alt = Trans.r_to_surface_lla(r)

    # Book values are given in km
    assert lat == pytest.approx(34.352496 * DEG_TO_RAD, rel=1e-5)
    assert lon == pytest.approx(46.4464 * DEG_TO_RAD)
    assert alt == pytest.approx(5085.22 * 1e3)

#########################################################################################################
#               ITRF
#########################################################################################################

##################################################
#                       W
##################################################

def test_vallado_itrf_W():
    """ Tests that determinant is 1. \\
        Vallado 4e Example 3-14 p. 220"""
    t_tt = 0.0426236319
    xp = -0.140682 * ARCSEC_TO_RAD
    yp = 0.333309 * ARCSEC_TO_RAD

    W = Trans.W_matrix(xp, yp, t_tt)

    # Tests that determinant is 1
    assert np.linalg.det(W) == pytest.approx(1, 1e-9)

    # Transforms vector correctly
    r_itrf = np.array([-1033.479383, 7901.2952754, 6380.3565958])
    r_tirs = W @ r_itrf
    assert np.allclose(r_tirs, np.array([-1033.4750312, 7901.3055856, 6380.3445327]))

##################################################
#                       R
##################################################

def test_vallado_itrf_R():
    """ Tests that determinant is 1 \\
        Vallado 4e Example 3-14 p. 220"""

    jd_ut1 = greg_to_jd(2004, 4, 6, 7, 51, 27.946047 + 0.4399619)

    R = Trans.R_matrix(jd_ut1)

    # Tests that determinant is 1 and checks rotation angle 
    assert np.linalg.det(R) == pytest.approx(1, 1e-9)
    assert R[0,0] == pytest.approx(np.cos(312.7552829 * DEG_TO_RAD), rel=1e-4)

    # Transforms vector correctly
    r_tirs = np.array([-1033.4750312, 7901.3055856, 6380.3445327])
    r_cirs = R @ r_tirs
    assert np.allclose(r_cirs, np.array([5100.0184047, 6122.7863648, 6380.3445327]), rtol=1e-3)

def test_vallado_itrf_R_vel():
    jd_ut1 = greg_to_jd(2004, 4, 6, 7, 51, 27.946047 + 0.4399619)

    R = Trans.R_matrix(jd_ut1)

    r_tirs = np.array([-1033.4750312, 7901.3055856, 6380.3445327])
    v_tirs = np.array([-3.22563652, -2.87245145, 5.531924446])
    
    v_expected = np.array([-4.745380330, 0.790341453, 5.531931288])

    # Checks to make sure vector is the same length
    v_cirs = Trans.calc_v_tirs(R, v_tirs, np.array([0,0,W_EARTH]), r_tirs)

    assert norm(v_cirs) == pytest.approx( norm(v_expected) )

    # Transforms velocity vector correctly
    assert np.allclose(v_cirs, v_expected, rtol=1e-3)

##################################################
#                       PN
##################################################

def test_vallado_itrf_XYsa_const():
    """ Tests X,Y,s,a are calculated correctly for PN matrix\\
        Vallado 4e Example 3-14 p. 220"""
    t_tt = 0.0426236319
    X,Y,s,a = Trans._X_Y_s_a(t_tt)

    assert X * RAD_TO_ARCSEC == pytest.approx(80.531880, rel=1e-6)
    assert Y * RAD_TO_ARCSEC == pytest.approx(7.273921, rel=1e-5)
    assert s * RAD_TO_ARCSEC == pytest.approx(-0.003027, rel=1e-3)
    assert a * RAD_TO_DEG == pytest.approx(28.647891, rel=1e-6)

def test_vallado_itrf_PN():
    """Tests precession-nutation matrix
    Vallado 4e Example 3-14 p. 220"""

    t_tt = 0.0426236319
    dX = -0.000205
    dY = -0.000136
    
    PN = Trans.PN_matrix(t_tt, dX, dY)

    # Tests that determinant is 1
    assert np.linalg.det(PN) == pytest.approx(1, 1e-9)

    # Transforms vector correctly
    r_cirs = np.array([5100.0184047, 6122.7863648, 6380.3445327])
    r_gcrs = PN @ r_cirs
    assert np.allclose(r_gcrs, np.array([5102.508959, 6123.011403, 6378.136925]))

##################################################
#                       Position
##################################################

def test_vallado_full_itrf():
    # Book values are given in km
    r_itrf = np.array([-1033.479383, 7901.2952754, 6380.3565958])
    expected = np.array([5102.508959, 6123.011403, 6378.136925])
    jd = greg_to_jd(2004, 4, 6, 7, 51, 28.386009)

    xp = -0.140682 * ARCSEC_TO_RAD
    yp = 0.333309 * ARCSEC_TO_RAD
    dAT = 32
    dUT1 = 0.4399619
    dX = -0.000205 * ARCSEC_TO_RAD
    dY = -0.000136 * ARCSEC_TO_RAD

    PN, R, W = Trans.itrf_to_gcrs_matrices(xp, yp, jd, dX, dY, dUT1, dAT)

    # Tests that determinant is 1
    assert np.linalg.det(PN @ R @ W) == pytest.approx(1, 1e-9)
    
    # Transforms vector correctly
    r_gcrs = PN @ R @ W @ r_itrf
    assert np.allclose(r_gcrs, expected, rtol=1e-4)


#TODO: velocity and acceleration