import sys
import numpy as np
import pytest
from numpy.linalg import norm

from cesium.math.base import unit
import cesium.orbital.transformations as Trans
from cesium.constants import DEG_TO_RAD, ARCSEC_TO_RAD, RAD_TO_ARCSEC, RAD_TO_DEG, M_TO_KM, KM_TO_M
from cesium.math.time import greg_to_jd

#########################################################################################################
#               LLA to R
#########################################################################################################


def test_lla_to_r_example_3_1():
    """Vallado 4e Example 3.1 (p. 141)"""
    lat_geodetic = 39.586667
    lon_geodetic = -105.640
    h_MSL = 4347.667 # m
    r = Trans.geodetic_LLA_to_ECEF(lat_geodetic, lon_geodetic, h_MSL)

    r_horizontal = np.sqrt(r[0] ** 2 + r[1] ** 2)
    r_vert = r[2]

    # Book values are given in km
    assert r_horizontal == pytest.approx(4925.4298026 * KM_TO_M)  # m
    assert r_vert == pytest.approx(4045.4937426 * KM_TO_M)  # m
    assert np.arctan(r[1] / r[0]) == pytest.approx(DEG_TO_RAD * (lon_geodetic + 180))


def test_lla_to_r_example_3_2():
    """Vallado 4e Example 3.2 (p. 144)"""
    lat_geodetic = -(7 + 54 / 60 + 23.886 / 3600)
    lon_geodetic = 345 + 35 / 60 + 51 / 3600
    h_MSL = 56
    r = Trans.geodetic_LLA_to_ECEF(lat_geodetic, lon_geodetic, h_MSL)

    # Book values are given in km
    assert r[0] == pytest.approx(6119.400269 * KM_TO_M)
    assert r[1] == pytest.approx(-1571.47955545 * KM_TO_M)
    assert r[2] == pytest.approx(-871.56118090 * KM_TO_M)


def test_r_to_lla_example_3_3():

    # Book values are given in km
    r = np.array([6524.834, 6862.875, 6448.296]) * KM_TO_M

    lat, lon, alt = Trans.ECEF_to_geodetic_LLA(r)

    # Book values are given in km
    assert lat == pytest.approx(34.352496, rel=1e-5)
    assert lon == pytest.approx(46.4464, rel=1e-5)
    assert alt == pytest.approx(5085.22 * KM_TO_M)
    
#########################################################################################################
#               RSW
#########################################################################################################

def test_PCI_to_RSW_Curtis_example_7_2():
    """Curtis Example 7.2 p. 331"""
    
    # From book
    r_t = np.array([1622.39, 5305.10, 3717.44]) * KM_TO_M
    v_t = np.array([-7.29977, 0.492357, 2.48318]) * KM_TO_M
    r_c = np.array([ 1612.75, 5310.19, 3750.33]) * KM_TO_M
    v_c = np.array([-7.35321, 0.463856, 2.46920]) * KM_TO_M
    
    dcm_RSW_wrt_ECI = Trans.PCI_to_RSW(np.concat((r_t, v_t)))

    w_t_ECI = np.cross(r_t, v_t) / np.linalg.norm(r_t)**2
    
    r0_c_ECI = r_c - r_t
    r0_c_RSW = dcm_RSW_wrt_ECI @ r0_c_ECI
    v0_c_ECI = (v_c - v_t) - np.cross(w_t_ECI, r0_c_ECI)
    v0_c_RSW = dcm_RSW_wrt_ECI @ v0_c_ECI
    
    # sus bounds though
    assert r0_c_RSW == pytest.approx([20000, 20000, 20000], abs=20)
    assert v0_c_RSW == pytest.approx([-20, 20, -5], abs=1.1)
    
#########################################################################################################
#               cart to RaDec
#########################################################################################################

def test_cartesian_to_radec():
    """Curtis Equation 4.5 p. 155"""
    r = np.array([-5368, -1784, 3691])
    
    ra, dec, rho = Trans.cartesian_to_radec(r, deg=True)
    assert ra == pytest.approx(198.4, abs=0.1)
    assert dec == pytest.approx(33.12, abs=0.01)
    assert rho == pytest.approx(norm(r))
    
def test_radec_roundtrip():
    r0 = np.array([-536328, -17132484, 3234691])
    
    ra, dec, rho = Trans.cartesian_to_radec(r0, deg=True)
    r = Trans.radec_to_cartesian(ra, dec, rho, deg=True)
    assert r == pytest.approx(r0)
    
    ra, dec, rho = Trans.cartesian_to_radec(r0, deg=False)
    r = Trans.radec_to_cartesian(ra, dec, rho, deg=False)
    assert r == pytest.approx(r0)
    

#########################################################################################################
#               ITRF
#########################################################################################################

##################################################
#                       W
##################################################


def test_vallado_itrf_W():
    """Tests that the W matrix has determinant 1 and transforms ITRF to TIRS.

    Vallado 4e Example 3-14 p. 220
    """
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
    """Tests that the R matrix has determinant 1 and transforms TIRS to CIRS.

    Vallado 4e Example 3-14 p. 220
    """

    jd_ut1 = greg_to_jd(2004, 4, 6, 7, 51, 27.946047 + 0.4399619)

    R = Trans.R_matrix(jd_ut1)

    # Tests that determinant is 1 and checks rotation angle
    assert np.linalg.det(R) == pytest.approx(1, 1e-9)
    assert R[0, 0] == pytest.approx(np.cos(312.7552829 * DEG_TO_RAD), rel=1e-4)

    # Transforms vector correctly
    r_tirs = np.array([-1033.4750312, 7901.3055856, 6380.3445327])
    r_cirs = R @ r_tirs
    assert np.allclose(
        r_cirs, np.array([5100.0184047, 6122.7863648, 6380.3445327]), rtol=1e-3
    )


##################################################
#                       PN
##################################################


def test_vallado_itrf_XYsa_const():
    """Tests X, Y, s, and a are calculated correctly for the PN matrix.

    Vallado 4e Example 3-14 p. 220
    """
    t_tt = 0.0426236319
    X, Y, s, a = Trans._X_Y_s_a(t_tt)

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

    eop = Trans.EOP(
        xp=-0.140682 * ARCSEC_TO_RAD,
        yp=0.333309 * ARCSEC_TO_RAD,
        dX=-0.000205 * ARCSEC_TO_RAD,
        dY=-0.000136 * ARCSEC_TO_RAD,
        deltaUT1_s=-0.4399619,
        deltaAT_s=32,
    )

    PN, R, W = Trans.itrf_to_gcrs_matrices(jd, eop)

    # Tests that determinant is 1
    assert np.linalg.det(PN @ R @ W) == pytest.approx(1, 1e-9)

    # Transforms vector correctly
    r_gcrs = PN @ R @ W @ r_itrf
    assert np.allclose(r_gcrs, expected, rtol=1e-4)


def test_vallado_full_itrf_rv():
    """Vallado 4e Example 3-14 p. 220"""
    # Book values are given in km
    r_itrf = np.array([-1033.479383, 7901.2952754, 6380.3565958])
    v_itrf = np.array([-3.225636520, -2.872451450, 5.531924446])
    r_expected = np.array([5102.508959, 6123.011403, 6378.136925])
    v_expected = np.array([-4.743220161, 0.790536495, 5.533755724])
    jd = greg_to_jd(2004, 4, 6, 7, 51, 28.386009)

    eop = Trans.EOP(
        xp=-0.140682 * ARCSEC_TO_RAD,
        yp=0.333309 * ARCSEC_TO_RAD,
        dX=-0.000205 * ARCSEC_TO_RAD,
        dY=-0.000136 * ARCSEC_TO_RAD,
        deltaUT1_s=-0.4399619,
        deltaAT_s=32,
        lod_s=0.0015563,
    )

    r_gcrs, v_gcrs = Trans.itrf_to_gcrs(r_itrf, v_itrf, jd, eop)

    assert np.allclose(r_gcrs, r_expected, rtol=1e-4)
    assert np.allclose(v_gcrs, v_expected, rtol=1e-4)


# TODO: acceleration
