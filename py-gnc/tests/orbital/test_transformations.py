import numpy as np
import pytest
from numpy.linalg import norm

import cesium.orbital.transformations as Trans
from cesium.constants import (
    ARCSEC_TO_RAD,
    DEG_TO_RAD,
    KM_TO_M,
    M_TO_KM,
    RAD_TO_ARCSEC,
    RAD_TO_DEG,
)
from cesium.math.time import greg_to_jd

#########################################################################################################
#               LLA to R
#########################################################################################################


def test_lla_to_r_example_3_1():
    """Vallado 4e Example 3.1 (p. 141)"""
    lat_geodetic = 39.586667
    lon_geodetic = -105.640
    h_MSL = 4347.667  # m
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
#               SEZ
#########################################################################################################

def test_ecef_to_sez():
    # Book values are given in km
    r_ecef = np.array([6524.834, 6862.875, 6448.296]) * KM_TO_M
    lat, lon, alt_m = Trans.ECEF_to_geodetic_LLA(r_ecef)

    # Site on the ellipsoid directly underneath the satellite.
    r_site_ecef_m = Trans.geodetic_LLA_to_ECEF(lat, lon, 0)
    _, el, rho, _, _, _ = Trans.geodetic_ECEF_to_SEZ(r_ecef, np.zeros(3), lat, lon, 0)

    assert el == pytest.approx(90, rel=1e-5)
    assert rho == pytest.approx(norm(r_ecef - r_site_ecef_m))
    assert rho == pytest.approx(alt_m)


def test_ecef_to_sez_cardinal_directions():
    """Mainly tests azimuth. Edge cases"""
    r_site = Trans.geodetic_LLA_to_ECEF(0, 0)  # equator

    az, el, rho, _, _, _ = Trans.geodetic_ECEF_to_SEZ(
        r_site + [0, 1000, 0], np.zeros(3), 0, 0
    )
    assert np.allclose([az, el, rho], [90, 0, 1000])

    # arctan2 returns 0 for azimuth if no s direction
    az, el, rho, _, _, _ = Trans.geodetic_ECEF_to_SEZ(
        r_site + [0, 0, 1000], np.zeros(3), 0, 0
    )
    assert np.allclose([az, el, rho], [0, 0, 1000])

    az, el, rho, _, _, _ = Trans.geodetic_ECEF_to_SEZ(
        r_site + [0, -1000, 0], np.zeros(3), 0, 0
    )
    assert np.allclose([az, el, rho], [270, 0, 1000])

    az, el, rho, _, _, _ = Trans.geodetic_ECEF_to_SEZ(
        r_site + [0, 0, -1000], np.zeros(3), 0, 0
    )
    assert np.allclose([az, el, rho], [180, 0, 1000])


def test_ecef_to_sez_vertical_below_site():
    """Should still work"""
    r_site = Trans.geodetic_LLA_to_ECEF(0, 0)
    _az, el, rho, az_dot, el_dot, rho_dot = Trans.geodetic_ECEF_to_SEZ(
        r_site - [1000, 0, 0], np.array([-10, 0, 0]), 0, 0
    )
    assert np.allclose([el, rho], [-90, 1000])
    assert np.all(np.isnan([az_dot, el_dot]))
    assert rho_dot == pytest.approx(10)


def test_ecef_to_sez_geodetic_axes():
    """NOT on the ECEF cardinal directions though"""
    lat, lon = 45 * DEG_TO_RAD, 30 * DEG_TO_RAD
    r_site = Trans.geodetic_LLA_to_ECEF(lat, lon, 100, deg=False)
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0])
    az, el, rho, _, _, _ = Trans.geodetic_ECEF_to_SEZ(
        r_site + 1000 * up, np.zeros(3), lat, lon, 100, deg=False
    )

    assert az == pytest.approx(0)
    assert el == pytest.approx(np.pi / 2)
    assert rho == pytest.approx(1000)
    az, el, rho, _, _, _ = Trans.geodetic_ECEF_to_SEZ(
        r_site + 1000 * east + 1000 * up, np.zeros(3), 45, 30, 100
    )
    assert np.allclose([az, el, rho], [90, 45, np.sqrt(2) * 1000])


def test_ecef_to_sez_rates():
    r_site = Trans.geodetic_LLA_to_ECEF(0, 0)
    r_ecef = r_site + [1000, 1000, 0]  # rho = [0,1000,1000] -> 1000*sqrt(2)
    v_ecef = np.array([20, 10, 30])  # [-30,10,20] in sed, norm of sqrt(14)*10

    az, el, rho, az_dot, el_dot, rho_dot = Trans.geodetic_ECEF_to_SEZ(
        r_ecef, v_ecef, 0, 0, deg=False
    )

    # rho_dot = dot([0,1000,1000], [-30,10,20]) / 1000*sqrt(2)
    #         = 30 / sqrt(2)
    # az_dot = ((-30)*1000) / (1000^2) = -0.03
    # el_dot = (20 - 30/sqrt(2)*sin(np.pi / 4)) / 1000 = 0.005
    assert np.allclose([az, el, rho], [np.pi / 2, np.pi / 4, np.sqrt(2) * 1000])
    assert az_dot == pytest.approx(-0.03)
    assert el_dot == pytest.approx(0.005)
    assert rho_dot == pytest.approx(30 / np.sqrt(2))

    # works in degrees too
    az_deg, el_deg, rho_m, az_dot_deg, el_dot_deg, rho_dot_m_s = (
        Trans.geodetic_ECEF_to_SEZ(r_ecef, v_ecef, 0, 0)
    )
    assert np.allclose([az_deg, el_deg], np.array([az, el]) * RAD_TO_DEG)
    assert np.allclose(
        [az_dot_deg, el_dot_deg], np.array([az_dot, el_dot]) * RAD_TO_DEG
    )
    assert rho_m == pytest.approx(rho)
    assert rho_dot_m_s == pytest.approx(rho_dot)


#########################################################################################################
#               RSW
#########################################################################################################


def test_PCI_to_RSW_Curtis_example_7_2():
    """Curtis Example 7.2 p. 331"""

    # From book
    r_t = np.array([1622.39, 5305.10, 3717.44]) * KM_TO_M
    v_t = np.array([-7.29977, 0.492357, 2.48318]) * KM_TO_M
    r_c = np.array([1612.75, 5310.19, 3750.33]) * KM_TO_M
    v_c = np.array([-7.35321, 0.463856, 2.46920]) * KM_TO_M

    dcm_RSW_wrt_ECI = Trans.PCI_to_RSW(np.concat((r_t, v_t)))

    w_t_ECI = np.cross(r_t, v_t) / np.linalg.norm(r_t) ** 2

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
    v = np.zeros(3)

    ra, dec, rho, _, _, _ = Trans.cartesian_to_radec(r, v, deg=True)
    assert ra == pytest.approx(198.4, abs=0.1)
    assert dec == pytest.approx(33.12, abs=0.01)
    assert rho == pytest.approx(norm(r))


def test_radec_roundtrip():
    r0 = np.array([-536328, -17132484, 3234691]) # m but arbitrary
    v0 = np.array([-536, -17132, 3234]) # m/s but arbitrary

    for deg in (True, False):
        results = Trans.cartesian_to_radec(r0, v0, deg=deg)
        r, v = Trans.radec_to_cartesian(*results, deg=deg)
        assert r == pytest.approx(r0)
        assert v == pytest.approx(v0)

#########################################################################################################
#               azel to radec
#########################################################################################################


def test_azel_to_radec_zenith_and_north_horizon():
    """Zenith points at (LST, lat); north horizon points at dec = 90 - lat."""
    lat, lst = 39.007, 123.4
    ra, dec = Trans.azel_to_radec(0.0, 90.0, lat, lst)
    assert ra == pytest.approx(lst)
    assert dec == pytest.approx(lat)

    ra, dec = Trans.azel_to_radec(0.0, 0.0, lat, lst)
    assert dec == pytest.approx(90 - lat)
    assert ra == pytest.approx((lst - 180) % 360) # wrap to pi-ish


def test_azel_to_radec_matches_sez_rotation():
    """(1) Get cartesian with az/el (2) transform that into radec"""
    from cesium.math.rotations import dcm_y, dcm_z

    rng = np.random.default_rng(0)
    for az, el, lat, lst in rng.uniform([0, -80, -80, 0], [360, 80, 80, 360], (50, 4)):
        a_rad, e_rad = az * DEG_TO_RAD, el * DEG_TO_RAD
        u_sez = np.array([-np.cos(e_rad) * np.cos(a_rad), np.cos(e_rad) * np.sin(a_rad), np.sin(e_rad)])
        dcm_sez_wrt_ijk = dcm_y((90 - lat) * DEG_TO_RAD) @ dcm_z(lst * DEG_TO_RAD)
        ra_exp, dec_exp, *_ = Trans.cartesian_to_radec(
            dcm_sez_wrt_ijk.T @ u_sez, np.zeros(3)
        )

        ra, dec = Trans.azel_to_radec(az, el, lat, lst)
        assert dec == pytest.approx(dec_exp, abs=1e-9)
        assert (ra - ra_exp + 180) % 360 - 180 == pytest.approx(0, abs=1e-9)

#########################################################################################################
#               SO MANY OF THEM
#########################################################################################################


def test_vallado_example_4_1():
    """Vallado 4e. Example 4-1 p. 270. Just goin down and calculating"""
    rla = 294.9891458 * DEG_TO_RAD
    dla = -20.8234944 * DEG_TO_RAD
    rho_km = 4437725220.51  # km
    rla_dot = -0.00000012244 * DEG_TO_RAD
    dla_dot = -0.00000001794 * DEG_TO_RAD
    rho_km_dot = -25.53033094  # km

    r_eci_km_expected = np.array([
        1752246215, -3759563433, -1577568105
    ])  # km
    v_eci_km_expected = np.array([-18.323, 18.332, 7.777])

    # Radec
    r_eci_km, v_eci_km = Trans.radec_to_cartesian(rla, dla, rho_km, rla_dot, dla_dot, rho_km_dot, deg=False)
    assert r_eci_km == pytest.approx(r_eci_km_expected, abs = 10) # 10 km oop
    assert v_eci_km == pytest.approx(v_eci_km_expected, abs=1e-3)

    # Site vector
    lla = (39.007, -104.883, 2194.56)
    r_site_km = Trans.geodetic_LLA_to_ECEF(*lla) * M_TO_KM
    assert r_site_km == pytest.approx([-1275.123419, -4797.994704, 3994.302210])
    v_site_km = np.zeros(3)

    # Hopefully works from ECI to ITRF
    jd_utc = greg_to_jd(1994, 5, 14, 13, 11, 20.59856)
    r_site_eci_km, v_site_eci_km = Trans.itrf_to_gcrs(r_site_km, v_site_km, jd_utc)

    assert r_site_eci_km == pytest.approx([4068.213205, -2842.429051, 3996.349956], rel=1e-3)
    assert v_site_eci_km == pytest.approx([0.207272450, 0.296810996, 0.000116446], rel=1e-3)

    # SEZ
    r_ecef_km, v_ecef_km = Trans.gcrs_to_itrf(r_eci_km, v_eci_km, jd_utc)
    az, el, rho, az_dot, el_dot, rho_dot = Trans.geodetic_ECEF_to_SEZ(r_ecef_km * KM_TO_M, v_ecef_km * KM_TO_M, *lla, deg=True)

    assert az == pytest.approx(210.8250667)
    assert el == pytest.approx(23.8595052)
    assert rho * M_TO_KM == pytest.approx(4437722626.456)
    assert az_dot == pytest.approx(0.003862975427)
    assert el_dot == pytest.approx(-0.001663710935)
    assert rho_dot * M_TO_KM == pytest.approx(-25.3608138, rel=1e-5)


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
    dX = -0.000205 * ARCSEC_TO_RAD
    dY = -0.000136 * ARCSEC_TO_RAD

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

    dcm_gcrs_wrt_itrf = Trans.itrf_to_gcrs_matrix(jd, eop)

    # Tests that determinant is 1
    assert np.linalg.det(dcm_gcrs_wrt_itrf) == pytest.approx(1, 1e-9)

    # Transforms vector correctly
    r_gcrs = dcm_gcrs_wrt_itrf @ r_itrf
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


def test_vallado_full_gcrs_rv():
    """Vallado 4e Example 3-14 p. 220, transformed back to ITRF."""
    # Book values are given in km
    r_gcrs = np.array([5102.508959, 6123.011403, 6378.136925])
    v_gcrs = np.array([-4.743220161, 0.790536495, 5.533755724])
    r_expected = np.array([-1033.479383, 7901.2952754, 6380.3565958])
    v_expected = np.array([-3.225636520, -2.872451450, 5.531924446])
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

    r_itrf, v_itrf = Trans.gcrs_to_itrf(r_gcrs, v_gcrs, jd, eop)

    assert np.allclose(r_itrf, r_expected, rtol=1e-4)
    assert np.allclose(v_itrf, v_expected, rtol=1e-4)


def test_itrf_gcrs_round_trip():
    """Recover an ITRF state after forward and reverse transformations."""
    r_itrf = np.array([-1033.479383, 7901.2952754, 6380.3565958])
    v_itrf = np.array([-3.225636520, -2.872451450, 5.531924446])
    jd = greg_to_jd(2004, 4, 6, 7, 51, 28.386009)

    eop = Trans.EOP(
        xp=-0.140682 * ARCSEC_TO_RAD,
        yp=0.333309 * ARCSEC_TO_RAD,
        deltaUT1_s=-0.4399619,
        deltaAT_s=32,
        lod_s=0.0015563,
    )

    r_gcrs, v_gcrs = Trans.itrf_to_gcrs(r_itrf, v_itrf, jd, eop)
    r_recovered, v_recovered = Trans.gcrs_to_itrf(r_gcrs, v_gcrs, jd, eop)

    assert np.allclose(r_recovered, r_itrf, rtol=0, atol=1e-9)
    assert np.allclose(v_recovered, v_itrf, rtol=0, atol=1e-12)


#########################################################################################################
#               ICRF <--> Mars
#########################################################################################################


def test_mars_rotational_elements_pole_and_node():
    """Mars pole maps to MCI +z; MCI +x (node) lies on the ICRF equator."""
    jd = greg_to_jd(2035, 12, 25, 0, 0, 0)
    alpha_0, delta_0, _ = Trans.mars_rotational_elements(jd)
    dcm = Trans.MCI_from_ECI(jd)

    pole_eci, _ = Trans.radec_to_cartesian(alpha_0, delta_0, 1.0, 0, 0, 0, deg=False)
    assert np.allclose(dcm @ pole_eci, [0, 0, 1], atol=1e-12)
    assert dcm[0, 2] == pytest.approx(0.0, abs=1e-12)  # MCI x has no ECI z
    assert np.allclose(dcm @ dcm.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(dcm) == pytest.approx(1.0)


def test_MCI_from_ECI_j2000_pole_radec():
    """MCI z-axis expressed in ECI recovers the IAU 2009 J2000 pole."""
    pole_eci = Trans.MCI_from_ECI().T @ [0, 0, 1]
    ra, dec, *_ = Trans.cartesian_to_radec(pole_eci, np.zeros(3))
    assert ra == pytest.approx(317.68143)
    assert dec == pytest.approx(52.88650)


def test_mars_obliquity_from_MCI_from_HCI():
    """Mars pole vs. orbit normal gives Mars's obliquity, ~25.19 deg."""
    inc, raan = 1.85061 * DEG_TO_RAD, 49.57854 * DEG_TO_RAD  # J2000 ecliptic
    orbit_normal_hci = np.array(
        [np.sin(raan) * np.sin(inc), -np.cos(raan) * np.sin(inc), np.cos(inc)]
    )
    pole_hci = Trans.MCI_from_HCI().T @ [0, 0, 1]
    obliquity = np.arccos(pole_hci @ orbit_normal_hci) * RAD_TO_DEG
    assert obliquity == pytest.approx(25.19, abs=0.02)


def test_MCMF_from_MCI_sidereal_day():
    """Prime meridian returns to the same angle after one sidereal sol."""
    sidereal_day_days = 360 / 350.89198226  # 24.6229 hr
    jd = greg_to_jd(2035, 12, 25, 0, 0, 0)
    assert np.allclose(
        Trans.MCMF_from_MCI(jd), Trans.MCMF_from_MCI(jd + sidereal_day_days), atol=1e-9
    )
    assert np.allclose(
        Trans.MCMF_from_ECI(jd), Trans.MCMF_from_MCI(jd) @ Trans.MCI_from_ECI(jd)
    )
