import sys
import numpy as np
import pytest


from cesium.orbital.hcw import rv_hcw_3d

from cesium.world import MU_EARTH, R_EARTH
from cesium.constants import KM_TO_M, MIN_TO_SEC


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
    
    # M = 235.4 * DEG_TO_RAD
    # e = 0.4
    # E = kepler_eq(M, e)

    # assert E * RAD_TO_DEG == pytest.approx(220.512074767522)
