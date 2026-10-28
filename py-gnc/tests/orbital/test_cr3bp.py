import numpy as np
import pytest

from cesium.constants import DEG_TO_RAD, KM_TO_M, MIN_TO_SEC, RAD_TO_DEG
from cesium.orbital.cr3bp import lagrange_points


def test_279b_hw1():
    mu_pluto = 870.9 * KM_TO_M**3
    mu_charon = 101.0 * KM_TO_M**3
    r_pluto_charon = 19571.0 * KM_TO_M
    
    # Using Vallado
    L = lagrange_points(mu_pluto, mu_charon, r_pluto_charon)
    
    # Using method from class
    assert L[0] / KM_TO_M == pytest.approx([ 11774.278743037392, 0, 0])
    assert L[1] / KM_TO_M == pytest.approx([ 24680.215875872727, 0, 0])
    assert L[2] / KM_TO_M == pytest.approx([ -20417.15380694039, 0, 0])
    
    assert L[3] / KM_TO_M == pytest.approx([ 7751.678619199506, 16948.98317746525, 0])
    assert L[4] / KM_TO_M == pytest.approx([ 7751.678619199506, -16948.98317746525, 0])

