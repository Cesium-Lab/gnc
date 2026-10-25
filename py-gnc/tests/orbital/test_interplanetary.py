import pytest

from cesium.constants import KM_TO_M
from cesium.orbital.interplanetary import sph_of_infl
from cesium.world.constants import A_MOON, MU_EARTH, MU_MOON


def test_SOI():
    """Vallado 4e. p. 360"""
    soi = sph_of_infl(A_MOON, MU_MOON, MU_EARTH)
    assert soi / KM_TO_M == pytest.approx(66200, rel=1e-3)
