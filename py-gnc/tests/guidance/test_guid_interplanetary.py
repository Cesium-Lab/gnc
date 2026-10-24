import pytest

import cesium.guidance as guid
from cesium.constants import KM_TO_M


def test_279b():
    v_inf_arr = 2.61 * KM_TO_M
    r_p = 3997 * KM_TO_M
    mu_mars = 43050 * (KM_TO_M**3)  # for this

    dv_p, ta_deg, r_aim, e = guid.interplanetary_arrival(v_inf_arr, r_p, mu_mars)

    assert dv_p == pytest.approx(2.04 * KM_TO_M, abs=10)
    assert ta_deg == pytest.approx(127.8, abs=0.1)
    assert r_aim / KM_TO_M == pytest.approx(8155, abs=1)
    assert e == pytest.approx(1.6325, abs=0.0001)
