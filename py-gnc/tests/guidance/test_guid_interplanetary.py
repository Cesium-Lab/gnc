import pytest

import cesium.guidance as guid
from cesium.constants import KM_TO_M
from cesium.guidance.types import FlybyStichScalarInput, FlybyStichScalarOutput

###########################################################################
#                   interplanetary_circ_injection()
###########################################################################

def test_279b():
    v_inf_arr = 2.61 * KM_TO_M
    r_p = 3997 * KM_TO_M
    mu_mars = 43050 * (KM_TO_M**3)  # for this

    dv_p, ta_deg, b, e = guid.interplanetary_circ_injection(v_inf_arr, r_p, mu_mars)

    assert dv_p / KM_TO_M== pytest.approx(2.04, abs=0.01)
    assert ta_deg == pytest.approx(127.8, abs=0.1)
    assert b / KM_TO_M == pytest.approx(8155, abs=1)
    assert e == pytest.approx(1.6325, abs=0.0001)
    
def test_curtis_mars_example():
    """Curtis Example 8.5 p. 373.
    Adopting a Hohmann transfer"""
    
    v_inf = 2.648 * KM_TO_M 
    rp = 5447 * KM_TO_M
    mu = 42830 * (KM_TO_M**3)  # for this from Curtis
    
    dv_p, ta_deg, b, e = guid.interplanetary_circ_injection(v_inf, rp, mu, deg=True)
    
    e_exp = 1 + rp*v_inf**2/mu
    
    assert dv_p / KM_TO_M == pytest.approx(1.470, abs=1)
    assert (180 - ta_deg) == pytest.approx(58.09, abs=0.01) # true anomaly + turning angle = 180
    assert b / KM_TO_M == pytest.approx(9809, abs=1)
    assert e == pytest.approx(e_exp, abs=0.0001)
    
def test_infinite_vinf():
    """Curtis Example 8.5 p. 373.
    Adopting a Hohmann transfer"""
    
    v_inf = 1e20
    rp = 5447 * KM_TO_M
    mu = 42830 * (KM_TO_M**3)  # for this from Curtis
    
    dv_p, ta_deg, b, e = guid.interplanetary_circ_injection(v_inf, rp, mu, deg=True)
    
    e_exp = 1 + rp*v_inf**2/mu
    
    assert dv_p == pytest.approx(v_inf, rel=1e-18)
    assert ta_deg == pytest.approx(90, abs=0.01) # Basically tangent
    assert b == pytest.approx(rp, abs=1)
    assert e == pytest.approx(e_exp, abs=0.0001)

###########################################################################
#                   interplanetary_circ_injection()
###########################################################################

def test_flyby_0_angle():
    # Arbitrary
    v_inf = 3.648 * KM_TO_M 
    mu = 42830 * (KM_TO_M**3)  # for this from Curtis
    
    inputs = FlybyStichScalarInput(v_inf, v_inf, 0, mu)
    outputs = guid.flyby_stich(inputs, eps=2e-4, iter=100)
    
    assert outputs.dv == 0.0
    assert outputs.delta1 == pytest.approx(0, abs=1e-4)
    assert outputs.delta2 == pytest.approx(0, abs=1e-4)
    
    assert outputs.v_p1 == pytest.approx(v_inf, abs=1)
    assert outputs.v_p2 == pytest.approx(v_inf, abs=1)
    
    # Something crazy
    assert outputs.r_p > 4e10
    assert outputs.b1 == pytest.approx(outputs.r_p, rel=1e-3)
    assert outputs.b2 == pytest.approx(outputs.r_p, rel=1e-3)
    
    # Straight line
    assert outputs.e1 > 1e4
    assert outputs.e2 > 1e4
    
    assert outputs.delta1 == pytest.approx(0.0, abs=1e-4)
    assert outputs.delta2 == pytest.approx(0.0, abs=1e-4)
    