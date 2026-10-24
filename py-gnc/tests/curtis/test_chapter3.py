import numpy as np
import pytest


from cesium.orbital.coes import kepler_eq, ta_from_E
from cesium.constants import DEG_TO_RAD, KM_TO_M, HOUR_TO_SEC
from cesium.world import MU_EARTH

def test_curtis_example_3_2():
    """Curtis Example 3.2 p. 116"""
    rp = 9600 * KM_TO_M
    ra = 21000 * KM_TO_M
    e = (ra-rp)/(ra+rp)
    
    a = rp/(1-e)
    T = 2*np.pi*np.sqrt(a**3 / MU_EARTH)
    
    t_des = 3*HOUR_TO_SEC
    M_des = 2*np.pi*t_des/T
    
    E_des = kepler_eq(M_des, e)
    assert E_des == pytest.approx(3.4794, abs=0.0001)
    
    ta = ta_from_E(E_des, e)
    assert ta / DEG_TO_RAD == pytest.approx(193.2, abs=0.1)
    
# TODO: parabolic and hyperbolic trajectories