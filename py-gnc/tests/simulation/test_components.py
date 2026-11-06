import pytest

from cesium.world.constants import G0
from csim.entities import Thruster


def test_thruster_force():
    thruster = Thruster(I_sp=300, thrust_max=1000, thrust_min=100)

    # Test force at different throttle levels
    assert thruster.force(-1) == 100
    assert thruster.force(0) == 100
    assert thruster.force(0.5) == 550  # halfway between min and max
    assert thruster.force(1) == 1000
    assert thruster.force(1.5) == 1000

    # Test m_dot at different throttle levels
    assert thruster.m_dot(-1) == 100 / (300 * G0)
    assert thruster.m_dot(0) == 100 / (300 * G0)
    assert thruster.m_dot(0.5) == 550 / (300 * G0)
    assert thruster.m_dot(1) == 1000 / (300 * G0)
    assert thruster.m_dot(1.5) == 1000 / (300 * G0)


def test_thruster_invalid_params():
    with pytest.raises(ValueError):
        Thruster(I_sp=-1, thrust_max=1000)  # I_sp negative
    with pytest.raises(ValueError):
        Thruster(I_sp=0, thrust_max=1000)  # I_sp 0
    with pytest.raises(ValueError):
        Thruster(I_sp=300, thrust_max=-1000)  # thrust negative
    with pytest.raises(ValueError):
        Thruster(
            I_sp=300, thrust_max=1000, thrust_min=2000
        )  # thrust min greater than thrust max
