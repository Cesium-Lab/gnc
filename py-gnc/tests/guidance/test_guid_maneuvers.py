import numpy as np
import pytest

from cesium.guidance import Burn


def test_burn_magnitude_and_defaults():
    burn = Burn(t_s=12.5, r=np.array([7e6, 0.0, 0.0]), dv=np.array([3.0, 4.0, 0.0]))

    assert burn.dv_mag == pytest.approx(5.0)
    assert burn.frame == "PCI"


def test_burn_is_immutable():
    burn = Burn(t_s=0.0, r=np.zeros(3), dv=np.zeros(3), frame="RSW")
    with pytest.raises(AttributeError):
        burn.t_s = 1.0
