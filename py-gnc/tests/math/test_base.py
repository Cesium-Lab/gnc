import numpy as np
import pytest

from cesium.math.base import center_of_curvature, unit
from cesium.math.rotations import wrap_to_pi


def test_unit():
    """Sanity?"""

    # Super small
    v = np.zeros(100)
    v[0] = 1e-9
    assert unit(v) == pytest.approx(np.zeros(100))

    v = np.zeros(1000)
    v[0] = 1
    v[-1] = 1
    assert unit(v)[0] == 1 / np.sqrt(2)
    assert unit(v)[-1] == 1 / np.sqrt(2)


def test_center_of_curvature():
    """Curtis, Example 1.2 p.6"""

    r = np.array([250, 630, 430])
    v = np.array([90, 125, 170])
    a = np.array([16, 125, 30])

    # 3 sig figs past
    assert center_of_curvature(r, v, a) == pytest.approx([95.16, 1141, 136.0], rel=1e-3)


def test_wrap_to_pi():
    assert wrap_to_pi(np.pi) == -np.pi  # at bound
    eps = 1e-9
    assert wrap_to_pi(np.pi - eps) == np.pi - eps
    assert wrap_to_pi(eps) == pytest.approx(eps)

    assert wrap_to_pi(0) == 0
    assert wrap_to_pi(2 * np.pi) == 0  # at bound
