import numpy as np
import pytest
from numpy.linalg import norm

from cesium.physics.dynamics import propagate_two_body
from cesium.world import MU_EARTH, R_EARTH

R0 = R_EARTH + 500e3  # m
V_CIRC = np.sqrt(MU_EARTH / R0)
RV0 = np.array([R0, 0.0, 0.0, 0.0, V_CIRC, 0.0])
PERIOD_S = 2 * np.pi * np.sqrt(R0**3 / MU_EARTH)


def test_circular_orbit_returns_after_one_period():
    rv = propagate_two_body(RV0, np.linspace(0, PERIOD_S, 200), MU_EARTH)

    assert rv.shape == (200, 6)
    assert np.allclose(rv[-1], RV0, rtol=0, atol=[0.1] * 3 + [1e-4] * 3)  # Last point matches
    assert np.allclose(norm(rv[:, :3], axis=1), R0, rtol=1e-8)  # circle condition


def test_forward_then_backward_round_trip():
    t_s = np.linspace(0, 0.37 * PERIOD_S, 50)
    rv_forward = propagate_two_body(RV0, t_s, MU_EARTH)

    rv_back = propagate_two_body(rv_forward[-1], t_s[::-1], MU_EARTH)

    assert np.allclose(rv_back[::-1], rv_forward, rtol=0, atol=[0.1] * 3 + [1e-4] * 3)


def test_backward_velocities_are_physical():
    """One minute before t=0 the spacecraft was at -y, still moving forward along +y."""
    dt_s = 60.0
    rv = propagate_two_body(RV0, [0.0, -dt_s], MU_EARTH)

    angle = V_CIRC / R0 * dt_s  # orbit angle covered in dt_s
    expected = np.array(
        [
            R0 * np.cos(angle),
            -R0 * np.sin(angle),
            0.0,
            V_CIRC * np.sin(angle),
            V_CIRC * np.cos(angle),
            0.0,
        ]
    )
    assert np.allclose(rv[-1], expected, rtol=0, atol=[0.1] * 3 + [1e-4] * 3)


def test_raises_when_integration_fails():
    """Radial free-fall from rest hits the r = 0 singularity, so the integrator fails."""
    rv0 = np.array([R0, 0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(RuntimeError):
        propagate_two_body(rv0, [0.0, 2 * PERIOD_S], MU_EARTH)
