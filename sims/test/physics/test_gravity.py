import sys

sys.path.append(".")

import numpy as np

# from csim.physics.energy import calc_total_energy
# from csim.physics.rigid_body import rigid_body_derivative, RigidBodyParams
# from csim.math.integrators import rk4_func
from csim.world import R_EARTH

from csim.physics import grav_accel


def test_normal_grav_edge_cases():
    """Just to make sure the sim does not break with singularities"""
    # r = 0
    assert np.array_equal(grav_accel(np.zeros(3)), [0, 0, 0])

    # r = inf
    assert np.allclose(grav_accel(np.ones(3) * 1e12), [0, 0, 0])
    assert np.allclose(grav_accel(np.ones(3) * np.inf), [0, 0, 0])

    # mu = 0
    assert np.array_equal(grav_accel(np.array([6800e3, 0, 0]), 0), [0, 0, 0])


def test_normal_grav():
    assert np.allclose(
        grav_accel(np.array([R_EARTH, 0, 0])), [-9.80665, 0, 0], rtol=1e-4
    )
