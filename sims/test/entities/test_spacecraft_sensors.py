# ruff: noqa: E741
import sys
sys.path.append(".")

import numpy as np
import pytest

from csim.entities import Spacecraft
from csim.entities.components.sensors import RelativeGPS, RFRanging, AnglesOnly, Lidar
from csim.physics.relative_motion import eci_relative_to_hill
from csim.world import MU_EARTH

RHO_TEST = np.array([120.0, -300.0, 40.0, 0.3, -0.2, 0.05])


def _finite_diff_H(sensor, rho, eps=1e-4):
    H_fd = np.zeros((len(sensor.h(rho)), 6))
    for i in range(6):
        d = np.zeros(6)
        d[i] = eps
        H_fd[:, i] = (sensor.h(rho + d) - sensor.h(rho - d)) / (2 * eps)
    return H_fd


@pytest.mark.parametrize("sensor", [
    RelativeGPS(),
    RFRanging(),
    AnglesOnly(),
    Lidar(),
])
def test_sensor_jacobian_matches_finite_difference(sensor):
    H_analytic = sensor.H(RHO_TEST)
    H_fd = _finite_diff_H(sensor, RHO_TEST)
    assert np.allclose(H_analytic, H_fd, atol=1e-6)


def test_sensor_measure_returns_noisy_z_near_truth():
    rng = np.random.default_rng(0)
    gps = RelativeGPS(pos_noise_std=0.02, vel_noise_std=0.001, rng=rng)

    z, valid = gps.measure(RHO_TEST)

    assert valid
    assert np.allclose(z, RHO_TEST, atol=0.5)  # well within a handful of sigma


def test_ranging_out_of_max_range_is_invalid():
    far_sensor = RFRanging(max_range_m=100.0)
    z, valid = far_sensor.measure(np.array([500.0, 0, 0, 0, 0, 0]))

    assert not valid
    assert z is None


def test_angles_only_outside_fov_is_invalid():
    narrow_fov = AnglesOnly(fov_half_angle_rad=np.deg2rad(10), boresight=np.array([-1.0, 0, 0]))
    # Target roughly 90 degrees off boresight
    z, valid = narrow_fov.measure(np.array([0.0, 500.0, 0, 0, 0, 0]))

    assert not valid
    assert z is None


def test_spacecraft_sense_matches_relative_motion_conversion():
    r0 = 6900e3
    v0 = np.sqrt(MU_EARTH / r0)

    target_state = np.hstack(([r0, 0, 0], [0, v0, 0]))
    # Chaser offset 100m along-track with matched circular velocity
    dtheta = 100.0 / r0
    r_c = r0 * np.array([np.cos(dtheta), np.sin(dtheta), 0])
    v_c = v0 * np.array([-np.sin(dtheta), np.cos(dtheta), 0])
    chaser_state = np.hstack((r_c, v_c))

    rng = np.random.default_rng(1)
    gps = RelativeGPS(pos_noise_std=1e-9, vel_noise_std=1e-12, rng=rng)
    chaser = Spacecraft(mass=500, I=np.eye(3), sensors=[gps])

    measurements = chaser.sense(chaser_state, target_state)
    z, valid = measurements[gps]

    expected_rho = eci_relative_to_hill(
        target_state[:3], target_state[3:6], chaser_state[:3], chaser_state[3:6])

    assert valid
    assert np.allclose(z, expected_rho, atol=1e-4)


def test_spacecraft_default_sensors_is_empty_list():
    sc = Spacecraft(mass=100, I=np.eye(3))
    assert sc.sensors == []
