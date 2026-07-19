# ruff: noqa: E741
"""End-to-end RPO demo: two vehicles propagated with full nonlinear (two-body) truth
dynamics, a relative-GPS sensor generating noisy measurements, an EKF estimating the
relative state (using the linear CW model as its process model between measurements),
and HCW guidance planning a two-impulse rendezvous from that estimate.

This is the scenario described in the RPO vehicle/sensor modeling plan -- see
`guidance/hcw_targeting.py` and `csim/physics/relative_motion.py`.
"""
import sys
sys.path.append(".")
sys.path.append("..")

import numpy as np
import pytest

from csim.math.integrators import rk4_func
from csim.physics.gravity import grav_accel
from csim.physics.relative_motion import mean_motion, cw_stm, eci_relative_to_hill
from csim.world import MU_EARTH
from csim.entities import Spacecraft
from csim.entities.components.sensors import RelativeGPS

from estimation.filtering.ekf import EKF, EkfParams, State
from guidance.hcw_targeting import two_impulse_rendezvous


def two_body_eom(t, state, params=None):
    """Truth dynamics: full nonlinear two-body (position/velocity only)"""
    r, v = state[:3], state[3:6]
    return np.hstack((v, grav_accel(r)))


@pytest.mark.slow
def test_rpo_scenario_ekf_converges_and_guidance_targets_origin():
    rng = np.random.default_rng(42)

    r0 = 6900e3
    v0 = np.sqrt(MU_EARTH / r0)
    n = mean_motion(r0, MU_EARTH)
    period = 2 * np.pi / n

    r_t0 = np.array([r0, 0, 0])
    v_t0 = np.array([0, v0, 0])

    # Chaser starts 1km behind on V-bar, on a matched circular orbit
    rho0_truth = np.array([0.0, -1000.0, 0.0, 0.0, 0.0, 0.0])
    dtheta = rho0_truth[1] / r0
    r_c0 = r0 * np.array([np.cos(dtheta), np.sin(dtheta), 0])
    v_c0 = v0 * np.array([-np.sin(dtheta), np.cos(dtheta), 0])

    gps = RelativeGPS(pos_noise_std=0.05, vel_noise_std=0.002, rng=rng)
    chaser = Spacecraft(mass=500, I=np.eye(3), sensors=[gps])

    dt = 5.0
    n_steps = int(period / dt)
    Phi_dt = cw_stm(dt, n)

    def f(x, u):
        return Phi_dt @ x

    def F(x, u):
        return Phi_dt

    Q = np.diag([1e-4]*3 + [1e-7]*3)  # covers CW-vs-nonlinear-truth model mismatch
    x0_guess = rho0_truth + rng.normal(scale=[50, 50, 50, 0.05, 0.05, 0.05])
    ekf = EKF(
        State(x=x0_guess, P=np.diag([100.]*3 + [0.1]*3)),
        EkfParams(f=f, F=F, h=gps.h, H=gps.H, Q=Q, R=gps.R),
    )

    state_t, state_c = np.hstack((r_t0, v_t0)), np.hstack((r_c0, v_c0))

    for _ in range(n_steps):
        state_t = rk4_func(0, dt, state_t, two_body_eom)
        state_c = rk4_func(0, dt, state_c, two_body_eom)

        measurements = chaser.sense(state_c, state_t)
        z, valid = measurements[gps]

        ekf.predict(u=np.zeros(0))
        if valid:
            ekf.update(z)

    rho_truth_final = eci_relative_to_hill(state_t[:3], state_t[3:6], state_c[:3], state_c[3:6])
    rho_est_final = ekf.state.x

    pos_err = np.linalg.norm(rho_est_final[:3] - rho_truth_final[:3])
    vel_err = np.linalg.norm(rho_est_final[3:] - rho_truth_final[3:])

    # After a full orbit of direct GPS updates the filter should track truth tightly --
    # generous tolerances (vs. the noise floor) to stay robust to rng variation.
    assert pos_err < 2.0
    assert vel_err < 0.01

    # Guidance: plan a rendezvous to the origin from the EKF's final estimate
    t_go = 600.0
    Phi_go = cw_stm(t_go, n)
    dv1, dv2 = two_impulse_rendezvous(rho_est_final, Phi_go, rho_f=np.zeros(3))

    assert np.all(np.isfinite(dv1)) and np.all(np.isfinite(dv2))

    rho_post_burn = rho_est_final.copy()
    rho_post_burn[3:] += dv1
    rho_arrival = Phi_go @ rho_post_burn

    # The planned maneuver must be internally consistent with its own (linear) model:
    # propagating the post-burn state with the same STM should land exactly at rho_f=0.
    assert np.allclose(rho_arrival[:3], np.zeros(3), atol=1e-6)
