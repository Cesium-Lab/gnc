import numpy as np
import pytest
from numpy.linalg import norm

from cesium.constants import DAY_TO_SEC
from cesium.math.integrators import rk4_func
from cesium.math.rotations import dcm_z
from cesium.physics.dynamics import (
    propagate_synodic,
    propagate_two_body,
    synodic_dynamics,
)
from cesium.physics.energy import calc_jacobi_constant
from cesium.world import MU_EARTH, R_EARTH
from cesium.world.constants import A_MOON, MU_MOON

R0 = R_EARTH + 500e3  # m
V_CIRC = np.sqrt(MU_EARTH / R0)
RV0 = np.array([R0, 0.0, 0.0, 0.0, V_CIRC, 0.0])
PERIOD_S = 2 * np.pi * np.sqrt(R0**3 / MU_EARTH)


def test_circular_orbit_returns_after_one_period():
    rv = propagate_two_body(RV0, np.linspace(0, PERIOD_S, 200), MU_EARTH)

    assert rv.shape == (200, 6)
    assert np.allclose(
        rv[-1], RV0, rtol=0, atol=[0.1] * 3 + [1e-4] * 3
    )  # Last point matches
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


################################################################################
#               Circular restricted three-body (Earth-Moon)
################################################################################

L_EM = A_MOON  # Earth-Moon separation [m]
R1_BC = L_EM * MU_MOON / (MU_EARTH + MU_MOON)  # Earth to barycenter [m]
R2_BC = L_EM * MU_EARTH / (MU_EARTH + MU_MOON)  # Moon to barycenter [m]
W_S = np.sqrt((MU_EARTH + MU_MOON) / L_EM**3)  # synodic rate [rad/s]
EM = {"mu1": MU_EARTH, "mu2": MU_MOON, "r1_bc": R1_BC, "r2_bc": R2_BC, "w_s": W_S}

# L4 leads the Moon: equilateral triangle with both primaries
RV_L4 = np.array([L_EM / 2 - R1_BC, np.sqrt(3) / 2 * L_EM, 0.0, 0.0, 0.0, 0.0])

# Generic out-of-plane state between the primaries (for pointwise checks)
RV_GENERIC = np.array([1.5e8, 1.2e8, 2.0e7, 150.0, -300.0, 40.0])

# Bound, inclined Earth orbit at ~100,000 km: never nears the Moon, so integration stays
# accurate. Inertial circular speed minus frame rotation, plus some out-of-plane velocity.
R_NEAR = 1.0e8
RV_NEAR_EARTH = np.array(
    [R_NEAR - R1_BC, 0.0, 0.0, 0.0, np.sqrt(MU_EARTH / R_NEAR) - W_S * R_NEAR, 500.0]
)


class Test_synodic_dynamics:
    def test_position_rates_are_velocity(self):
        rv_dot = synodic_dynamics(RV_GENERIC, **EM)
        assert rv_dot[0:3] == pytest.approx(RV_GENERIC[3:6])

    def test_l4_is_an_equilibrium(self):
        """Gravity of both primaries balances the centrifugal term at L4."""
        a = synodic_dynamics(RV_L4, **EM)[3:6]
        a_scale = MU_EARTH / L_EM**2  # Earth's pull at lunar distance
        assert norm(a) < 1e-12 * a_scale

    def test_mirror_symmetry_about_orbit_plane(self):
        """Reflecting z and vz flips only az."""
        flip = np.array([1, 1, -1, 1, 1, -1])
        assert synodic_dynamics(RV_GENERIC * flip, **EM) == pytest.approx(
            synodic_dynamics(RV_GENERIC, **EM) * flip
        )

    def test_no_secondary_matches_rotated_inertial_two_body(self):
        """With mu2 = 0, CR3BP motion is two-body motion viewed from a rotating frame.

        Propagate with `propagate_two_body` and rotate into the synodic frame
        (r_S = Rz(w t) r_I, v_S = Rz(w t) v_I - w x r_S),
        and compare with `propagate_synodic`. Frames coincide at t = 0.
        """
        w_s = np.sqrt(MU_EARTH / L_EM**3)
        sys = (MU_EARTH, 0.0, 0.0, L_EM, w_s)  # Earth at the origin, massless "Moon"
        w = np.array([0.0, 0.0, w_s])
        t_s = np.linspace(0.0, 5 * DAY_TO_SEC, 6)

        rv0_S = RV_GENERIC
        rv0_I = np.concatenate((rv0_S[0:3], rv0_S[3:6] + np.cross(w, rv0_S[0:3])))
        rv_I = propagate_two_body(rv0_I, t_s, MU_EARTH, rtol=1e-12, atol=1e-6)
        rv_S = propagate_synodic(rv0_S, t_s, *sys)

        for t, rvi, rvs in zip(t_s, rv_I, rv_S, strict=True):
            r_S = dcm_z(w_s * t) @ rvi[0:3]
            v_S = dcm_z(w_s * t) @ rvi[3:6] - np.cross(w, r_S)
            assert rvs[0:3] == pytest.approx(r_S, abs=1.0)  # m, over ~2e8 m
            assert rvs[3:6] == pytest.approx(
                v_S, abs=1e-4
            )  # m/s, ~1e-8 relative at 2.5 km/s

    @pytest.mark.parametrize(
        "overrides",
        [
            {"mu1": MU_MOON, "mu2": MU_EARTH},  # primaries swapped
            {"r1_bc": R2_BC, "r2_bc": R1_BC},  # distances swapped
            {"r1_bc": 2 * R1_BC},  # barycenter not balanced
            {"w_s": 1.01 * W_S},  # rate inconsistent with Kepler's third law
        ],
    )
    def test_rejects_inconsistent_system(self, overrides):
        with pytest.raises(AssertionError):
            synodic_dynamics(RV_GENERIC, **(EM | overrides))


class Test_propagate_synodic:
    T_20D = np.linspace(0.0, 20 * DAY_TO_SEC, 200)

    def test_output_grid_and_initial_state(self):
        rv = propagate_synodic(RV_NEAR_EARTH, self.T_20D, **EM)
        assert rv.shape == (200, 6)
        assert rv[0] == pytest.approx(RV_NEAR_EARTH)

    def test_jacobi_constant_is_conserved(self):
        rv = propagate_synodic(RV_NEAR_EARTH, self.T_20D, **EM)
        c = np.array([calc_jacobi_constant(s, **EM) for s in rv])
        assert c == pytest.approx(c[0], rel=1e-9)

    def test_forward_then_backward_round_trip(self):
        t_s = np.linspace(0.0, 5 * DAY_TO_SEC, 50)
        rv_fwd = propagate_synodic(RV_NEAR_EARTH, t_s, **EM)
        rv_back = propagate_synodic(rv_fwd[-1], t_s[::-1], **EM)
        assert np.allclose(rv_back[::-1], rv_fwd, rtol=0, atol=[1.0] * 3 + [1e-5] * 3)

    def test_matches_fixed_step_rk4(self):
        """Independent integrator check: 60 s RK4 steps over one day."""
        dt, n_steps = 60.0, 1440
        rv = RV_NEAR_EARTH.copy()
        for k in range(n_steps):
            rv = rk4_func(k * dt, dt, rv, lambda _, s: synodic_dynamics(s, **EM))

        rv_ivp = propagate_synodic(RV_NEAR_EARTH, [0.0, n_steps * dt], **EM)[-1]
        assert np.allclose(rv, rv_ivp, rtol=0, atol=[1.0] * 3 + [1e-5] * 3)

    def test_spacecraft_at_l4_stays_there(self):
        rv = propagate_synodic(RV_L4, [0.0, 30 * DAY_TO_SEC], **EM)
        assert norm(rv[-1, 0:3] - RV_L4[0:3]) < 1e3  # m, over a month
        assert norm(rv[-1, 3:6]) < 1e-3  # m/s

    def test_no_secondary_circular_orbit_is_stationary(self):
        """With mu2 = 0, a circular orbit at rate w_s sits still in the rotating frame."""
        w_s = np.sqrt(MU_EARTH / L_EM**3)
        rv0 = np.array(
            [0.0, L_EM, 0.0, 0.0, 0.0, 0.0]
        )  # 90 deg from the (massless) Moon
        rv = propagate_synodic(
            rv0, [0.0, 27 * DAY_TO_SEC], MU_EARTH, 0.0, 0.0, L_EM, w_s
        )
        assert norm(rv[-1, 0:3] - rv0[0:3]) < 1.0  # m, after ~one sidereal month
        assert norm(rv[-1, 0:6] - rv0[0:6]) < 0.001  # m, after ~one sidereal month

    def test_raises_when_integration_fails(self):
        """Radial free-fall onto Earth (at rest relative to it) hits the r1 = 0 singularity."""
        d = 1e7  # m from Earth's center
        rv0 = np.array([-R1_BC + d, 0.0, 0.0, 0.0, -W_S * d, 0.0])
        with pytest.raises(RuntimeError):
            propagate_synodic(rv0, [0.0, DAY_TO_SEC], **EM)
