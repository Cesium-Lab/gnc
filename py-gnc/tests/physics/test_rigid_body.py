import numpy as np
import pytest

from cesium.math.integrators import rk4_func
from cesium.physics.energy import calc_total_energy
from cesium.physics.rigid_body import rigid_body_derivative
from cesium.world import MU_EARTH


def propagate(dt: float, tmax: float, state0: np.ndarray, params: dict):
    state = state0
    t = 0
    while t < tmax:
        state = rk4_func(
            t,
            dt,
            state,
            lambda time, stage_state: rigid_body_derivative(
                time,
                stage_state,
                params["mass"],
                params["force_I"],
                params.get("I"),
                params.get("torque_B"),
            ),
        )
        state[6:10] /= np.linalg.norm(state[6:10])
        t += dt
    return state


class TestRigidBodyDeriv:
    def test_force_no_torque(self):
        params = {
            "mass": 2,
            "I": np.diag([2, 2, 2]),
            "force_I": [1, 0, 0],
            "torque_B": [0, 0, 0],
        }

        state = [
            0,
            0,
            0,  # r = 0
            1.1,
            1.2,
            1.3,  # v != 0
            1,
            0,
            0,
            0,  # Identity quaternion
            0,
            0,
            0,  # w = 0
        ]

        expected = [
            1.1,
            1.2,
            1.3,
            0.5,
            0,
            0,  # Some acceleration
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]

        dot = rigid_body_derivative(0, state, **params)

        assert np.array_equal(dot, expected)

    def test_torque_no_force(self):
        params = {
            "mass": 2,
            "I": np.diag([2, 2, 2]),
            "force_I": [0, 0, 0],
            "torque_B": [1.0, 2.0, 4.0],
        }

        state = [
            0,
            0,
            0,  # r = 0
            1.1,
            1.2,
            1.3,  # v != 0
            1,
            0,
            0,
            0,  # Identity quaternion
            0,
            0,
            0,  # w = 0
        ]

        expected = [
            1.1,
            1.2,
            1.3,
            0,
            0,
            0,  # Some acceleration
            0,
            0,
            0,
            0,
            0.5,
            1,
            2,
        ]

        dot = rigid_body_derivative(0, state, **params)
        assert np.array_equal(dot, expected)

    def test_random_nonsense(self):
        params = {
            "mass": 17,
            "I": np.array([[15, 0.1, 0.5], [0.1, 15, 3], [0.5, 3, 4]]),
            "force_I": [12.0, 200.0, 14.0],
            "torque_B": [0.4, 0.2, 0.7],
        }

        state = np.array(
            [
                0.0,
                0.0,
                0.0,  # r = 0
                1.1,
                1.2,
                1.3,  # v = 0
                0.5,
                0.5,
                0.5,
                0.5,
                2.0,
                3.14,
                3.52,  # ω = 0
            ]
        )

        dot = rigid_body_derivative(0, state, **params)

        # Generated using Julia as a simple calculator with the textbook functions

        # Simple acceleration
        assert np.array_equal(dot[0:3], [1.1, 1.2, 1.3])

        # F/m
        assert np.allclose(
            dot[3:6],
            [0.7058823529411765, 11.764705882352942, 0.8235294117647058],
            atol=1e-12,
        )

        # .5 * ham_prod(q,w)
        assert np.allclose(
            dot[6:10], [-2.165, 0.595, 0.40500000000000014, 1.165], atol=1e-12
        )

        # Euler: w_dot = I^-1 (tau - w x I w), by hand with the adjugate (no numpy):
        #   I w           = [32.074, 57.86, 24.5]
        #   w x I w       = [-126.7372, 63.90048, 15.00764]
        #   tau - w x I w = [127.1372, -63.70048, -14.30764]
        #   det(I) = 761.51,  adj(I) = [[51, 1.1, -7.2], [1.1, 59.75, -44.95],
        #                               [-7.2, -44.95, 224.99]]
        #   w_dot = adj @ rhs / det = [8.557920, -3.969908, -1.669219]
        assert dot[10:13] == pytest.approx([8.557920, -3.969908, -1.669219], abs=1e-6)


class TestRigidBodyRk4Integration:
    def test_0_time(self):
        dt = 0
        t = 1.0

        I = np.diag((1, 1, 1))

        # Initialized with random but nonzero params
        params = {
            "mass": 17.0,  # kg
            "I": I,
            "force_I": [12.0, 200.0, 14.0],  # force
            "torque_B": [0.4, 0.2, 0.7],  # torque
        }

        state = np.array(
            [
                5,
                6,
                7,  # r = 0
                1.1,
                1.2,
                1.3,  # v = 0
                0.5,
                0.5,
                0.5,
                0.5,
                2.0,
                3.14,
                3.52,  # ω = 0
            ]
        )

        next_step = rk4_func(
            t,
            dt,
            state,
            lambda time, stage_state: rigid_body_derivative(
                time,
                stage_state,
                params["mass"],
                params["force_I"],
                params.get("I"),
                params.get("torque_B"),
            ),
        )

        assert np.array_equal(next_step, state)

    def test_force(self):
        dt = 0.001
        t = 1

        params = {
            "mass": 17.0,  # kg
            "I": np.diag((1, 1, 1)),
            "force_I": [1.0, 10.0, 100.0],  # force for acceleration
            "torque_B": [0.0, 0.0, 0.0],  # torque
        }

        state = np.array(
            [
                0.1,
                100.4,
                5,  # r
                1.1,
                1.2,
                1.3,  # v
                1,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )

        next_step = propagate(dt, t, state, params)

        assert np.allclose(
            next_step[0:3],
            [1.2294117647058824, 101.89411764705883, 9.241176470588236],
            rtol=1e-6,
        )
        assert np.allclose(
            next_step[3:6],
            [1.1588235294117648, 1.788235294117647, 7.182352941176471],
            rtol=1e-6,
        )
        assert np.array_equal(next_step[6:10], [1, 0, 0, 0])
        assert np.array_equal(next_step[10:14], [0, 0, 0])

    @pytest.mark.slow
    def test_pi_rotation(self):
        dt = 0.001
        t = 3.14159

        params = {
            "mass": 17.0,  # kg
            "I": np.diag((1, 1, 1)),
            "force_I": [0, 0, 0],  # force for acceleration
            "torque_B": [0, 0, 0],  # torque
        }

        state = np.array(
            [
                0,
                0,
                0,  # r
                0,
                0,
                0,  # v
                1,
                0,
                0,
                0,
                1,
                0,
                0,  # 1 rad/s
            ]
        )

        next_step = propagate(dt, t, state, params)

        assert np.array_equal(next_step[0:3], [0, 0, 0])  # no positions
        assert np.array_equal(next_step[3:6], [0, 0, 0])  # no positions
        assert np.allclose(
            next_step[6:10], [0, 1, 0, 0], atol=1e-3
        )  # 180 around X axis
        assert np.array_equal(next_step[10:14], [1, 0, 0])  # No torque to change it


# Would test torque but it is nonlinear so don't really want to so just gotta trust textbooks


class TestRigidBodyEnergy:
    @pytest.mark.slow
    def test_stationary_with_gravity(self):
        dt = 0.01
        t = 60
        mass = 1
        I = np.diag((2, 1, 4))
        # Initialized with random but nonzero params
        params = {
            "mass": mass,  # kg
            "I": I,
            "force_I": [0, 0, 0],  # force for acceleration
            "torque_B": [0, 0, 0],  # torque
        }

        r = [6800e3, 0, 0]
        v = [0, 0, 0]
        w = np.array([1, 0, 0])

        state = np.array(
            [
                *r,  # r
                *v,  # v
                1,
                0,
                0,
                0,
                *w,
            ]
        )

        assert calc_total_energy(mass, I, r, v, w, MU_EARTH) == pytest.approx(
            -58617712.0
        )
        next_step = propagate(dt, t, state, params)

        assert calc_total_energy(
            mass, I, next_step[0:3], next_step[3:6], next_step[10:14], MU_EARTH
        ) == pytest.approx(-58617712, rel=0.0001)

    @pytest.mark.slow
    def test_just_spinning(self):
        dt = 0.01
        t = 60
        mass = 1
        I = np.diag((2, 1, 4)) * 100

        params = {
            "mass": mass,  # kg
            "I": I,
            "force_I": [0, 0, 0],  # force for acceleration
            "torque_B": [0, 0, 0],  # torque
        }

        r = [0, 0, 0]
        v = [0, 0, 0]
        w = np.array([4, 0, 0])

        state = np.array(
            [
                *r,  # r
                *v,  # v
                1,
                0,
                0,
                0,
                *w,
            ]
        )

        assert calc_total_energy(mass, I, r, v, w, MU_EARTH) == 1600
        next_step = propagate(dt, t, state, params)

        assert (
            calc_total_energy(
                mass, I, next_step[0:3], next_step[3:6], next_step[10:14], MU_EARTH
            )
            == 1600
        )

    @pytest.mark.slow
    def test_just_accel(self):
        dt = 0.001
        t = 20
        mass = 0.001
        I = np.diag((2, 1, 4)) * 100

        params = {
            "mass": mass,  # kg
            "I": I,
            "force_I": [12, 15, 16],  # force for acceleration
            "torque_B": [0, 0, 0],  # torque
        }

        r = [0, 0, 0]
        v = [0, 0, 0]
        w = np.array([0, 0, 0])

        state = np.array(
            [
                *r,  # r
                *v,  # v
                1,
                0,
                0,
                0,
                *w,
            ]
        )

        assert calc_total_energy(mass, I, r, v, w, MU_EARTH) == 0
        next_step = propagate(dt, t, state, params)

        # r = [2400000, 3000000, 3200000] (.5 * F/m * t^2)
        # PE = -79720.08836 J
        # v = [240000, 300000, 320000] (F/m * t)
        # KE = 125000000 J

        assert (
            calc_total_energy(
                mass, I, next_step[0:3], next_step[3:6], next_step[10:14], MU_EARTH
            )
            == -79720.08836 + 125000000
        )

    def test_just_ang_accel(self):
        dt = 0.001
        mass = 2
        I = np.diag((2, 1, 4)) * 100
        torque = [12.0, 15.0, 16.0]

        params = {
            "mass": mass,  # kg
            "I": I,
            "force_I": [0, 0, 0],  # force for acceleration
            "torque_B": torque,  # torque
        }

        r = [0, 0, 0]
        v = [0, 0, 0]
        w0 = np.array([0, 0, 0])

        state = np.array(
            [
                *r,  # r
                *v,  # v
                1,
                0,
                0,
                0,
                *w0,
            ]
        )

        assert calc_total_energy(mass, I, r, v, w0, MU_EARTH) == 0
        next_step = propagate(dt, dt, state, params)

        # Hand calc: from rest, w = dt * tau / I_diag (gyroscopic term ~ w^2 is negligible)
        #   w  = 0.001 * [12/200, 15/100, 16/400]          = [6.0e-5, 1.5e-4, 4.0e-5] rad/s
        #   KE = 0.5 * (200*6e-5^2 + 100*1.5e-4^2 + 400*4e-5^2) = 1.805e-6 J
        assert next_step[10:13] == pytest.approx([6.0e-5, 1.5e-4, 4.0e-5], rel=1e-4)
        assert calc_total_energy(
            mass, I, next_step[0:3], next_step[3:6], next_step[10:13], MU_EARTH
        ) == pytest.approx(1.805e-6, rel=1e-4)


def test_no_I_returns_3DOF():
    state = np.array([0.0, 0.0, 0.0, 1.1, 1.2, 1.3, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    dot = rigid_body_derivative(
        0.0, state, mass=2.0, force_I=[1.0, 0.0, 0.0], I=None, torque_B=[3.0, 4.0, 5.0]
    )
    assert dot.shape == (6,)
    assert np.allclose(dot, [1.1, 1.2, 1.3, 0.5, 0.0, 0.0])
