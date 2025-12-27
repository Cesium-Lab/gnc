# ruff: noqa: E741
import sys
sys.path.append(".")
import pytest

import numpy as np
from csim.physics.energy import calc_total_energy
from csim.physics.rigid_body import rigid_body_derivative, RigidBodyParams
from csim.math.integrators import rk4_func
from csim.world import MU_EARTH
def propagate(dt: float, tmax: float, state0: np.ndarray, params: RigidBodyParams):
    state = state0
    t = 0
    while t < tmax:
        state = rk4_func(t, dt, state, rigid_body_derivative, params)
        state[6:10] /= np.linalg.norm(state[6:10])
        t += dt
    return state

class TestRigidBodyDeriv:

    def test_force_no_torque(self):
        params = RigidBodyParams(
            mass_kg = 2,
            I = np.diag([2,2,2]),
            force_N = [1,0,0],
            torque_Nm = [0,0,0]
        )

        state = [
            0,0,0, # r = 0
            1.1, 1.2, 1.3, # v != 0
            1,0,0,0, # Identity quaternion
            0,0,0 # w = 0
        ]

        expected = [
            1.1, 1.2, 1.3,
            0.5,0,0, # Some acceleration
            0,0,0,0,
            0,0,0
        ]

        dot = rigid_body_derivative(0, state, params)

        assert np.array_equal(dot, expected)

    def test_torque_no_force(self):
        params = RigidBodyParams(
            mass_kg = 2,
            I = np.diag([2,2,2]),
            force_N = [0,0,0],
            torque_Nm = [1.0, 2.0, 4.0]
        )

        state = [
            0,0,0, # r = 0
            1.1, 1.2, 1.3, # v != 0
            1,0,0,0, # Identity quaternion
            0,0,0 # w = 0
        ]

        expected = [
            1.1, 1.2, 1.3,
            0,0,0, # Some acceleration
            0,0,0,0,
            0.5, 1, 2
        ]

        dot = rigid_body_derivative(0, state, params)
        assert np.array_equal(dot, expected)

    def test_random_nonsense(self):
        params = RigidBodyParams(
            mass_kg = 17,
            I = np.array([
                [15, .5, .5],
                [.1, 15, 0],
                [0, 3, 4]
            ]),
            force_N = [12.0, 200.0, 14.0],
            torque_Nm = [0.4, 0.2, 0.7]
        )

        state = np.array([
            0.0, 0.0, 0.0, # r = 0
            1.1, 1.2, 1.3, # v = 0
            0.5, 0.5, 0.5, 0.5,
            2.0, 3.14, 3.52 # ω = 0
            ])
            


        dot = rigid_body_derivative(0, state, params)

        # Generated using Julia as a simple calculator with the textbook functions

        # Simple acceleration
        assert np.array_equal(dot[0:3], [1.1, 1.2, 1.3]) 

        # F/m
        assert np.allclose(dot[3:6], [0.7058823529411765, 11.764705882352942, 0.8235294117647058], atol=1e-12)

        # .5 * ham_prod(q,w)
        assert np.allclose(dot[6:10], [-2.165, 0.595, 0.40500000000000014, 1.165], atol=1e-12)

        # wack one
        assert np.allclose(dot[10:14], [6.1567301516750925, -4.715818201011166, 6.225913650758373], atol=1e-12)


class TestRigidBodyRk4Integration:

    
    def test_0_time(self):
        dt = 0
        t = 1.0

        I = np.diag((1,1,1))

        # Initialized with random but nonzero params
        params = RigidBodyParams(
            17.0, # kg
            I,
            [12.0, 200.0, 14.0], # force
            [0.4, 0.2, 0.7] # torque
        )

        state = np.array([
            5,6,7, # r = 0
            1.1, 1.2, 1.3, # v = 0
            0.5, 0.5, 0.5, 0.5,
            2.0, 3.14, 3.52 # ω = 0
            ])
        
        next_step = rk4_func(t, dt, state, rigid_body_derivative, params)

        assert np.array_equal(next_step, state)

    def test_force(self):
        dt = 0.001
        t = 1

        
        params = RigidBodyParams(
            17.0, # kg
            np.diag((1,1,1)),
            [1.0, 10.0, 100.0], # force for acceleration
            [0.0, 0.0, 0.0] # torque
        )

        state = np.array([
            0.1, 100.4, 5, # r
            1.1, 1.2, 1.3, # v
            1,0,0,0,
            0,0,0
            ])
        
        next_step = propagate(dt, t, state, params)
        # next_step = rk4_func(t, dt, state, rigid_body_derivative, params)

        assert np.allclose(next_step[0:3], [1.2294117647058824, 101.89411764705883, 9.241176470588236], rtol=1e-6)
        assert np.allclose(next_step[3:6], [1.1588235294117648, 1.788235294117647, 7.182352941176471], rtol=1e-6)
        assert np.array_equal(next_step[6:10], [1,0,0,0])
        assert np.array_equal(next_step[10:14], [0,0,0])

    def test_pi_rotation(self):
        dt = 0.001
        t = 3.14159

        
        params = RigidBodyParams(
            17.0, # kg
            np.diag((1,1,1)),
            [0,0,0], # force for acceleration
            [0,0,0] # torque
        )

        state = np.array([
            0,0,0, # r
            0,0,0, # v
            1,0,0,0,
            1,0,0 # 1 rad/s
            ])
        
        next_step = propagate(dt, t, state, params)
        # next_step = rk4_func(t, dt, state, rigid_body_derivative, params)

        assert np.array_equal(next_step[0:3], [0,0,0]) # no positions 
        assert np.array_equal(next_step[3:6], [0,0,0]) # no positions 
        assert np.allclose(next_step[6:10], [0,1,0,0], atol=1e-3) # 180 around X axis
        assert np.array_equal(next_step[10:14], [1,0,0]) # No torque to change it


# Would test torque but it is nonlinear so don't really want to so just gotta trust textbooks

class TestRigidBodyEnergy:

    def test_stationary_with_gravity(self):
        dt = 0.01
        t = 60
        mass = 1
        I = np.diag((2,1,4))
        # Initialized with random but nonzero params
        params = RigidBodyParams(
            mass, # kg
            I,
            [0,0,0], # force for acceleration
            [0,0,0] # torque
        )

        r = [6800e3,0,0]
        v = [0,0,0]
        w = np.array([1,0,0])

        state = np.array([
            *r, # r
            *v, # v
            1,0,0,0,
            *w
            ])
        
        assert calc_total_energy(mass, I, r, v, w, MU_EARTH) == pytest.approx(-58617712.)
        next_step = propagate(dt, t, state, params)

        assert (calc_total_energy(mass, I, next_step[0:3], next_step[3:6], next_step[10:14], MU_EARTH) 
                == pytest.approx(-58617712, rel=0.0001))
        
    def test_just_spinning(self):
        dt = 0.01
        t = 60
        mass = 1
        I = np.diag((2,1,4)) * 100
        
        params = RigidBodyParams(
            mass, # kg
            I,
            [0,0,0], # force for acceleration
            [0,0,0] # torque
        )

        r = [0,0,0]
        v = [0,0,0]
        w = np.array([4,0,0])

        state = np.array([
            *r, # r
            *v, # v
            1,0,0,0,
            *w
            ])
        
        assert calc_total_energy(mass, I, r, v, w, MU_EARTH) == 1600
        next_step = propagate(dt, t, state, params)

        assert (calc_total_energy(mass, I, next_step[0:3], next_step[3:6], next_step[10:14], MU_EARTH) 
                == 1600)
        
    def test_just_accel(self):
        dt = 0.001
        t = 20
        mass = .001
        I = np.diag((2,1,4)) * 100
        
        params = RigidBodyParams(
            mass, # kg
            I,
            [12,15,16], # force for acceleration
            [0,0,0] # torque
        )

        r = [0,0,0]
        v = [0,0,0]
        w = np.array([0,0,0])

        state = np.array([
            *r, # r
            *v, # v
            1,0,0,0,
            *w
            ])
        
        assert calc_total_energy(mass, I, r, v, w, MU_EARTH) == 0
        next_step = propagate(dt, t, state, params)

        # r = [2400000, 3000000, 3200000] (.5 * F/m * t^2)
        # PE = -79720.08836 J
        # v = [240000, 300000, 320000] (F/m * t)
        # KE = 125000000 J

        assert (calc_total_energy(mass, I, next_step[0:3], next_step[3:6], next_step[10:14], MU_EARTH) 
                == -79720.08836 + 125000000)

    def test_just_ang_accel(self):
        dt = 0.001
        mass = 2
        I = np.diag((2,1,4)) * 100
        torque = [12.0, 15.0, 16.0] 
        
        params = RigidBodyParams(
            mass, # kg
            I,
            [0,0,0], # force for acceleration
            torque # torque
        )

        r = [0,0,0]
        v = [0,0,0]
        w0 = np.array([0,0,0])

        state = np.array([
            *r, # r
            *v, # v
            1,0,0,0,
            *w0
            ])
        
        assert calc_total_energy(mass, I, r, v, w0, MU_EARTH) == 0
        next_step = propagate(dt, dt, state, params)

        w_dot = np.linalg.inv(I) @ (torque - np.cross(w0, I@w0))
        w = w0 + dt * w_dot
        KE = .5 * w.T @ I @ w

        assert calc_total_energy(mass, I, next_step[0:3], next_step[3:6], next_step[10:13], MU_EARTH) == KE

def test_no_I():
    params = RigidBodyParams(
        mass_kg = 2,
        I = None,
        force_N = [1,0,0],
        torque_Nm = [0,0,0]
    )

    state = [
        0,0,0, # r = 0
        1.1, 1.2, 1.3, # v != 0
        1,0,0,0, # Identity quaternion
        0,0,0 # w = 0
    ]

    expected = [
        1.1, 1.2, 1.3,
        0.5,0,0, # Some acceleration
        0,0,0,0,
        0,0,0
    ]

    dot = rigid_body_derivative(0, state, params)

    assert np.array_equal(dot, expected)