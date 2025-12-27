import numpy as np
from numpy.linalg import norm

from ..entities import Spacecraft
from ..math import rk4_func, quat_apply, unit
from ..physics.rigid_body import rigid_body_derivative, RigidBodyParams
from ..physics import grav_accel
from ..world import MU_EARTH
from ..physics.energy import calc_potential_energy, calc_kinetic_energy
# TODO: def finish this

# TODO: Different integrators for better ECI-ECEF OR interpolate every second or something

class Simulator:
    def __init__(self, state0: np.ndarray, t0: float, dt: float, n_steps: float, spacecraft: Spacecraft):

        self.t = t0
        self.dt = dt
        self.n_steps = n_steps
        self.sc = spacecraft

        self.X = np.zeros((n_steps+1, 13))
        self.a_meas = np.zeros((n_steps+1, 3))
        self.w_meas = np.zeros((n_steps+1, 3))
        self.X[0] = state0

        self.a_meas[0] = [0,0,0]
        self.w_meas[0] = [0,0,0]

    def step_one(self, state: np.ndarray):
        """Return false if simulation is done"""

        # Get measured angular velocity
        w = state[10:13]
        q_B2I = unit(state[6:10]) # Turns body vector into inertial vector
        # q_B2I = state[6:10] # 
        w_body = quat_apply(q_B2I, w)


        # Simulate next state
        grav = grav_accel(state[:3])
        accel = grav
        a_meas = accel - grav

        # Propagate 1 step
        sc = self.sc
        params = RigidBodyParams(sc.mass, sc.I, accel*sc.mass, np.zeros(3))

        next_state = rk4_func(self.t, self.dt, state, rigid_body_derivative, params)

        next_state[6:10] = unit(next_state[6:10])
        
        return next_state, a_meas, w_body

    def simulate(self):

        self.final_step = self.n_steps
        
        for step in range(1,self.n_steps+1):

            next_step, a_meas, w_meas = self.step_one(self.X[step-1])
            self.t += self.dt

            self.X[step] = next_step
            self.a_meas[step] = a_meas
            self.w_meas[step] = w_meas

            # Vz
            # if step > 10 and self.X[step][2] <= 0:
            #     # breakpoint()
            #     self.final_step = step
            #     break


        self.X = self.X[:self.final_step]


    