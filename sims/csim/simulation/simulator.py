import numpy as np
from numpy.linalg import norm

from ..entities import Spacecraft
from ..math import rk4_func, quat_apply, unit
from ..physics.rigid_body import rigid_body_derivative, RigidBodyParams
from ..physics import grav_accel
from ..world import MU_EARTH, R_EARTH, density
from ..physics.energy import calc_potential_energy, calc_kinetic_energy
from ..physics.aerodynamics import drag_accel, drag_v_rel
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
        self.KE_pos = np.zeros((n_steps+1, 1))

        self.X[0] = state0
        self.a_meas[0] = [0,0,0]
        self.w_meas[0] = [0,0,0]
        self.KE_pos[0] = calc_kinetic_energy(state0[3:6], state0[10:13], self.sc.mass, None) # Put in I later

    def step_one(self, state: np.ndarray, step, debug = False):
        """Return false if simulation is done"""

        ########### Extract state ###########
        r = state[0:3]
        v = state[3:6]
        q_B2I = unit(state[6:10]) # Turns body vector into inertial vector
        w = state[10:13]

        sc = self.sc

        ########### Get measured angular velocity ###########
        w_body = quat_apply(q_B2I, w)

        ########### Simulate next state ###########
        accel = np.zeros(3)

        grav = grav_accel(state[:3])
        accel += grav

        v_rel = drag_v_rel(r, v)
        rho = density(norm(r) - R_EARTH)
        Cd = sc.get_Cd()
        A = 0.02
        
        drag = drag_accel(v_rel, rho, A, sc.mass, Cd)
        accel += drag

        a_meas = accel - grav


        ########### Integrate with RK4 ###########
        params = RigidBodyParams(sc.mass, sc.I, accel*sc.mass, np.zeros(3))

        next_state = rk4_func(self.t, self.dt, state, rigid_body_derivative, params)

        next_state[6:10] = unit(next_state[6:10])

        ########### Energy normalization (NOT WORKING RN BECUASE WE NEED POSITION TO WORK TOO. potentially use ) ###########
        F_non_grav = self.sc.mass * a_meas
        dx = next_state[3:6] * self.dt
        dKE = np.dot(F_non_grav, dx)

        # dE = KE2 - KE1 = 1/2 * m (v2^2 - v1^2)  
        # v2^2 = 2*dE/m + v1^2                     to get what it should be
        # v_act = v_calc * (v2^2)/(v_calc^2)

        v2_2 = 2*dKE/self.sc.mass + np.dot(v,v)
        v_calc_2 = np.dot(next_state[3:6], next_state[3:6])

        # next_state[3:6] = next_state[3:6] * v2_2 / v_calc_2

        # self.KE_pos = calc_kinetic_energy(next_state[3:6], next_state[10:13], self.sc.mass, None) # Put in I later
        

        if debug: 
            print(f"a_grav: {grav} m/s2")
            print(f"a_drag: {drag} m/s2")
            print(f"F_drag: {norm(F_non_grav)} m/s2")
            print(f"a_drag•v: {np.dot(drag, v)}")
        
        return next_state, a_meas, w_body

    def simulate(self, debug = False, max_position = 1e15):

        self.final_step = self.n_steps
        
        for step in range(1,self.n_steps+1):

            if debug:
                print(f"Step {step}:")

            ########### Propagate ###########
            next_state, a_meas, w_meas = self.step_one(self.X[step-1], step, debug)
            self.t += self.dt

            self.X[step] = next_state
            self.a_meas[step] = a_meas
            self.w_meas[step] = w_meas
            
            if debug:
                print()




            ########### Conditions for stopping ###########
            r_sat = norm(self.X[step][0:3])

            if step > 10 and r_sat < R_EARTH:
                print(f"Stopped early after {step} steps ({step * self.dt}s) due to crashing!")
                self.final_step = step
                break

            if step > 10 and r_sat > max_position:
                print(f"Stopped early after {step} steps ({step * self.dt}s) due to being greater than {max_position}m from Earth center!")
                self.final_step = step
                break

        self.X = self.X[:self.final_step]
        self.a_meas = self.a_meas[:self.final_step]
        self.w_meas = self.w_meas[:self.final_step]


    