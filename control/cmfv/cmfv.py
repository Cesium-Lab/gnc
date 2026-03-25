"""Module for the classes"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from dataclasses import dataclass

DEG_TO_RAD = np.pi/180
RAD_TO_DEG = 180/np.pi
MAX_ANGLE = np.pi/2

# CONVENTION: 0º IS OPEN AND 90º IS CLOSED

# ----------------------------------------------------------------------------------------------------------------------------------
#                   Observer
# ----------------------------------------------------------------------------------------------------------------------------------

class PressureTransducer:
    def __init__(self, sigma_p: float, p0 = 0):
        self.sigma_p = sigma_p
        self.p_hat = p0

    def observe(self, p):
        """Adds noise to pressure and returns"""
        self.p_hat = np.random.normal(p, self.sigma_p)
        return self.p_hat
    
# ----------------------------------------------------------------------------------------------------------------------------------
#                   Valve
# ----------------------------------------------------------------------------------------------------------------------------------

class ValveAngle:
    def __init__(self, dt: float, theta0 = 0.0, sigma = 0, bounds = (0,90)):
        self.dt = dt
        self.theta = theta0
        self.sigma = sigma
        self.bounds = bounds

    def step(self, theta_dot: float):
        """Integrates then adds noise"""
        self.theta = self.theta + self.dt * theta_dot
        self.theta = np.random.normal(self.theta, self.sigma) # Maybe not add this?
        return np.clip(self.theta, *self.bounds)
    
class Servo:
    def __init__(self, dt: float, Kp: float, theta0 = 0.0, sigma = 0.0, bounds = (0,np.pi/2)) -> float:
        """Kp is for the servo control loop"""
        self.dt = dt
        self.omega = 0.0

        self.theta = theta0
        self.sigma = sigma
        self.bounds = bounds
        self.Kp = Kp

        self.max_omega = 300.      # deg/s
        self.max_accel = 1000.     # deg/s²

    def step(self, theta_cmd: float):
        err = theta_cmd - self.theta

        omega_cmd = self.Kp * err

        # acceleration limit
        domega = np.clip(omega_cmd - self.omega,
                         -self.max_accel * self.dt,
                         self.max_accel * self.dt)

        self.omega += domega
        self.omega = np.clip(self.omega, -self.max_omega, self.max_omega)
        self.theta += self.omega * self.dt

        self.theta += np.random.normal(0, self.sigma) # Maybe not add this?
        clipped_theta = np.clip(self.theta, *self.bounds)
        self.theta = clipped_theta

        if clipped_theta != self.theta:
            self.omega = 0


        return self.theta

class Valve:
    def __init__(self, rho = 1.0, A = 1.0):
        self.rho = rho
        self.A = A
        

    def across(self, p1: float, v1: float, theta: float):
        """(p1, v1) --> p2 TODO: fix"""
        E1 = p1 + 0.5*self.rho*v1*v1
        
        E_loss = (theta/np.pi/2)*E1 # Make this better but it's linear for now

        # Function I threw together from the last cell
        # E_loss = (1 - np.cos(theta*2)) * E1 

        E2 = E1 - E_loss

        # Must be traveling at same speed since pipe is same A_cs
        v2 = v1
        p2 = E2 - 0.5*self.rho*v2*v2

        p2 = -p1*(theta/np.pi*2) + p1

        return p2
    
# ----------------------------------------------------------------------------------------------------------------------------------
#                   Controllers
# ----------------------------------------------------------------------------------------------------------------------------------

CONTROLLER_ANGLE_BOUND = (0,np.pi/2)

class Controller:
    def __init__(self):
        pass

    def control(self):
        pass

class PController(Controller):
    """If want to INCREASE PRESSURE, DECREASE ANGLE to make it MORE open"""
    def __init__(self, Kp: float):
        self.Kp = Kp

    def control(self, p_ref: float, p_meas: float, angle: float):
        dp = -(p_ref - p_meas)
        command = angle + self.Kp * dp
        return np.clip(command, *CONTROLLER_ANGLE_BOUND) # Torque Nm
    
        
# # Kp has units of (deg/s)/Pa because of transfer function
# class PDController(Controller):
#     def __init__(self, Kp: float, Kd: float, dt: float):
#         self.Kp = Kp
#         self.Kd = Kd
#         self.dt = dt
#         self.err_prev = None
        

#     def control(self, p_ref: float, p_meas: float):
#         err = p_ref - p_meas

#         if self.err_prev is None:
#             self.err_prev = err
#             de = 0
#         else:
#             de = (err - self.err_prev) / self.dt
#             self.err_prev = err


#         P_term = self.Kp * err
#         D_term = self.Kd * de

#         return -np.clip(P_term + D_term, -MAX_ALPHA_NM, MAX_ALPHA_NM) # Torque Nm
    

    
# class PIController(Controller):
#     def __init__(self, Kp: float, Ki: float, dt: float, windup_tolerance: float):
#         self.Kp = Kp
#         self.Ki = Ki
#         self.dt = dt
#         self.tol = windup_tolerance

#         self.integraded_err = 0

#     def control(self, p_ref: float, p_meas: float):
#         err = p_ref - p_meas

#         if abs(self.integraded_err) < self.tol:
#             self.integraded_err += self.dt*err


#         P_term = self.Kp * err
#         I_term = self.Ki * self.integraded_err
#         # print(de)
#         # print(P_term)
#         # print(D_term)
#         return -np.clip(P_term + I_term, -MAX_ALPHA_NM, MAX_ALPHA_NM) # Torque Nm
    
# class PIDController(Controller):
#     def __init__(self, Kp: float, Ki: float, Kd: float, dt: float, windup_tolerance: float):
#         self.Kp = Kp
#         self.Ki = Ki
#         self.Kd= Kd
#         self.dt = dt

#         self.integraded_err = 0
#         self.tol = windup_tolerance

#         self.err_prev = None


#     def control(self, p_ref: float, p_meas: float):
#         err = p_ref - p_meas


#         # Integral
#         if abs(self.integraded_err) < self.tol:
#             self.integraded_err += self.dt*err

#         # Derivative
#         if self.err_prev is None:
#             self.err_prev = err
#             de = 0
#         else:
#             de = (err - self.err_prev) / self.dt
#             self.err_prev = err


#         P_term = self.Kp * err
#         I_term = self.Ki * self.integraded_err
#         D_term = self.Kd * de

#         return -np.clip(P_term + I_term + D_term, -MAX_ALPHA_NM, MAX_ALPHA_NM) # Torque Nm

# ----------------------------------------------------------------------------------------------------------------------------------
#                   Simulations
# ----------------------------------------------------------------------------------------------------------------------------------



@dataclass
class CmfvSimOutputs:
    time: np.ndarray
    
    # Truth
    theta: np.ndarray
    p_truth: np.ndarray

    # Command
    theta_cmd: np.ndarray
    p_setpoint: np.ndarray

    # Observation
    p_meas: np.ndarray

@dataclass 
class CmfvSimInputs:
    # Sim
    dt: float
    nsteps: int
    servo_KP: int = 5

    # 

    # Noise
    sigma_angle: float = 0
    sigma_p: float = 0



    seed: int | None = None

@dataclass 
class CmfvSimInitialConditions:
    p_tank: float # Pa
    v: float # m/s
    theta: float

# ----------------------------------------------------------------------------------------------------------------------------------
#                   Running Sim
# ----------------------------------------------------------------------------------------------------------------------------------


def run_sim(sim_in: CmfvSimInputs, ICs: CmfvSimInitialConditions, controller: Controller, p_r_func: Callable[[float], float]) -> CmfvSimOutputs:
    
    if sim_in.seed is not None:
        np.random.seed(sim_in.seed) 

    angle_integrator = ValveAngle(sim_in.dt, ICs.theta, sim_in.sigma_angle)
    valve = Valve(rho=1, A=1)
    
    p_out0 = valve.across(ICs.p_tank, ICs.v, ICs.theta) # Initial pressure on other side
    ducer = PressureTransducer(sim_in.sigma_p, p_out0)

    N = sim_in.nsteps
    dt = sim_in.dt

    ts = np.arange(N+1)*dt

    angles = np.zeros((N+1))
    p_truth = np.zeros((N+1))
    p_meas = np.zeros((N+1))
    p_setpoints = np.zeros((N+1))
    theta_dots = np.zeros((N+1))

    angles[0] = ICs.theta
    p_truth[0] = p_out0

    p_meas[0] = ducer.observe(p_truth[0])
    p_setpoints[0] = p_r_func(ts[0])
    theta_dots[0] = controller.control(p_setpoints[0], p_meas[0])

    for i in range(N):

        # Advance angle according to command
        angle = angle_integrator.step(theta_dots[i])
        
        # Just having velocity equal to starting one 
        v = ICs.v
        p_downstream = valve.across(ICs.p_tank, v, angle)

        angles[i+1] = angle
        p_truth[i+1] = p_downstream
        p_meas[i+1] = ducer.observe(p_downstream)

        # Get setpoint for the current step
        p_setpoints[i+1] = p_r_func(ts[i])
        theta_dots[i+1] = controller.control(p_setpoints[i+1], p_meas[i+1])

    return CmfvSimOutputs(
        time = ts,
        theta = angles,
        p_truth = p_truth,
        theta_dot_cmd = theta_dots,
        p_meas = p_meas,
        p_setpoint = p_setpoints
    )
    
