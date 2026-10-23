import numpy as np
from numpy.linalg import norm
from ...world.constants import G0, MU_EARTH
from ...constants import DEG_TO_RAD
from ...math.transformations import PCI_to_NTW

# TODO: NTW

# def guid_low_thrust_transfer(a_init: float, a_final: float, i_init: float, i_final: float,
#                              Isp: float, F: float, m: float, scale: str = "short", 
#                              nsteps: int = 1000, iter = 1000, lambda_guess = 1.0,
#                              mu: float = MU_EARTH, deg = False) -> tuple[float, float]:
#     """Low Thrust Transfer \\
#     Vallado 4e Algorithm 47 p. 387

#     Parameters
#     ----------
#     a_init : float
#         Initial semi-major axis [m]
#     a_final : float
#         Final semi-major axis [m]
#     i_init : float
#         Initial inclination [rad or deg]
#     i_final : float
#         Final inclination [rad or deg]
#     Isp : float
#         Specific impulse [s]
#     F : float
#         Thrust [N]
#     m : float
#         Mass [kg]
#     mu : float, optional
#         Gravitational parameter of central body [m3/s2]

#     Returns
#     -------
#     tau : float
#         Time of transfer [s]
#     dV : float
#         Delta-V for maneuver [m/s] 
#     """
    
#     if scale != "short":
#         raise ValueError("only short way implemented")
    
#     R = a_final / a_init
    
#     m_dot = -F/Isp/G0
#     m_dot_spec = m_dot/m
#     a_thrust_i = F/m
    
#     if deg:
#         i_final *= DEG_TO_RAD
#         i_init *= DEG_TO_RAD
    
#     di = i_final - i_init
#     theta_p = 0.0
#     dt = 0.0
    
    
#     period = 2*np.pi*np.sqrt(a_init**3 / mu)
#     dt_s = period/nsteps
    
#     # r0 = a_init 
    
#     a_curr = a_init # 1 DU*
    
#     eps = 1e-9
    
#     for i in range(iter):
        
#         c_v = 1 / (4* lambda_guess**2 * a_curr**2 + 1)
        
#         # Steering angle
#         phi_c = np.arctan2(np.cos(theta_p), np.sqrt(1/c_v - 1))
        
#         a_thrust = a_thrust_i / (1 + m_dot_spec*dt)
#         a_NTW = a_thrust * np.array([0, np.cos(phi_c), np.sin(phi_c)])
        
#         # Propagate Kepler
#         rv_curr_ECI = 0.0
#         dcm_ECI_to_NTW = PCI_to_NTW(rv_curr_ECI)
#         a_ECI = dcm_ECI_to_NTW.T @ a_NTW
            
#         if norm(a_final - a_curr < eps) and norm(a_final - a_curr < eps):
#             print(f"Converged in {i} iterations")
#             break
        