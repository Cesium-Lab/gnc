# from .spacecraft import Spacecraft


# class Simulator:
#     """Propagate a list of satellites forward in time under central-body + perturbations.

#     All N satellites are integrated in a single solve_ivp call for efficiency.
#     """

#     def __init__(
#         self, satellites: list[Spacecraft], rtol=1e-12, atol=1e-12, method="DOP853"
#     ):
#         self.satellites = list(satellites)
#         self.rtol = rtol
#         self.atol = atol
#         self.method = method

#     def propagate(self, tspan, dt):
#         N = len(self.satellites)
#         initial_state = np.zeros((N, N_STATES))
