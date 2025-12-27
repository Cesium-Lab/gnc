import numpy as np

# TODO: may split between Spacecraft and SpacecraftInterface to mimic what spacecraft actually knows 
class Spacecraft:
    """Can be a rocket or satellite or anything
    
    TODO: change moments of inertia to a function?
    TODO:
    - from prev
        - Add components
        - Add engine
        - Add sensors, which get sensed every step
    - get_I #property?
    - calc_CP #property?
    - calc_CG #property?
    - get_mass #property?
    - state_dot function
    
    """
    def __init__(self, mass: float, I: np.ndarray):#, state: np.ndarray, state_truth: np.ndarray = None):
        """Keeps track of its measured state and truth state.

        Args:
            mass (float): Initial mass [kg]
            I (np.ndarray): Initial moment of inertia tensor (whether changing or not) [kg m2]
            state (np.ndarray): Initial measured state [p, v, q, w]
            state_truth (np.ndarray): Initial truth state [p, v, q, w]
        """
        self.mass = mass
        self.I = I
        # self.state = state
        # self.truth = state_truth if state_truth else state
