import numpy as np


# TODO: may split between Spacecraft and SpacecraftInterface to mimic what spacecraft actually knows
class Spacecraft:
    """Mass properties of a simulated vehicle (rocket, satellite, etc.).

    Attributes
    ----------
    mass : float
        Vehicle mass [kg].
    I : np.ndarray (3, 3) or None
        Moment of inertia tensor about the center of mass, in the body frame
        [kg*m2]. None disables rotational dynamics.

    Notes
    -----
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

    def __init__(
        self, mass: float, I: np.ndarray
    ):  # , state: np.ndarray, state_truth: np.ndarray = None):
        """
        Parameters
        ----------
        mass : float
            Initial vehicle mass [kg].
        I : np.ndarray (3, 3) or None
            Initial moment of inertia tensor about the center of mass, in the body
            frame [kg*m2]. None disables rotational dynamics.
        """
        self.mass = mass
        self.I = I
        # self.state = state
        # self.truth = state_truth if state_truth else state
