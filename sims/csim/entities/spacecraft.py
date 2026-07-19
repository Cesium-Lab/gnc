# ruff: noqa: E741
import numpy as np

from .components.sensors import Sensor
from ..physics.relative_motion import eci_relative_to_hill

# TODO: may split between Spacecraft and SpacecraftInterface to mimic what spacecraft actually knows
class Spacecraft:
    """Can be a rocket or satellite or anything

    TODO: change moments of inertia to a function?
    TODO:
    - from prev
        - Add components
        - Add engine
    - get_I #property?
    - calc_CP #property?
    - calc_CG #property?
    - get_mass #property?
    - state_dot function

    """
    def __init__(self, mass: float, I: np.ndarray, sensors: list[Sensor] | None = None):#, state: np.ndarray, state_truth: np.ndarray = None):
        """Keeps track of its measured state and truth state.

        Args:
            mass (float): Initial mass [kg]
            I (np.ndarray): Initial moment of inertia tensor (whether changing or not) [kg m2]
            sensors (list[Sensor], optional): Onboard RPO sensors, sensed every step via `sense()`. Defaults to none.
            state (np.ndarray): Initial measured state [p, v, q, w]
            state_truth (np.ndarray): Initial truth state [p, v, q, w]
        """
        self.mass = mass
        self.I = I
        self.sensors = sensors if sensors is not None else []
        # self.state = state
        # self.truth = state_truth if state_truth else state

    def sense(self, own_state: np.ndarray, target_state: np.ndarray):
        """Runs every onboard sensor against another spacecraft's truth ECI state.

        Args:
            own_state (np.ndarray): This spacecraft's truth ECI state, `[p(3), v(3), ...]`
            target_state (np.ndarray): The other spacecraft's truth ECI state, `[p(3), v(3), ...]`

        Returns:
            dict: `{sensor: (z, valid)}` for each sensor in `self.sensors`
        """
        rho_hill = eci_relative_to_hill(
            target_state[:3], target_state[3:6], own_state[:3], own_state[3:6])

        return {sensor: sensor.measure(rho_hill) for sensor in self.sensors}
