from dataclasses import dataclass

import numpy as np

from cesium.orbital import coes_to_rv
from cesium.world.constants import MU_EARTH

from .components import Propulsion


@dataclass
class Spacecraft:
    """ECI state (km, km/s) plus ballistic properties for a single spacecraft."""

    r: np.ndarray | list
    v: np.ndarray | list
    mass: float
    dry_mass: float = None
    #     area: float
    #     cd: float = 2.2
    name: str = ""
    prop_system: Propulsion = None
    thrust_command: object = None

    def __post_init__(self):
        self.r = np.asarray(self.r, dtype=float)
        self.v = np.asarray(self.v, dtype=float)

    @classmethod
    def from_elements(
        cls,
        elements,
        mass,
        area,
        cd=2.2,
        name="",
        mu=MU_EARTH,
        *,
        thruster=None,
        dry_mass=None,
        thrust_command=None,
    ):
        """Build from classical elements (a, e, inc, raan, arg_peri, M). Angles in rad."""
        r, v = coes_to_rv(elements, MU_EARTH)
        return cls(r, v, mass, area, cd, name, thruster, dry_mass, thrust_command)


# # TODO: may split between Spacecraft and SpacecraftInterface to mimic what spacecraft actually knows
# class Spacecraft:
#     """Mass properties of a simulated vehicle (rocket, satellite, etc.).

#     Attributes
#     ----------
#     mass : float
#         Vehicle mass [kg].
#     I : np.ndarray (3, 3) or None
#         Moment of inertia tensor about the center of mass, in the body frame
#         [kg*m2]. None disables rotational dynamics.

#     Notes
#     -----
#     TODO: change moments of inertia to a function?
#     TODO:
#     - from prev
#         - Add components
#         - Add engine
#         - Add sensors, which get sensed every step
#     - get_I #property?
#     - calc_CP #property?
#     - calc_CG #property?
#     - get_mass #property?
#     - state_dot function

#     """

#     def __init__(
#         self, mass: float, I: np.ndarray
#     ):  # , state: np.ndarray, state_truth: np.ndarray = None):
#         """
#         Parameters
#         ----------
#         mass : float
#             Initial vehicle mass [kg].
#         I : np.ndarray (3, 3) or None
#             Initial moment of inertia tensor about the center of mass, in the body
#             frame [kg*m2]. None disables rotational dynamics.
#         """
#         self.mass = mass
#         self.I = I
#         # self.state = state
#         # self.truth = state_truth if state_truth else state
