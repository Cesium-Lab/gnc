from .component import Component
import numpy as np


class Protuberance(Component):
    def __init__(
        self,
        top_pos,
        length,
        diameter,
        distance_protruded,
        thickness=None,
        density=None,
        name="Protuberance",
        mass=None,
    ):

        self.diameter = diameter
        self.length = length
        # self.A_cs =

        # Temporary
        mass = 0
        CG_loc = top_pos + np.array([0, 0, length / 2])

        super().__init__(
            mass,
            top_pos,
            CG_loc=CG_loc,
            I_matrix=self.I_matrix,
            name=name,
            type="Boattail",
        )
        pass
