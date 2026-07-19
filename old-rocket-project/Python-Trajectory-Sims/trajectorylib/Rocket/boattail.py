from .component import Component
import numpy as np


class Boattail(Component):
    def __init__(
        self,
        top_pos,
        length,
        D_front,
        D_aft,
        thickness,
        density=None,
        angle=None,
        name="Boattail",
        mass=None,
    ):

        self.D_front = D_front
        self.D_aft = D_aft
        self.length = length
        self.thickness = thickness

        if not all([D_front, D_aft, top_pos, length, thickness]):
            raise ValueError(
                'Must define a bodytube with "D_front", "D_aft", "top_pos", "length", and "thickness"'
            )

        R = D_front / 2.0
        r = D_aft / 2.0

        self.wind_area = np.pi * (R + r) * np.sqrt(length**2 + (R - r) ** 2)

        if angle:
            self.angle = angle
        else:
            self.angle = np.arctan((R - r) / (length))  # rad

        self.volume = thickness * self.wind_area

        # If mass is a parameter, then it overrides the
        if mass:
            self.mass = mass
            self.density = mass / self.volume
        elif density:
            self.mass = density * self.volume
            self.density = density
        else:
            raise ValueError('Must define boattail with "mass" or "density"')

        self.I_matrix = self._I_matrix_calc()

        # CG/CP Calculation https://www.physicsforums.com/threads/equation-for-centre-of-gravity-of-a-hollow-conical-frustum.732155/
        # this also works --> z_pos = length/3 * (1 + (1 - D_front/D_aft) / (1 - (D_front/D_aft)**2))
        z_pos = length / 3 * (D_front + 2 * D_aft) / (D_front + D_aft)
        self.CG_loc = top_pos + np.array([0, 0, z_pos])
        self.CP_loc = top_pos + np.array([0, 0, z_pos])

        super().__init__(
            mass=self.mass,
            top_pos=top_pos,
            CG_loc=self.CG_loc,
            I_matrix=self.I_matrix,
            name=name,
            type="Boattail",
        )

        self.C_Na = 2 * ((self.D_aft / self.D_front) ** 2 - 1)
        # print(f"C_Na: {self.C_Na}")

        self.set_plot_elements()
        self.get_2d_points()
        self.set_plot_elements(fmt="g:", name=self.name)

    def _I_matrix_calc(self):  # TODO

        # ratio = self.D_front / self.D_aft
        # l = self.length

        # Moment of inertia of hollow cone
        # self.Ixx = self.Iyy = 1/12 * self.mass * (2*(r_o**2 + r_i**2)+ l*l) # kg.m2
        self.Iyy = self.Ixx = 0  # kg.m2

        self.Izz = 1 / 2 * self.mass * (self.D_front**2 - self.D_aft**2)  # kg.m2

        I_matrix = np.array([self.Ixx, self.Iyy, self.Izz])

        return np.diag(I_matrix)

    def get_2d_points(self):

        # X axis is along rocket, positive goes from nose to engine
        # Y axis is radial
        r_front = self.D_front / 2.0
        r_aft = self.D_aft / 2.0
        top_x, top_y, top_z = self.top_pos

        top = top_z
        bottom = top_z + self.length

        top_left = top_x - r_front
        top_right = top_x + r_front

        bottom_left = top_x - r_aft
        bottom_right = top_x + r_aft

        self.points_2d = np.array(
            [
                [top, top_right],
                [top, top_left],
                [bottom, bottom_left],
                [bottom, bottom_right],
                [top, top_right],
            ]
        ).T

        return self.points_2d
