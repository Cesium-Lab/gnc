from .component import Component
import numpy as np


class Nosecone(Component):
    def __init__(
        self,
        length,
        shape,
        D_o,
        thickness,
        shoulder_diameter=None,
        shoulder_length=None,
        parabolic_K=None,
        density=None,
        name="Nosecone",
        mass=None,
    ):

        self.length = length
        self.shape = shape
        self.D_o = D_o

        self.thickness = thickness
        self.shoulder_diameter = shoulder_diameter
        self.parabolic_K = parabolic_K

        self._attributes_based_on_shape(shape)

        self.I_matrix = 0  # self._I_matrix_calc()
        self.mass = mass

        super().__init__(
            mass=self.mass,
            top_pos=0,
            CG_loc=self.CG_loc,
            I_matrix=self.I_matrix,
            name=name,
            type="Nosecone",
        )

        self.C_Na = 2.0

        self.set_plot_elements()
        self.get_2d_points()
        self.set_plot_elements(fmt="r:", name=self.name)

    def _I_matrix_calc(self):

        # radius
        r = self.D_o / 2.0
        length = self.length
        m = self.mass

        # I matrix of a straight cone
        self.Ixx = self.Iyy = 1 / 10 * m * length**2 + 3 / 20 * m * r**2
        self.Izz = 3 / 10 * m * r**2

        I_matrix = np.array([self.Ixx, self.Iyy, self.Izz])

        return np.diag(I_matrix)

    def get_2d_points(self):

        # X axis is along rocket, positive goes from nose to engine
        # Y axis is radial

        x_arr = np.linspace(0, self.length, 100).tolist()
        x_arr_bottom = x_arr[::-1]

        y_arr = [self.func(i) for i in x_arr]
        y_arr_bottom = [-i for i in y_arr][::-1]

        x_points = x_arr + x_arr_bottom + [x_arr[0]]
        y_points = y_arr + y_arr_bottom + [y_arr[0]]

        self.points_2d = [x_points, y_points]

    def _attributes_based_on_shape(self, shape):  # TODO

        r = self.D_o / 2
        length = self.length
        parabolic_K = self.parabolic_K

        # For shape formulas look at wikipedia or http://servidor.demec.ufpr.br/CFD/bibliografia/aerodinamica/Crowell_1996.pdf
        if shape == "ogive":
            rho = (r * r + length * length) / 2 / r
            self.func = lambda x: np.sqrt(rho**2 - (length - x) ** 2) + r - rho

            # a = length * rho * rho
            # b = - length**3 / 3
            # c = -(rho-r) * rho * rho * np.arcsin(length/rho)
            # self.volume = np.pi * (a + b + c)

            # print(self.volume)
            # self.volume = self._volume_from_func()
            # print(self.volume)

            # a = np.sqrt(rho**2 - length**2)
            # b = rho*rho / 2 * np.arcsin(length/rho)
            # c = -length * (rho - r)
            # self.wind_area = length * np.pi * (a + b + c)
            # self.bruh = (a,b,c)

            # Analysis of New Aerodynamic Design of the Nose Cone Section Using CFD and SPH
            # T = length/self.D_o
            # Y = (4*T*T + 1)/self.D_o
            # Z = (4*T*T - 1)/self.D_o
            # self.wind_area = (4*Y*Y*np.arcsin(T/Y) - 4*T*Z) * np.pi * r**2

            # Approx at cone then 1.3 cone to ogive
            self.wind_area = np.pi * r * np.sqrt(length**2 + r**2) * 1.3

            # Good estimate for L > 6r,
            self.CP_loc = np.array([0, 0, 1 - 0.534 * length])

            # This is probably wrong
            self.CG_loc = np.array([0, 0, 1 - 0.534 * length])

        elif shape == "conical":
            # conical is so bad
            self.func = lambda x: x * r / length
            self.half_angle = np.arctan(r / length)

            # self.wind_area = np.pi * r * np.sqrt(r**2 + length**2)

            # It is a cone. That's where the 1/3 area in I think
            self.CP_loc = np.array([0, 0, 1 - length / 3.0])

            # This is probably wrong
            self.CG_loc = np.array([0, 0, 1 - length / 3.0])

        elif shape == "parabolic":
            if not parabolic_K or parabolic_K < 0 or parabolic_K > 1:
                raise ValueError(
                    "Parabolic nosecone requires K between 0 and 1 inclusive"
                )

            self.func = lambda x: (
                r
                * (2 * x / length - parabolic_K * (x / length) ** 2)
                / (2 - parabolic_K)
            )

            self.CP_loc = np.array([0, 0, 1 - length / 2.0])

            # This is probably wrong
            self.CG_loc = np.array([0, 0, 1 - length / 2.0])

        else:
            raise ValueError(
                "Shape must be 'ogive', 'conical', or 'parabolic'. Other shapes not supported yet"
            )

    # def _volume_from_func(self):

    #     # Better calc later
    #     x_arr = np.linspace(0, self.length, 1000)
    #     dx = x_arr[1] - x_arr[0]

    #     # print(dx)

    #     sum = 0.0

    #     for x in x_arr:

    #         r1 = self.func(x)
    #         r2 = self.func(x + dx)

    #         dr_dx = (r2 - r1) / dx

    #         a = (r1+r2)/2 * np.sqrt(1 + (dr_dx)**2) * dx

    #         # arr.append(1/ np.sqrt(1 + dr_dx**2))
    #         sum += a

    #     sum *= 2*np.pi

    #     return sum
