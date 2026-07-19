from .component import Component
import numpy as np

class Bodytube(Component):

    def __init__(self, top_pos, length, density = None, thickness = None, D_o = None, D_i = None, name = "Bodytube", mass = None):


        # If defined by outer diameter and inner diameter
        if D_o and D_i:
            thickness = (D_o - D_i) / 2.0

        # If defined by inner diameter and thickness
        elif D_i and thickness:
            D_o = D_i + 2*thickness

        # If defined by inner diameter and thickness
        elif D_o and thickness:
            D_i = D_o - 2*thickness

        else:
            raise ValueError('Must define a bodytube with two of three of [thickness, OD, ID]')
            
        self.A_cs = np.pi / 4.0 * (D_o**2 - D_i**2)

        self.volume = length * self.A_cs

        # If mass is a parameter, then it overrides the 
        if mass:
            self.mass = mass
            self.density = mass / self.volume
        else:    
            self.mass = density * self.volume
            self.density = density


        self.length = length
        self.thickness = thickness
        self.D_o = D_o
        self.D_i = D_i

        self.wind_area = np.pi * D_o * length

        # Moment of inertia about its top point
        self.I_matrix = self._I_matrix_calc()

        CG_loc = top_pos + np.array([0, 0, self.length/2])
        self.CP_loc = CG_loc

        super().__init__(mass=self.mass, top_pos=top_pos, CG_loc = CG_loc, I_matrix=self.I_matrix, name = name, type = "Bodytube")

        self.C_Na = 0.

        self.set_plot_elements()
        self.get_2d_points()
        self.set_plot_elements(fmt = 'b:',
                               name = self.name)

    def _I_matrix_calc(self):

        # radius
        r_o = self.D_o / 2.0
        r_i = self.D_i / 2.0
        l = self.length
        # Moment of inertia of cylindrical thin shell
        self.Ixx = self.Iyy = 1/12 * self.mass * (3*(r_o**2 + r_i**2)+ l*l) # kg.m2
        

        self.Izz = 1/2 * self.mass * (r_o**2 + r_i**2) # kg.m2


        I_matrix = np.array([self.Ixx, self.Iyy, self.Izz])

        return np.diag(I_matrix)
    
    def get_2d_points(self):
        
        # X axis is along rocket, positive goes from nose to engine
        # Y axis is radial
        r = self.D_o / 2.0
        top_x, top_y, top_z = self.top_pos

        top = top_z
        bottom = top_z + self.length

        left = top_x - r
        right = top_x + r

        self.points_2d = np.array([
            [top, left],
            [top, right],
            [bottom, right],
            [bottom, left],
            [top, left]
        ]).T

        return self.points_2d
