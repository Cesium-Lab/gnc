from .component import Component
import numpy as np

class Fluid(Component):

    def __init__(self, top_pos, burn_time, length = None, density = None, diameter = None, name = "Bodytube", wet_mass = None):


        # wet_mass is unknown
        if length and density and diameter:
            self.volume = length * np.pi / 4 * diameter**2
            self.wet_mass = density * self.volume

            self.length = length
            self.diameter = diameter
            self.density = density

        # Diameter is unknown
        elif length and density and wet_mass:
            self.volume = wet_mass / density
            self.diameter = np.sqrt(4*self.volume / length / np.pi)
            
            self.density = density
            self.wet_mass = wet_mass
            self.length = length

        # Density is unknown
        elif length and diameter and wet_mass:
            self.volume = length * np.pi / 4 * diameter**2
            self.density = wet_mass / self.volume

            self.length = length
            self.diameter = diameter
            self.wet_mass = wet_mass

        # length is unknown
        elif density and diameter and wet_mass:
            self.volume = wet_mass / density
            self.length = 4 * self.volume / np.pi / (diameter**2)

            self.density = density
            self.diameter = diameter
            self.wet_mass = wet_mass

        else:
            raise ValueError('Must define a fluid with three of four of [volume, length, density, diameter, wet_mass]')
            

        self.mass = self.wet_mass
        
        self.length_initial = self.length
        # Moment of inertia about its top point
        self.I_matrix = self._I_matrix_calc()

        self.CG_loc_dry = top_pos + np.array([0, 0, self.length/2])
        
        

        super().__init__(mass=self.wet_mass, top_pos=top_pos, CG_loc = self.CG_loc_dry, I_matrix=self.I_matrix, name = name, type = "Fluid")
        
        self.tb = burn_time

        self.set_plot_elements()
        self.get_2d_points()
        self.set_plot_elements(fmt = 'r:',
                               name = self.name)

    def _I_matrix_calc(self):

        # radius
        r = self.diameter / 2.0
        l = self.length

        # Moment of inertia of cylinder
        self.Ixx = self.Iyy = 1/12 * self.mass * (3*(r**2)+ l*l) # kg.m2
        

        self.Izz = 1/2 * self.mass * (r**2) # kg.m2


        I_matrix = np.array([self.Ixx, self.Iyy, self.Izz])

        return np.diag(I_matrix)
    
    def get_2d_points(self, t = 0):
        
        self.update(t)
        # X axis is along rocket, positive goes from nose to engine
        # Y axis is radial
        r = self.diameter / 2.0 
        bottom_x, bottom_y, bottom_z = self.top_pos + np.array([0, 0, self.length_initial])

        top = bottom_z - self.length
        bottom = bottom_z

        left = bottom_x - r
        right = bottom_x + r

        self.points_2d = np.array([
            [top, left],
            [top, right],
            [bottom, right],
            [bottom, left],
            [top, left]
        ]).T

        return self.points_2d
    
    def update(self, t):

        self.mass = self.get_mass(t)
        self.length = self.__length(t)

        self.CG_loc = self.CG_loc_dry + np.array([0, 0, (self.length_initial - self.length)/2])

        self.I_matrix = self._I_matrix_calc()


    def get_mass(self, t):

        if t < self.tb:
            return self.wet_mass * ( 1 - t/self.tb)
        else: 
            return 0
        

        
        

    def __length(self, t):

        if t < self.tb:
            return self.length_initial * ( 1 - t/self.tb)
        else: 
            return 0
        
    
            


