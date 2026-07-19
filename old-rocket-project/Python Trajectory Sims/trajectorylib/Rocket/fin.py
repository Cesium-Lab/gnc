from .component import Component
import numpy as np
from ..constants import *

class Fin(Component):

    # only hex fins for now
    def __init__(self, top_pos,
                 roll_orientation,
                 span, thickness,
                 Ct, Cr, sweep_angle,
                 boattail_length,
                 rocket_diam, aft_diam,
                 name = "Fin", density = None, mass = None):

        # Fin parameters
        self.sweep_angle = sweep_angle * DEG2RAD # degrees
        self.span = span
        self.Cr = Cr
        self.Ct = Ct
        self.Cm = (Cr+Ct)/2.0 # This is wrong given the fin dimensions though
        self.thickness = thickness


        # Rocket parameters
        self.rocket_diam = rocket_diam
        self.aft_diam = aft_diam
        self.boattail_length = boattail_length
        self.roll_orientation = roll_orientation * DEG2RAD

        self.tip_top_pos = top_pos + np.array([rocket_diam /2* np.cos(self.roll_orientation), rocket_diam/2 * np.sin(self.roll_orientation), 0])
        
        self.__dimensions_calculations()
        
        z, r = self.CG_2d

        CG_loc = self.tip_top_pos + np.array([r * np.cos(self.roll_orientation), r * np.sin(self.roll_orientation), z]) 

        self.CP_loc = CG_loc

        # print(CG_loc * M2FT)
        self.volume = self.area * thickness

        # If mass is a parameter, then it overrides the 
        if mass:
            self.mass = mass
            self.density = mass / self.volume
        else:    
            self.mass = density * self.volume
            self.density = density

        # self.I_matrix = self._I_matrix_calc()

        # self.CG_loc = top_pos + np.array([0, 0, self.length/2])

        super().__init__(mass=mass, top_pos=top_pos, CG_loc = CG_loc, I_matrix=0, name = name, type = "Fin")

        K_fb = 1 + (self.rocket_diam/2) / (self.span + self.rocket_diam/2)
        a = 4*(self.span/self.rocket_diam)**2
        b = 1 + (2*self.Cm/(self.Cr + self.Ct))**2
        self.C_Na = K_fb * a/(1 + np.sqrt(b))

        self.get_2d_points()
        self.set_plot_elements(fmt = 'k:',
                               name = self.name)

    def __dimensions_calculations(self):

        s = self.span
        Cr = self.Cr
        Ct = self.Ct
        phi = self.sweep_angle # radians
        r_rocket = self.rocket_diam / 2

        a1 = 1/2 * s * s * np.tan(phi)
        a2 = (Cr - s*np.tan(phi)) * s
        a3 = s * (Ct - Cr + s*np.tan(phi)) / 2

        area = a1+a2+a3

        # tip_coords = np.array([0, r_rocket])

        centroid_1 = np.array([s * np.tan(phi) * 2/3, s/3])
        centroid_2 = np.array([s * np.tan(phi) + (Cr - s * np.tan(phi))/2, s/2])
        centroid_3 = np.array([Cr + (Ct - Cr + s * np.tan(phi))/3, 2/3 * s])

        # With respect to front point of root chord
        CG_2d = (a1*centroid_1 + a2*centroid_2 + a3*centroid_3)/(area)

        print(CG_2d)

        self.CG_2d = self.CP_2d = CG_2d


        self.area = area
        self.wind_area = 2 * area

        # self.aspect_ratio = 

        # return {"area_array": [a1, a2, a3], "area_sum": area}

        # CALCULATE MOMENTS OF INERTIA AND CP
    
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
    

    # def get_CP(self, speed, compressibility_factor):
        
    #     return self.CG_loc
    
    def get_2d_points(self):
        
        s = self.span
        Cr = self.Cr
        Ct = self.Ct
        phi = self.sweep_angle # radians

        r_rocket = self.rocket_diam / 2
        r_tail = self.aft_diam / 2
        l_bt = self.boattail_length

        tip_location = np.array([self.top_pos[2], 0])

        a = np.array([0, r_rocket]) + tip_location
        b = np.array([s*np.tan(phi), r_rocket + s]) + tip_location
        c = np.array([s*np.tan(phi) + Ct, r_rocket + s]) + tip_location
        d = np.array([Cr, r_tail]) + tip_location

        if Cr > l_bt:
            print("fins longer")
            e = np.array([Cr - l_bt, r_rocket]) + tip_location
        else:
            print("tail longer")
            e = np.array([0, (Cr - l_bt)*(r_rocket-r_tail)/l_bt + r_rocket]) + tip_location


        self.points_2d = np.array([a,b,c,d,e,a]) * np.array([1, np.cos(self.roll_orientation)])
        self.points_2d = self.points_2d.T




    
    