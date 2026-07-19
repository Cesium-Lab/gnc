from .component import *
from .nosecone import *
from .fin import *
from .bodytube import *
from .boattail import *
from .motor import *

from ..Environment.environment import Drag0AoAModel, InducedDragModel

from copy import copy

class Rocket:

    def __init__(self,
                 motor: Motor,
                 diameter,
                 name = "Rocket",
                 mass = (0,0),
                 ):

        # self.components = []

        # if components_list:
        #     for piece in components_list:
        #         self.components.append(piece)


        self.A_cs = np.pi * (diameter/2)**2

        # self.Cd = 0.46


        self.mass_wet = mass[0]
        self.mass_dry = mass[1]

        self.name = name

        self.motor = motor
        self.tb = motor.tb

        self.components_list = []
        self.airframe_list = []
        self.fluids = []

        self.bodytubes = []
        self.fins = []
        self.nosecone = None
        self.boattail = None

        self.length = 0
        
        self.initial_values()

        self.parachute_Cd = 20

    def init_drag(self, k_roughness = 0.00025, nu = 1.81 * 10**(-5)):

        self.wetted_area = 0
        self.max_body_diameter = 0

        # for fin in self.fins:
        #     self.wetted_area += 2 * fin.wind_area

        for tube in self.bodytubes:
            self.wetted_area += tube.wind_area
            self.max_body_diameter = max(self.max_body_diameter, tube.D_o)

        self.wetted_area += self.nosecone.wind_area


        self.drag_0AoA = Drag0AoAModel(self.max_body_diameter,
                                self.length,
                                self.wetted_area,
                                self.boattail.D_aft)
        
        self.induced_drag = InducedDragModel(self.length,
                                             self.nosecone.length,
                                             self.max_body_diameter)
        
        self.drag_k_roughness = k_roughness
        self.default_nu = nu
        
    def calc_drag(self, rocket, mach, v_sound, aoa):
        skin_fric_body = self.drag_0AoA.skin_fric_coeff(self.length, self.drag_k_roughness, mach, v_sound, self.default_nu)
        skin_fric_fins = self.drag_0AoA.skin_fric_coeff(self.fins[0].Cr, self.drag_k_roughness, mach, v_sound, self.default_nu)

        params = {
            'wetted_area': self.wetted_area,
            'body_length': self.length,
            'max_body_diameter': self.max_body_diameter
        }

        Cf_body = self.drag_0AoA.friction_drag_coeff('body', skin_fric_body, params)

        params = {
            'Cr': self.fins[0].Cr,
            'Ct': self.fins[0].Ct,
            'max_thickness': self.fins[0].thickness,
            'max_thickness_distance': self.fins[0].Cr / 2, # approximation
            'num_fins': len(rocket.fins),
            'fin_area': self.fins[0].wind_area,
            'Mach': mach,
            'v_sound': v_sound,
            'kinematic_viscosity': self.default_nu,
            'body_length': self.length,
            'max_body_diameter': self.max_body_diameter
        }
        Cf_fins = self.drag_0AoA.friction_drag_coeff('fins', skin_fric_fins, params)

        Cd_friction = 1.04*Cf_fins + Cf_body

        Cd_base = self.drag_0AoA.base_drag_coeff(mach, Cd_friction)

        

        Cd_induced = self.induced_drag.induced_drag(aoa)

        Cd_total = Cd_friction + Cd_base + Cd_induced

        return Cd_total, Cd_friction, Cd_base, Cd_induced 


    

    def get_mass(self, t):
        if t < self.tb:
            return -(self.mass_wet - self.mass_dry) / self.tb * t + self.mass_wet
        else:
            return self.mass_dry
    
    def get_thrust(self, t):
        return self.motor.calc_thrust(t)
    

    
    def plot_thrust(self):
        self.motor.plot_thrust()

    def plot_mass(self, units = 'kg'):

        t = np.linspace(0, self.tb * 1.5, 1000)

        mass = np.array([self.get_mass(i) for i in t])

        if units == 'lbm':
            mass *= KG2LBM
        

        plt.xlabel(f'Time (s)')
        plt.ylabel(f'Mass ({units})')
        plt.xlim(-self.tb * 0.1, self.tb * 1.1)
        plt.ylim(0, max(mass) * 1.1)
        plt.title(f"Mass of '{self.name}'")
        plt.plot(t,mass)
        plt.show()

    def add_nosecone(self, nosecone: Nosecone):

        self.nosecone = nosecone
        self.length += nosecone.length
        self.components_list.append(nosecone)
        self.airframe_list.append(nosecone)

    def add_bodytube(self, bodytube: Bodytube):

        self.bodytubes.append(bodytube)
        self.length += bodytube.length
        self.components_list.append(bodytube)
        self.airframe_list.append(bodytube)

    def add_boattail(self, boattail: Boattail):

        self.boattail = boattail
        self.length += boattail.length
        self.components_list.append(boattail)
        self.airframe_list.append(boattail)

    def add_fin(self, fin: Fin):

        self.fins.append(fin)
        self.components_list.append(fin)
        self.airframe_list.append(fin)
    
    def add_airframe_component(self, component: Component):

        self.components_list.append(component)
        self.airframe_list.append(component)
    
    def add_fluid(self, component: Component):

        self.components_list.append(component)
        self.fluids.append(component)
        

    def get_Cd(self, rocket, mach, v_sound, angle_of_attack):
        return self.calc_drag(rocket, mach, v_sound, angle_of_attack)
    
    def print_components(self):
        print(f"{self.name}'s Components:")
            
        for component in self.components_list:
            print(f"--> {component.name}")
        
    def plot_components(self, cg = False):

        plt.figure(figsize=(20,10))

        for component in self.components_list:
            plt.plot(*component.points_2d, component.plot_fmt)
            if cg:
                component.plot_CG()

        plt.xlim(-self.length * 0.1, self.length * 1.1)
        # plt.ylim(-0.75, 0.75)
        plt.axis('equal')
        # plt.grid(which='both')

    def plot_airframe_components(self, cg=True):

        plt.figure(figsize=(20,10))

        for component in self.airframe_list:
            plt.plot(*component.points_2d, component.plot_fmt)
            if cg:
                component.plot_CG()

        plt.xlim(-self.length * 0.1, self.length * 1.1)
        plt.ylim(-0.75, 0.75)
        # plt.grid(which='both')


    def initial_values(self):

        # Ixx_dry = self.Ixx_dry = 77.746 # kg m2
        # Iyy_dry = self.Iyy_dry  = 77.746 # kg m2
        # Izz_dry = self.Izz_dry  = 0.249 # kg m2

        Ixx_dry = self.Ixx_dry = 160.7 # kg m2
        Iyy_dry = self.Iyy_dry  = 160.7 # kg m2
        Izz_dry = self.Izz_dry  = 1.37 # kg m2

        self.I_matrix_dry = np.diag(np.array([Ixx_dry, Iyy_dry, Izz_dry]))
        self.CG_loc_dry = np.array([0,0,2.5425])
        # self.body_length = 46 * IN2M + 2 # m


    def get_mass(self, t):

        if len(self.fluids) > 0:
            fluid_mass = 0

            for f in self.fluids:
                fluid_mass += f.get_mass(t)

            return self.mass_dry + fluid_mass
        
        else: # Linear from wet mass to dry mass

            if t < self.tb:
                return - (self.mass_wet - self.mass_dry) / self.tb * t + self.mass_wet
            else:
                return self.mass_dry
    
    def get_CG(self, t, updated = False):

        numerator = (self.CG_loc_dry * self.mass_dry)
        denomimator = self.mass_dry

        for f in self.fluids:
            if not updated:
                f.update(t)
            numerator += f.mass * f.CG_loc
            denomimator += f.mass

        self.CG_loc = numerator/denomimator
        return self.CG_loc
    
    def get_CP(self, compressibility = 1):

        numerator = 0
        denominator = 0

        for component in self.airframe_list:
            numerator += component.CP_loc * component.C_Na
            denominator += component.C_Na

        CP = np.array(numerator/denominator)

        # print(CP)
        return CP
        
        CP = 3.566227194362575
        return np.array([0,0,CP])
    
    def get_I_matrix(self, t, updated = False):

        if not updated:
            
            rkt_CG = self.get_CG(t)
        else:
            rkt_CG = self.CG_loc


        I_tot = np.zeros((3,3))

        for idx, f in enumerate(self.fluids):
            if not updated:
                f.update(t)
            cg = f.CG_loc
            I = f.I_matrix
            r = rkt_CG - cg

            # https://en.wikipedia.org/wiki/Parallel_axis_theorem
            I_rel = I + f.mass * (np.inner(r,r) * np.identity(3) - np.outer(r,r))

            I_tot += I_rel
            # return(r)
        
        self.I_matrix = I_tot + self.I_matrix_dry
        return self.I_matrix
    
        







    # def add_component(self, component)