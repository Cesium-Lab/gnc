import numpy as np
from ..constants import *
from scipy.linalg import norm

class Atmosphere:

    def __init__(self):
        self.temp = 20.0 # C
        self.p = 100 # kPa
        self.rho = 1 # kg/m3
        self.mu = 1.813 * 10*(-5) # Pa.s
        self.nu = 1.506 * 10**(-5) # m2/s
 
    # Look into this https://www.aerodynamics4students.com/properties-of-the-atmosphere/variation-with-altitude.php
    # https://www.grc.nasa.gov/www/k-12/airplane/viscosity.html
    def update(self,alt):
        """alt --> (T, p, rho)
        
        T in degC

        P in kPa

        rho in kg/m3
        """

        if alt < 11000:
            T = 15.04 - 0.00649*alt
            p = 101.29 * pow((T + 273.15)/288.08, 5.256)
        elif alt < 25000:
            T = -56.46
            p = 22.65 * np.exp(1.73 - 0.000157*alt)
        else:
            T = -131.21 + 0.00299*alt
            p = 2.488 * pow((T + 273.15)/216.6, -11.388)

        rho = p / (0.2869 * (T + 273.15))
        
        self.temp = T
        self.p = p * 1e3
        self.rho = rho

        # Sutherland's law
        self.mu = (1.458 * 10**(-6) * (T + 273.15)**(3/2))

        return [T, p, rho]
    
class WindModel:

    def __init__(self,
                 latitude, launch_site_wind,
                 launch_rail_height = 55 * FT2M,
                 von_karman_const = 0.4,
                 turbulence_num = 0):
        """
        latitude: degrees
        launch_site_wind_speed: m/s
        
        """
        self.earth_angular_velocity = 2 * np.pi / (1 * DAY2SEC)
        self.latitude = latitude * DEG2RAD
        self.launch_site_wind_speed = launch_site_wind
        self.k = von_karman_const
        self.z0 = 0.001 * FT2M # Surface roughness for desert
        
        # print(str(self.earth_angular_velocity) + "rad/s")
        self.coriolis_f = 2 * self.earth_angular_velocity * np.sin(self.latitude)

        self.c = 0.25

        self.u_star0 = self.launch_site_wind_speed * self.k / np.log(launch_rail_height/self.z0)
        self.alt_initial = launch_rail_height
        A = 1.9
        B = 4.9
        
        # For dealing with zeros
        if norm(self.u_star0) == 0:
            self.abl_height = 0
        else:
            self.abl_height = self.c * norm(self.u_star0) / self.coriolis_f

        # For dealing with zeros
        if norm(self.u_star0) == 0:
            self.l_MBL = 0
        else:
            part1 = np.log(norm(self.u_star0) / self.coriolis_f/ self.z0) - A
            part2 = part1**2 + B**2
            part3 = np.sqrt(part2) - np.log(self.abl_height/self.z0)

            self.l_MBL = launch_rail_height/2 * part3**(-1)

        print(f"Wind speed: {self.launch_site_wind_speed}")
        print(f"l_MBL: {self.l_MBL} m")
        print(f"ABL Height: {self.abl_height} m")


        # Turbulence
        # if turbulence_num == 1:
        #     wind_20m = 15 * KNOTS2MPS
        # elif turbulence_num == 2:
        #     wind_20m = 30 * KNOTS2MPS
        # elif turbulence_num == 3:
        #     wind_20m = 45 * KNOTS2MPS

        # self.turb_variance_u = 

        
        


    def updateBase(self,alt, rho, vert_vel, prev_pressure_derivative, P_current = None, P_last = None):
        """alt, rho --> (v_wind, dPdX)
        
        """
        # return self.launch_site_wind_speed, prev_pressure_derivative
        # Adjusted for zeros
        if alt == 0:
            wind_speed = self.launch_site_wind_speed
            pressure_derivative = wind_speed * rho * self.coriolis_f
        elif alt < self.abl_height:

            

            wind_speed = self.u_star0/self.k * (np.log(alt/self.z0) + alt / self.l_MBL - alt/self.abl_height * (alt/2/self.l_MBL))

            if vert_vel >= 0:
                pressure_derivative = wind_speed * rho * self.coriolis_f
            else:
                pressure_derivative = prev_pressure_derivative

        else:

            pressure_derivative = np.zeros(2)
            for i, wind in enumerate(self.launch_site_wind_speed):

                # print(f"{wind} {i}")
                if wind == 0:
                    pressure_derivative[i] = 0
                else:
                    # Find out where this came from
                    pressure_derivative[i] = prev_pressure_derivative[i] * (P_current/(P_last) + 1)/2

            wind_speed = pressure_derivative / rho / self.coriolis_f

        self.wind_speed = wind_speed
        self.pressure_deriv = pressure_derivative
        return wind_speed, pressure_derivative
    
    # def turbulence(self, alt, v_rocket_wind, )


class Drag0AoAModel:

    def __init__(self,
                 max_body_diameter,
                 body_length,
                 body_wetted_area,
                 boattail_diam
                 ):
        self.max_body_diameter = max_body_diameter
        self.body_length = body_length
        self.body_wetted_area = body_wetted_area
        self.boattail_diam = boattail_diam

        # Base drag coefficients
        self.base_drag_Kb = 0.0274 * np.arctan(body_length/max_body_diameter + 0.0116)
        self.base_drag_n = 3.6542 * pow(body_length/max_body_diameter, -0.2733)
        print(f"Kb: {self.base_drag_Kb}\t\tn: {self.base_drag_n}")
        
        
    def skin_fric_coeff(self, length_c, k_roughness, Mach, v_sound, kinematic_viscosity = 0):

        nu = 1.81 * 10**(-5) # m2/s

        # Compressible Reynold's Number
        Rn = v_sound * Mach * length_c / 12 / nu * (1
                                                    +0.0283 * Mach
                                                    -0.043 * Mach**2
                                                    +0.2107 * Mach**3
                                                    -0.03829 * Mach**4
                                                    +0.002709 * Mach**5)
        
        # Incompressible skin friction coefficient
        Cf_star = 0.037036 * pow(Rn, -0.155079) if Rn != 0 else 0

        # Compressible skin friction coefficient
        Cf = Cf_star * (1
                        +0.00798 * Mach
                        -0.1813 * Mach**2
                        +0.0632 * Mach**3
                        -0.00933 * Mach**4
                        +0.000549 * Mach**5)
        
        # Incompressible skin friction coefficient with roughness
        a = 1.89 + 1.62 * np.log10(length_c/k_roughness)
        Cf_star_rough = pow(a, -2.5)

        # Compressible skin friction coefficient with roughness
        Cf_rough = Cf_star_rough / (1 + 0.2044 * Mach**2)

        # Final skin friction coefficient
        Cf_final = max(Cf_rough, Cf)
        
        return Cf_final
    

    def friction_drag_coeff(self, component, Cf, params_dict = {}):
        """ body params_dict = 'wetted_area', 'body_length', 'max_body_diameter'
        
        fins params_dict = 'Cr', 'Ct', 'max_thickness', 'max_thickness_distance', 'num_fins', 'fin_area', 'max_body_diameter', 'Mach', 'v_sound', 'kinematic_viscosity', 
        """
        if component == 'body':
            wetted_area = params_dict['wetted_area']
            body_length = params_dict['body_length']
            max_diam = params_dict['max_body_diameter']

            a = 1 + 60 / pow(body_length/max_diam, 3) + 0.0025 * (body_length/max_diam)
            b = 4 * wetted_area / np.pi / max_diam**2

            Cd_friction = Cf * a * b
        
        elif component == 'fins':
            Cr = params_dict['Cr']
            Ct = params_dict['Ct']
            max_t = params_dict['max_thickness']
            max_t_dist = params_dict['max_thickness_distance']
            N = params_dict['num_fins']
            fin_area = params_dict['fin_area']
            max_diam = params_dict['max_body_diameter']
            Mach = params_dict['Mach']
            v_sound = params_dict['v_sound']
            nu = params_dict['kinematic_viscosity']
            
            # Ratio of fin tip chord to root chord
            ratio = Ct/Cr
            # Incompressible reynolds number at base of fin
            Rn = v_sound * Mach * Cr / 12 / nu

            # Average flat plate skin friction coeff for each fin panel
            if ratio == 0 or Rn == 0:
                Cf_lambda = Cf * (1 +  0.5646/np.log10(Rn)) if Rn != 0 else Cf
            else:
                a = pow(np.log10(Rn), 2.6) / (ratio**2 - 1)
                b = ratio**2 / pow(np.log10(Rn * ratio), 2.6)
                c = 1 / pow(np.log10(Rn), 2.6)
                d = ratio**2 / pow(np.log10(Rn * ratio), 3.6)
                e = 1 / pow(np.log10(Rn), 3.6)

                Cf_lambda = Cf * a * (b - c + 0.5646*(d - e))

            a = 60 * (max_t/Cr)**4
            b = 0.8 * (1 + 5 * pow(max_t_dist/Cr,2)) * (max_t/Cr)
            c = 4 * N * fin_area / np.pi / max_diam**2

            Cd_friction = Cf_lambda * (1 + a + b) * c

        return Cd_friction
        
        # Protuberances :(

    def base_drag_coeff(self, mach, Cd_friction_total):
        Kb = self.base_drag_Kb
        n = self.base_drag_n

        if mach < 0.6:
            Cd_b = Kb * pow(self.boattail_diam/self.max_body_diameter, n) / np.sqrt(Cd_friction_total)

            # Saves last one in Mach6 Base drag 
            self.Cd_base_mach6 = Cd_b
        else:

            if mach < 1:
                fb = 1.0 + 215.8 * (mach - 0.6)**6
            elif mach < 2:
                fb = 2.0881 * (mach - 1)**3 - 3.7938 * (mach - 1)**2 + 1.4618 * (mach - 1) + 1.883917
            else:
                fb = 0.297 * (mach - 2)**3 - 0.7937 * (mach - 2)**2 - 0.1115 * (mach - 2) + 1.64006

            Cd_b = fb * self.Cd_base_mach6
            

        # else
        # print(self.Cd_base_mach6)
        return Cd_b


class InducedDragModel:

    def __init__(self, rocket_length, nosecone_length, diameter):
        self.rocket_length = rocket_length
        self.nosecone_length = nosecone_length
        self.diameter = diameter

        self.interp_table = {
            'aoa': [4,6,8,10,12,14,16,18,20],
            'delta': [0.78,0.82,0.92,0.94,0.96,0.97,0.975,0.98,0.985],
            'eta': [0.6,0.63,0.66,0.68,0.72,0.73,0.74,0.75,0.76]
        }

        self.delta = lambda aoa: np.interp(aoa, self.interp_table['aoa'], self.interp_table['delta'])
        self.eta = lambda aoa: np.interp(aoa, self.interp_table['aoa'], self.interp_table['eta'])

    def induced_drag(self, aoa):
        delta = self.delta(aoa)
        eta = self.eta(aoa)

        # print(aoa, end="")
        
        a = 2 * delta * aoa**2
        b = 3.6 * eta * (1.36 * self.rocket_length - 0.55 * self.nosecone_length) / (np.pi * self.diameter)
        Cd_body = a + b * aoa**3

        return Cd_body






