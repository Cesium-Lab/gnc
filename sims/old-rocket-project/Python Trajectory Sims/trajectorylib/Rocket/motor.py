import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ..constants import *


class Motor:

    def __init__(self, name = "Motor"):
        self.name = name
        
        self.loaded_thrust = False

        print(f"Creating motor with name '{name}'")
        pass
        
    def thrust_from_step_throttle(self, thrust, time, thrust_units = "lbf"):

        if thrust_units == 'lbf':
            print("Setting default thrust units to lbf")
            unit_conversion = LBF2N
        elif thrust_units == 'N':
            print("Setting thrust units to Newtons")
            unit_conversion = 1.0
        else:
            print("Setting thrust units to Newtons by default")
            unit_conversion = 1.0
        
        self.time = np.array([0.0, time[0], time[0] + 0.00001, time[1]])
        self.thrust = np.array([thrust[0], thrust[0], thrust[1], thrust[1]]) * unit_conversion
        self.tb = max(time)

   
      
        self.I_tot = np.trapz(y=self.thrust, x=self.time)

        self.loaded_thrust = True


    def thrust_from_csv(self, filepath, delimeter = ",", column_names = None, thrust_units = None):

        data = pd.read_csv(filepath, delimiter=delimeter)

        unit_conversion = 1.0

        if column_names:
            time_col = column_names[0]
            thrust_col = column_names[1]
            

        else:
            time_col = data.columns[0]
            thrust_col = data.columns[1]

        if thrust_units == 'lbf':
            print("Overriding thrust units to lbf")
            unit_conversion = LBF2N
            
        if '(lbf)' in thrust_col:
            print("Auto-detected thrust in units of lbf")
            unit_conversion = LBF2N
        
        self.thrust = data[thrust_col] * unit_conversion
        self.time = data[time_col]
        self.tb = max(self.time)

        self.I_tot = np.trapz(y=self.thrust, x=self.time)

        self.loaded_thrust = True


    def thrust_from_rse(self, filepath):

        force_arr = []
        time_arr = []
        # mass_arr = []
        with open(filepath) as rse_file:
            for line in rse_file:

                if '<eng-data' not in line:
                    continue

                line = line.split('"')

                force = float(line[3])
                # mass = float(line[5])
                time = float(line[7])

                force_arr.append(force)
                time_arr.append(time)
                # mass_arr.append(mass)

                # print(line)
                # print(f"{time}s, {force * N2LBF} LBF")
                
        
        self.thrust = np.array(force_arr)
        self.time = np.array(time_arr)

        self.I_tot = np.trapz(y=self.thrust, x=self.time)

        print(self.I_tot)

        self.tb = self.time[-1]

        self.loaded_thrust = True

    
    def __repr__(self):

        return str([self.thrust, self.time])
    

    def calc_thrust(self, time):
        thrust = np.interp(time, self.time, self.thrust, right=0)
        return thrust
    
    def plot_thrust(self, units = 'N'):

            
        t = np.linspace(0, self.tb, 1000)

        T = np.array([self.calc_thrust(i) for i in t])

        if units == 'lbf':
            T *= N2LBF
        
        plt.xlabel(f'Time (s)')
        plt.ylabel(f'Thrust ({units})')
        plt.xlim(-self.tb * 0.1, self.tb * 1.1)
        plt.ylim(0, max(T) * 1.1)
        plt.title(f"Thrust of '{self.name}'")
        plt.plot(t,T)
        plt.show()

        # self.thrust[self.step] = interpolate.interp1d(self.motor['seconds'], self.motor['thrust'], bounds_error = False, fill_value=0)(t) # N

    


