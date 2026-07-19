import numpy as np
import matplotlib.pyplot as plt

plt.style.use('dark_background')

from scipy.linalg import norm

import trajectorylib as TL
from trajectorylib.constants import *

from time import time

from pprint import pprint


cb = EARTH

def plot_3(t, thing, title=None, factor = 1, fmt = ''):
    f = plt.figure()
    plt.plot(t[:flight.step], thing[:flight.step,0] * factor, fmt, label = "x rotation")
    plt.plot(t[:flight.step], thing[:flight.step,1] * factor, fmt, label = "y rotation")
    plt.plot(t[:flight.step], thing[:flight.step,2] * factor, fmt, label = "z rotation")
    # plt.xlim(0,100)
    # plt.ylim(0,100)
    # plt.zlim(0,100)
    if title:
        plt.title(title)
    plt.grid(True)
    plt.legend()
    f.show()

def plot_2(t, thing, title=None, factor = 1):
    f = plt.figure()
    plt.plot(t[:flight.step], thing[:flight.step] * factor, label = "x")
    plt.plot(t[:flight.step], thing[:flight.step] * factor, label = "y")
    if title:
        plt.title(title)
    plt.grid(True)
    # plt.legend()
    f.show()

def plot_1(t, thing, title=None, factor = 1):
    f = plt.figure()
    plt.plot(t[:flight.step], thing[:flight.step] * factor)
    if title:
        plt.title(title)
    plt.grid(True)
    # plt.legend()
    f.show()


if __name__ == '__main__':


    start = time()

    #----- Initial Conditions -----#

    r0 = np.zeros(3)
    v0 = np.zeros(3)
    # r0 = np.array([0, 0, 50]) * FT2M
    # v0 = np.array([0, 0, 30]) * FT2M

    latitude = 35.34706755287037 
    wind_speed = np.array([-27.1187, -11.1760])* MILE2FT * FT2M / HOUR2SEC

    # wind_speed = np.array([-18.49,-7.62])* MILE2FT * FT2M / HOUR2SEC # 20 mph
    # wind_speed = np.array([-13.87, -5.72])* MILE2FT * FT2M / HOUR2SEC # 15 mph
    # wind_speed = np.array([-9.2,-3.8])* MILE2FT * FT2M / HOUR2SEC # 10 mph
    
    # wind_speed = np.array([0,0]) * MILE2FT * FT2M / HOUR2SEC
    
    print(norm(wind_speed) / (MILE2FT * FT2M / HOUR2SEC))

    FAR_altitude = 2000 * FT2M

    tspan=40
    # tspan = 5 * MIN2SEC 
    dt = 0.01

    # mass = np.array([150, 88]) * LBM2KG
    mass = np.array([73.731, 4.60E+01])
    # mass = np.array([161.731, 4.60E+01])

    #----- Motor for the rocket -----#
    motor = TL.Motor()
    motor.thrust_from_csv('test_motor.csv')
    # motor.thrust_from_step_throttle([900,600], [10, 15])
    print(f"Total impulse: {motor.I_tot * N2LBF} lbfs")




    #----- Fluids inside the rocket -----#
    density_ox = 1141 # kg/m3
    density_fuel = 789 #kg/m3
    of_ratio = 1.4
    m_ox = (mass[0] - mass[1]) * (of_ratio / (of_ratio + 1))
    m_f = (mass[0] - mass[1]) * (1 / (of_ratio + 1))

    tank_diameter = 6 * IN2M

    top_pos_ox = np.array([0,0,2]) #m
    top_pos_fuel = np.array([0,0,3]) #m
    oxygen = TL.Fluid(top_pos=top_pos_ox, burn_time=motor.tb, density=density_ox, wet_mass=m_ox, diameter=tank_diameter, name = "Oxygen")
    fuel = TL.Fluid(top_pos=top_pos_fuel, burn_time=motor.tb, density=density_fuel, wet_mass=m_f, diameter=tank_diameter, name = "Fuel")


    #----- General rocket parameters -----#
    rocket_diameter = 7 * IN2M
    
    #----- Nosecone Class -----#
    nosecone_length = 35 * IN2M
    nosecone_diameter = rocket_diameter
    nosecone_mass = 3 * LBM2KG # lazy way out
    nosecone_thickness = 0.035 * IN2M

    nosecone = TL.Nosecone(length = nosecone_length, shape = "ogive", D_o = nosecone_diameter, thickness = nosecone_thickness, name = "Nosecone")

    #----- Bodytube Classes -----#
    bt1_position = [0, 0, nosecone_length] #m
    bt1_length = 1.5 # m
    bt1_density = 1750 #kg/m3 of carbon fiber
    bt1_inner_diam = rocket_diameter
    bt1_outer_diam = 7.192  * IN2M # inches
    bt1 = TL.Bodytube(bt1_position,bt1_length, bt1_density, D_i = bt1_inner_diam, D_o=bt1_outer_diam, name="Bodytube 1")

    bt2_position = [0, 0, nosecone_length + bt1_length] #m
    bt2_length = 2 # m
    bt2_density = 1750 #kg/m3 of carbon fiber
    bt2_inner_diam = rocket_diameter
    bt2_outer_diam = 7.192  * IN2M # inches
    bt2 = TL.Bodytube(bt2_position,bt2_length, bt2_density, D_i = bt2_inner_diam, D_o=bt2_outer_diam, name="Bodytube 2")

    #----- Boattail Classes -----#
    tail_position = [0, 0, nosecone_length + bt1_length + bt2_length] #m
    tail_density = 1750 #kg/m3 of carbon fiber
    tail_length = 11 * IN2M #m
    tail_rocket_diam = rocket_diameter
    tail_inner_diam = 4.25 * IN2M # m
    tail_thickness = 0.06 * IN2M # m
    tail = TL.Boattail(top_pos = tail_position, length = tail_length, density = tail_density, thickness = tail_thickness, D_front = tail_rocket_diam, D_aft = tail_inner_diam, name = "Boattail")


    #----- Fin Classes -----#
    fin_Ct = 4 * IN2M
    fin_Cr = 14 * IN2M
    fin_span = 7.3 * IN2M
    fin_sweep_angle = 62.4611766504 # degrees
    fin_position = [0, 0, nosecone_length + bt1_length + bt2_length + tail_length - fin_Cr] #m
    fin_thickness = 0.42 * IN2M
    fin_density = 1750 #kg/m3 of carbon fiber

    fin1 = TL.Fin(top_pos = fin_position,
                span = fin_span,
                thickness=fin_thickness,
                Ct=fin_Ct,
                Cr=fin_Cr,
                roll_orientation=0,
                sweep_angle=fin_sweep_angle,
                boattail_length=tail.length,
                rocket_diam=tail.D_front,
                aft_diam=tail.D_aft,
                name="Fin1",
                density=fin_density)

    fin2 = TL.Fin(top_pos = fin_position,
                span = fin_span,
                thickness=fin_thickness,
                Ct=fin_Ct,
                Cr=fin_Cr,
                roll_orientation=120,
                sweep_angle=fin_sweep_angle,
                boattail_length=tail.length,
                rocket_diam=tail.D_front,
                aft_diam=tail.D_aft,
                name="Fin2",
                density=fin_density)

    fin3 = TL.Fin(top_pos = fin_position,
                span = fin_span,
                thickness=fin_thickness,
                Ct=fin_Ct,
                Cr=fin_Cr,
                roll_orientation=240,
                sweep_angle=fin_sweep_angle,
                boattail_length=tail.length,
                rocket_diam=tail.D_front,
                aft_diam=tail.D_aft,
                name="Fin3",
                density=fin_density)
    
    #----- Rocket Class -----#
    rocket = TL.Rocket(name = "rocket", diameter=rocket_diameter, mass = mass, motor=motor)
    rocket.add_nosecone(nosecone)
    rocket.add_bodytube(bt1)
    rocket.add_bodytube(bt2)
    rocket.add_boattail(tail)
    rocket.add_fin(fin1)
    rocket.add_fin(fin2)
    rocket.add_fin(fin3)

    rocket.add_fluid(oxygen)
    rocket.add_fluid(fuel)

    rocket.init_drag()

    #----- Rocket Printing -----#

    # rocket.print_components()
    # CG = rocket.get_CG(0)
    # CP = rocket.get_CP(0)


    # plt.style.use('default')
    # rocket.plot_components()
    # plt.plot(CP[2],0, 'ro')
    # plt.plot(CG[2], 0, 'bo')
    # plt.plot(rocket.CG_loc_dry[2], 0, 'bx')
    # plt.style.use('dark_background')
    # plt.show()
    # input()

    
    

    #----- Flight! -----#
    flight = TL.Flight(r0, v0, mass, tspan, dt, rocket = rocket, launch_site_alt = FAR_altitude)  
    flight.define_wind(latitude, wind_speed)
    flight.propogate(rail_constraints = True)




    #----- Post Flight Analysis -----#
    max_alt = (max(flight.x[:, 2]) - flight.initial_altitude)* M2FT
    max_speed = max(flight.v[:, 2]) #* M2FT
    flight_time = flight.t[flight.step - 1]

    print(f"Max Alt: {max_alt} ft")
    print(f"Max Speed: {max_speed} m/s")
    print(f"Flight time: {flight_time} s")

    print(f"{time() - start}s, {max_alt} ft, {max_speed} m/s, {flight_time}s, ")


    flight.plot_state_vector(length_unit='meter')
    # flight.plot_state_vector(length_unit='feet')

    flight.plot_3d(length_unit='feet')

    # plot_3(flight.t, flight.omega, "omega")

    plot_3(flight.t, flight.stab_vector, "Stability vector")

    plot_3(flight.t, flight.rocket_axis, "rocket_axis")
    # print(flight.rocket_axis[0])

    plot_3(flight.t, flight.alpha, "Alpha")

    # plot_3(flight.t, flight.thrust, "thrust")

    plot_3(flight.t, flight.moment, "moment")

    # unit_moment = np.array([TL.unit(moment) for moment in flight.moment])
    # plot_3(flight.t, unit_moment, "moment")

    plot_3(flight.t, flight.accel, "accel")


    plot_3(flight.t, flight.drag, "drag")

    # plot_3(flight.t, flight.v_rocket_wind, "v roket wind")

    # plot_3(flight.altitude, flight.v_rocket_wind, "wind vs alt")

    # plot_2(flight.t, flight.wind, "Wind", factor = 1/(MILE2FT * FT2M / HOUR2SEC))

    # plot_2(flight.t, flight.prev_pressure_derivative, "prev pressure deriv", factor = 1/(MILE2FT * FT2M / HOUR2SEC))

    plot_1(flight.t, flight.angle_of_attack * RAD2DEG, "Angle of Attack")
    # plot_1(flight.t, flight.altitude, "Alt", M2FT)

    # plot_1(flight.t, flight.Cds[:,0], "CDs")
    # plot_1(flight.t, flight.Cds[:,1], "CDs friction")
    # plot_1(flight.t, flight.Cds[:,2], "CDs base")
    # plot_1(flight.t, flight.stability, "stability")

    plot_1(flight.t, flight.pitch, "pitch")
    # print(flight.stab_vector[0:flight.step])


    plt.show()
    # plt.plot(flight.t[:flight.step],flight.omega[:flight.step,0] * RAD2DEG, label = "x rotation")
    # plt.plot(flight.t[:flight.step],flight.omega[:flight.step,1] * RAD2DEG, label = "y rotation")
    # plt.plot(flight.t[:flight.step],flight.omega[:flight.step,2] * RAD2DEG, label = "z rotation")
    # plt.title("omega")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # plt.plot(flight.t[:flight.step],flight.angle_of_attack[:flight.step] * RAD2DEG, label = "Angle of Attack")
    # plt.title("Angle of Attack")
    # plt.grid(True)
    # plt.show()


#     plt.plot(rocket.t[:rocket.step],Mach)
#     plt.title("Mach")
#     plt.grid(True)
#     plt.show()

    # for i in flight.solutions:
    #     print(i)

    # plt.plot(flight.t[:flight.step])
    # for i in range(flight.step):
    #     print(flight.quats[i,0])
    # plt.plot(flight.t[:flight.step], flight.quats[:flight.step,1])
    # plt.show()

    # print(flight.steps)

    # l = len(flight.steps)

    # ha = []
    # for i in range (l-1):

    #     ha.append(flight.steps[i+1] - flight.steps[i])

    # print(min(ha))
        


# """
# method, program_time, alt (ft), max_speed (ft/s), flight_time (s), errors
# vode, 0.49068737030029297, 354026.2368589088, 4350.558364131156, 306.1399999998666, 0
# zvode, 0.5875959396362305, 354000.0968878023, 4350.376362899011, 306.1299999998666, 1
# lsoda, 0.45406007766723633, 354074.8075485479, 4351.035623047526, 306.16999999986655, 0
# dopri5, 1.959472894668579, 354094.7113075338, 4351.044155901694, 306.16999999986655, 0
# dop853, 3.4007182121276855, 354086.0270183614, 4351.044155901694, 306.16999999986655, 0

# """
    

#     # print(f'Max altitude: {max(rocket.x[:, 2]) * M2FT} ft')
#     # print(f'Max vertical speed: {max(rocket.v[:, 2]) * M2FT} ft/s')
#     # print(f'Flight time: {rocket.t[rocket.step - 1]} s')
#     # print()


