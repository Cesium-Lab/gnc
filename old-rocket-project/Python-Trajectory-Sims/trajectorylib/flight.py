import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm


from .constants import (
    DEG2RAD,
    FT2M,
    KM2M,
    M2FT,
    M2KM,
    RAD2DEG,
    EARTH,
    IN2M,
    SEC2MIN,
    SEC2HOUR,
    SEC2DAY,
)
from .math_helpers import quat_apply, quat_inv, unit, quat_mult, R_from_quat
from .Environment import Atmosphere, WindModel
from .Rocket import Rocket


# Operates in meters and NED coordinates assuming 2D plane
class Flight:
    def __init__(
        self,
        x0,
        v0,
        mass,
        tspan,
        dt,
        rocket: Rocket,
        launch_site_alt=0,
        central_body=EARTH,
    ):
        """
        tspan (seconds)
        dt (seconds)
        """

        # Vertical rocket
        q0 = np.array([1, 0, 0, 0])
        # q0 = quat_from_axis_rot(44*DEG2RAD, [1,0,0])
        # print(f"orientation0: {quat_apply(q0, [0,0,1])}")
        self.rail_length = 50 * FT2M
        omega0 = np.array([0, 0, 0]) * DEG2RAD
        # q0 = quat_from_axis_rot(45*DEG2RAD, [0,1,0])
        # omega0 = 45* DEG2RAD * np.array([0,1,0])

        self.alt = x0[2]

        self.tspan = tspan
        self.dt = dt
        self.central_body = central_body

        self.rocket = rocket

        self.tb = rocket.motor.tb

        self.atmos = Atmosphere()

        self.n_steps = int(np.ceil(self.tspan / self.dt) + 1)

        # initialize arrays
        # state = x, y, z, vx, vy, vz, qs, qx, qy, qz, omega_x, omega_y, omega_z, m
        self.state = np.zeros((self.n_steps, 14))
        self.t = np.zeros((self.n_steps))

        self.CG = np.zeros((self.n_steps, 3))
        self.CP = np.zeros((self.n_steps, 3))
        self.stability = np.zeros((self.n_steps))

        # Rocket parameters
        self.distance = 0
        self.altitude = np.zeros((self.n_steps))
        self.thrust = np.zeros((self.n_steps, 3))
        self.mass = np.zeros((self.n_steps))
        self.speed = np.zeros((self.n_steps))
        self.mach = np.zeros((self.n_steps))
        self.angle_of_attack = np.zeros((self.n_steps))
        self.rocket_axis = np.zeros((self.n_steps, 3))
        self.v_rocket_wind = np.zeros((self.n_steps, 3))
        self.stab_vector = np.zeros((self.n_steps, 3))
        self.stability_cal = np.zeros((self.n_steps))
        self.accel = np.zeros((self.n_steps, 3))
        self.alpha = np.zeros((self.n_steps, 3))

        # Forces
        self.Cds = np.zeros((self.n_steps, 4))
        self.drag = np.zeros((self.n_steps, 3))
        self.grav = np.zeros((self.n_steps))
        self.moment = np.zeros((self.n_steps, 3))

        self.mass_tuple = mass

        # Environment
        self.rho = np.zeros((self.n_steps))
        self.pressure = np.zeros((self.n_steps))
        self.temperature = np.zeros((self.n_steps))
        self.v_sound = np.zeros((self.n_steps))
        self.compressibility_factor = np.zeros((self.n_steps))
        self.dynamic_pressure = np.zeros((self.n_steps))

        self.wind = np.zeros((self.n_steps, 2))  # x and y
        self.prev_pressure_derivative = np.zeros((self.n_steps, 2))  # x and y

        # initial conditions
        self.initialize(x0, v0, q0, omega0, mass[0], launch_site_alt)

        self.solutions = []

    def initialize(self, x0, v0, q0, omega0, mass0, initial_altitude):

        self.initial_altitude = initial_altitude

        # Initializes mass,
        self.state[0, :] = (
            x0.tolist() + v0.tolist() + q0.tolist() + omega0.tolist() + [mass0]
        )
        self.altitude[0] = 0
        self.state[0, 2] += initial_altitude
        self.accel[0, :] = [0, 0, 0]
        self.alpha[0, :] = [0, 0, 0]

        # print(f"self state 0 = {self.state[0,:]}")

        self.t[0] = 0
        self.mass[0] = mass0

        self.atmos.update(initial_altitude)

        self.prev_pressure_derivative[0, :] = np.zeros(2)

        self.pressure[0] = self.atmos.p
        self.rho[0] = self.atmos.rho
        self.temperature[0] = self.atmos.temp
        self.dynamic_pressure[0] = 0

        self.v_sound[0] = np.sqrt(1.4 * 287 * (self.atmos.temp + 273.15))
        self.mach[0] = 0
        self.compressibility_factor[0] = 1
        self.grav[0] = 9.81

    def define_wind(self, latitude, wind_vector):

        # wind_vector_unit = wind_vector / norm(wind_vector)

        # wind_speed = wind_vector_unit * launch_site_wind_speed

        self.base_wind = WindModel(latitude=latitude, launch_site_wind=wind_vector)

        self.wind[0:,] = wind_vector
        self.prev_pressure_derivative[0:,] = [0, 0]
        self.base_wind.updateBase(self.initial_altitude, self.rho[0], 0, [0, 0], 1, 1)

    """"""

    def propogate(self, integrator="lsoda", rail_constraints=True):

        self.rail_constraints = rail_constraints
        self.step = 1

        # ----- LSODA -----#
        # solver = LSODA(self.general_flight, self.t[0], self.state[0], self.tspan, max_step=self.dt)

        # while (solver.status == 'running'):

        #     if solver.t > 0.5:
        #         if (self.step >= self.n_steps
        #             or self.alt <= 0
        #             or solver.t > self.tspan):
        #             break
        #     solver.step()
        #     curr_t = solver.t
        #     curr_state = solver.y

        #     self.t[self.step] = curr_t
        #     self.state[self.step] = curr_state

        #     self.step += 1

        # ----- ODE -----#
        # solver = ode(self.general_flight)
        # solver.set_initial_value(self.state[0], self.t[0])
        # solver.set_integrator('dopri5')

        # # propogate orbit
        # while (solver.successful()):

        #     if solver.t > 0.5:
        #         if (self.step >= self.n_steps
        #             or self.alt <= 0
        #             or solver.t > self.tspan):
        #             break

        #     solver.integrate(solver.t + self.dt)

        #     self.t[self.step] = solver.t
        #     self.state[self.step] = solver.y

        #     self.step += 1

        # ----- EULER -----#

        for i in range(1, self.n_steps):
            t = self.t[self.step - 1]
            if t > 2:
                if (
                    self.step >= self.n_steps
                    or self.alt <= 0
                    or self.state[self.step - 1, 5] < 0
                ):
                    print("Reached apogee. Stopping")
                    self.step -= 1
                    break

            y = np.array(self.state[self.step - 1])
            # print(y[0:6])
            dydt = np.array(self.general_flight(t, y))
            # print(dydt)
            new_y = y + dydt * self.dt
            # print(new_y)
            self.state[self.step] = new_y

            self.t[self.step] = self.t[self.step - 1] + self.dt
            # print(self.alt)
            self.step += 1

        ### end of Euler

        self.x = self.state[:, :3]
        self.v = self.state[:, 3:6]
        self.quats = self.state[:, 6:10]
        self.omega = self.state[:, 10:13]

        # R = [R_from_quat(q) for q in self.quats]
        # self.rocket_axis = np.array([np.matmul(r,[0,0,1]) for r in R])
        # self.rocket_axis = np.array([quat_apply(unit(q),[0,0,1]) for q in self.quats])

        self.pitch = (
            np.array([np.arccos(unit(v)[2]) for v in self.rocket_axis]) * RAD2DEG
        )
        # print(self.pitch)

        # self.solver = solver

    # y is State vector defined by state = [x, y, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3]
    def general_flight(self, t, state):

        # ----- unpack state -----#
        r = state[:3]
        x, y, z = r
        altitude = z - self.initial_altitude
        vx, vy, vz = v = state[3:6]
        in_recovery = True if (vz < 0 and t > self.tb) else False
        a = np.zeros(3)

        q_B2L = unit(state[6:10])  # Tracks how to get from launch to body axes
        q_L2B = quat_inv(q_B2L)

        # q_L2B = unit(state[6:10]) # Tracks how to get from launch to body axes
        # q_B2L = quat_inv(q_L2B)

        # print(f"{norm(q_B2L)}, {norm(q_L2B)}")
        omega = state[10:13]
        # haha_mass = state[13]

        rocket_axis = quat_apply(q_B2L, [0, 0, 1])

        # print((rocket_axis))

        # ----- Radius form earth -----#
        d_earth_rocket = z + self.central_body["radius"] * KM2M

        # ----- Gravity -----#
        grav = -d_earth_rocket * self.central_body["mu_meter"] / d_earth_rocket**3
        a[2] += grav

        # ----- Atmospheric conditions -----#
        self.atmos.update(z)

        T = self.atmos.temp
        rho = self.atmos.rho
        curr_pressure = self.atmos.p

        # ----- Base Wind model -----#
        lateral_wind, pressure_deriv = self.base_wind.updateBase(
            altitude,
            rho,
            vz,
            prev_pressure_derivative=self.prev_pressure_derivative[self.step - 1],
            P_current=curr_pressure,
            P_last=self.pressure[self.step - 1],
        )

        # lateral_wind = np.array([0,0])
        # pressure_deriv = np.array([0,0])
        # print(lateral_wind)
        v_wind = np.array([lateral_wind[0], lateral_wind[1], 0])

        v_rocket_wind = v - v_wind

        # print(v_rocket_wind)

        # ----- Wind Gust model -----#

        # ----- Wind Turbulence model -----#

        # ----- Local characteristics Model -----#
        v_sound = np.sqrt(1.4 * 287 * (T + 273.15))
        mach = norm(v_rocket_wind) / v_sound
        if mach <= 0.9:
            beta = np.sqrt(1 - mach**2)
        elif mach > 1.1:
            beta = np.sqrt(mach**2 - 1)
        else:
            beta = np.sqrt(1 - 0.9**2)

        # print(v_rocket_wind)
        Q = 1 / 5 * rho * norm(v_rocket_wind) ** 2

        angle_of_attack = np.arccos(np.dot(rocket_axis, unit(v_rocket_wind)))

        # ----- Center of Gravity -----#
        # ----- Thrust Model / Rocket Equation -----#

        curr_mass = self.rocket.get_mass(t)
        thrust_body = [0, 0, self.rocket.get_thrust(t)]  # N
        thrust_launch = quat_apply(q_B2L, thrust_body)
        # thrust_launch = thrust_body

        if t < self.tb:
            a += thrust_launch / curr_mass

        # ----- Stability Derivative -----#
        # ----- Center of Pressure and Gravity -----#
        # Update fins and nosecone centers of pressure
        CP = self.rocket.get_CP()
        CG = self.rocket.get_CG(t)
        # ----- 0 AoA Drag Model -----#
        # ----- Induced Drag model -----#
        # ----- Total Drag model (with new wind velocites) -----#
        if in_recovery:
            Cd = self.rocket.parachute_Cd
            Cd_friction = Cd_base = Cd_induced = 0
            CP = [0, 3.5 * IN2M, 1.2]
        else:
            CD_vals = self.rocket.get_Cd(self.rocket, mach, v_sound, angle_of_attack)
            Cd, Cd_friction, Cd_base, Cd_induced = CD_vals
            # Cd = 0.46
            # Cd *= 1.1
            # if t > 15:
            #     print(CD_vals)
        drag_accel = (
            -0.5
            * rho
            * v_rocket_wind
            * norm(v_rocket_wind)
            * Cd
            * self.rocket.A_cs
            / curr_mass
        )

        if vz < 0 and abs(drag_accel[2]) > abs(grav):
            drag_accel[2] = -grav
        a += drag_accel

        # ----- Lift model -----#

        # ----- Rail Friction -----#

        # ----- Shock Force Model -----#

        # ----- Ficticious forces -----#

        # ----- All other forces -----#

        # ----- Calculate MOI -----#
        # ----- Calculate Moments -----#

        # -(CP-CG) because the coordiante system on the rocket is negative

        r_CP_body = -quat_apply(q_B2L, CP - CG)
        stability_cal = norm(CP - CG) / self.rocket.max_body_diameter
        drag_moment = np.cross(r_CP_body, drag_accel) * curr_mass

        moment = drag_moment

        # ----- Linear Motion -----#

        # ----- Rotational Motion -----#

        R_L2B = R_from_quat(q_L2B)
        R_B2L = R_from_quat(q_B2L)
        I_matrix_body = self.rocket.get_I_matrix(t)

        # I_matrix = np.matmul(R_B2L, I_matrix_body)
        I_matrix = np.matmul(np.matmul(R_B2L, I_matrix_body), R_L2B)
        # print(np.diag(I_matrix))
        I_inverse = np.linalg.inv(I_matrix)
        # print(np.diag(I_matrix))
        # alpha_intermediate = moment - np.cross(omega, np.matmul(I_matrix, omega))

        # alpha = np.matmul(I_inverse, alpha_intermediate)
        # alpha = [0,0.1,0]

        # print(np.diag(I_inverse))
        alpha = np.matmul(I_inverse, moment)
        # print(alpha)

        # Do NOT UNITIFY Q_DOT.
        q_dot = np.zeros(4)
        # qd = q_L2B
        # q_dot[0] = 0.5 * (-omega[0] * qd[1] - omega[1] * qd[2] - omega[2] + qd[3])
        # q_dot[1] = 0.5 * (omega[0] * qd[0] + omega[2] * qd[2] - omega[1] + qd[3])
        # q_dot[2] = 0.5 * (omega[1] * qd[0] - omega[2] * qd[1] + omega[0] + qd[3])
        # q_dot[3] = 0.5 * (omega[2] * qd[0] + omega[1] * qd[1] - omega[0] + qd[2])
        # q_dot = quat_mult(omega, q_L2B) / 2.0
        q_dot = quat_mult(omega, q_B2L) / 2.0

        # ----- Dynamic Pressure -----#

        # ----- Last Minute Checks -----#

        if self.rail_constraints:
            # If not cleared rail yet, make it linear motion
            if self.alt < self.rail_length:
                self.last_rail = self.step
                a[0] = 0
                a[1] = 0
                q_dot = [0, 0, 0, 0]
                alpha = [0, 0, 0]

        # Ground condition until TW ratio > 0
        if abs(self.alt) < 0.1 and a[2] < -grav:
            a = [0, 0, 0]

        # ----- Record other variables -----#
        i = self.step

        # Rocket Parameters
        self.altitude[i] = altitude
        self.thrust[i] = thrust_launch
        self.mass[i] = curr_mass
        self.mach[i] = mach
        self.angle_of_attack[i] = angle_of_attack
        self.accel[i] = a
        self.alpha[i] = alpha

        self.v_rocket_wind[i] = v_rocket_wind
        self.stab_vector[i] = r_CP_body
        self.rocket_axis[i] = rocket_axis
        self.stability_cal[i] = stability_cal

        # Forces
        self.Cds[i] = [Cd, Cd_friction, Cd_base, Cd_induced]
        self.drag[i] = drag_accel * curr_mass
        self.grav[i] = grav

        # Moments
        self.moment[i] = moment

        # Environment
        self.temperature[i] = T
        self.pressure[i] = curr_pressure
        self.rho[i] = rho
        self.v_sound[i] = v_sound
        self.compressibility_factor[i] = beta
        self.dynamic_pressure[i] = Q

        self.wind[i] = lateral_wind
        self.prev_pressure_derivative[i] = pressure_deriv

        self.alt = altitude

        # if self.alt >= self.rail_length:
        #     print(f"Time:\t\t{t}")
        #     print(f"Alt:\t\t{self.alt * M2FT}")
        #     print(f"Drag:\t\t{drag_accel}")
        #     print(f"stab_vec:\t{r_CP_body}")
        #     # print(f"v_rocket_wind:\t{v_rocket_wind}")
        #     print(f"Moment:\t\t{moment}")
        #     print(f"Alpha:\t\t{alpha}")
        #     print()
        #     input()

        # q_dot = np.zeros(4)
        # self.solutions.append([t, *state])

        if i % 100 == 0:
            print(f"Step {i}, time {t}")

        # q_dot = np.zeros(4)
        # alpha = np.zeros(3)

        return [*v, *a, *q_dot, *alpha, 0]

    def plot_state_vector(
        self,
        title="State Vector",
        figsize=(20, 10),
        time_unit="second",
        length_unit="meter",
    ):

        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=figsize)

        fig.suptitle(title, fontsize=20)

        if time_unit == "minute":
            t = self.t * SEC2MIN
        elif time_unit == "hour":
            t = self.t * SEC2HOUR
        elif time_unit == "day":
            t = self.t * SEC2DAY
        elif time_unit == "second":
            t = self.t
        else:
            print("Unrecognized time unit")
            return

        if length_unit == "meter":
            x_arr = self.x
            v = self.v
        elif length_unit == "kilometer":
            x_arr = self.x * M2KM
            v = self.v * M2KM
        elif length_unit == "feet":
            x_arr = self.x * M2FT
            v = self.v * M2FT
        else:
            print("Unrecognized length unit")
            return

        x = x_arr[:, 0]
        y = x_arr[:, 1]
        z = x_arr[:, 2]

        vx = v[:, 0]
        vy = v[:, 1]
        vz = v[:, 2]

        t = t[: self.step]

        axs[0, 0].plot(t, x[: self.step])
        axs[0, 0].set_title("X Position vs. Time")
        axs[0, 0].grid("True")
        axs[0, 0].set_ylabel("X (km)")
        axs[0, 0].set_xlabel(f"time ({time_unit})")

        axs[0, 1].plot(t, y[: self.step])
        axs[0, 1].set_title("Y Position vs. Time")
        axs[0, 1].grid("True")
        axs[0, 1].set_ylabel("Y (km)")
        axs[0, 1].set_xlabel(f"time ({time_unit})")

        axs[0, 2].plot(t, z[: self.step])
        axs[0, 2].set_title("Z Position vs. Time")
        axs[0, 2].grid("True")
        axs[0, 2].set_ylabel("Z (km)")
        axs[0, 2].set_xlabel(f"time ({time_unit})")

        axs[1, 0].plot(t, vx[: self.step])
        axs[1, 0].set_title("X Velocity vs. Time")
        axs[1, 0].grid("True")
        axs[1, 0].set_ylabel("U (km/s)")
        axs[1, 0].set_xlabel(f"time ({time_unit})")

        axs[1, 1].plot(t, vy[: self.step])
        axs[1, 1].set_title("Y Velocity vs. Time")
        axs[1, 1].grid("True")
        axs[1, 1].set_ylabel("V (km/s)")
        axs[1, 1].set_xlabel(f"time ({time_unit})")

        axs[1, 2].plot(t, vz[: self.step])
        axs[1, 2].set_title("Z Velocity vs. Time")
        axs[1, 2].grid("True")
        axs[1, 2].set_ylabel("W (km/s)")
        axs[1, 2].set_xlabel(f"time ({time_unit})")

        fig.show()

    def plot_3d(
        self,
        title="Trajectory",
        figsize=(20, 10),
        time_unit="second",
        length_unit="meter",
    ):

        fig = plt.figure(figsize=figsize)

        fig.suptitle(title, fontsize=20)

        if length_unit == "meter":
            x_arr = self.x
            alt0 = self.initial_altitude
        elif length_unit == "kilometer":
            x_arr = self.x * M2KM
            alt0 = self.initial_altitude * M2KM
        elif length_unit == "feet":
            x_arr = self.x * M2FT
            alt0 = self.initial_altitude * M2FT
        else:
            print("Unrecognized length unit")
            return

        x = x_arr[:, 0]
        y = x_arr[:, 1]
        z = x_arr[:, 2] - alt0
        ax = fig.add_subplot(111, projection="3d")
        # ax.set_aspect('equal', adjustable='box')
        # ax.axis('square')

        # ax.set_zlim(self.initial_altitude, max(x_arr[:,2]))

        ax.set_xlim(-10000, 10000)
        ax.set_ylim(-10000, 10000)
        ax.set_zlim(0, 27000)

        ax.plot(x[: self.step], y[: self.step], z[: self.step])

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        fig.show()

    def plot_vectors(
        self, step, title="Trajectory", figsize=(20, 10), length_unit="meter"
    ):

        fig = plt.figure(figsize=figsize)

        fig.suptitle(title, fontsize=20)

        # q_B2L = self.quats[step]

        if length_unit == "meter":
            x_arr = self.x
            alt0 = self.initial_altitude
        elif length_unit == "kilometer":
            x_arr = self.x * M2KM
            alt0 = self.initial_altitude * M2KM
        elif length_unit == "feet":
            x_arr = self.x * M2FT
            alt0 = self.initial_altitude * M2FT
        else:
            print("Unrecognized length unit")
            return

        # head = x_arr[step]

        x = x_arr[:, 0]
        y = x_arr[:, 1]
        z = x_arr[:, 2] - alt0
        ax = fig.add_subplot(111, projection="3d")
        # ax.set_aspect('equal', adjustable='box')
        # ax.axis('square')

        ax.set_zlim(self.initial_altitude, max(x_arr[:, 2]))

        # ax.set_xlim(-50,50)
        # ax.set_ylim(-50,50)
        # ax.set_zlim(0,100)

        ax.plot(x[: self.step], y[: self.step], z[: self.step])

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        fig.show()
