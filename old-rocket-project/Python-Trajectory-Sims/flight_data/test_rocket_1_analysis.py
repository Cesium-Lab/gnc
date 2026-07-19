import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# from mpl_toolkits.basemap import Basemap
# import geopandas as gpd
# from shapely.geometry import Point, Polygon
# from scipy.signal import butter, filtfilt # For lowpass filters
# from gmplot import GoogleMapPlotter # For
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trajectorylib.quaternion_helpers import (
    quat_from_axis_rot,
    quat_apply,
    quat_mult,
    unit,
)  # For my quaternions
from trajectorylib.constants import DEG2RAD, RAD2DEG, M2FT
from scipy.linalg import norm

GRAVITY = 9.80665

############################## Import data ##############################

flight_data = pd.read_excel("nosecone data test rocket 1.xlsx")
# img = plt.imread('image.jpg')
# print(flight_data.columns)

TIME_MS_COL = "log time (ms)"
BMP_ALT_COL = "bmp altitude (m)"
MS_ALT_COL = "ms altitude (m)"
ICM_ACCEL_X_COL = "icm accel x"
ICM_ACCEL_Y_COL = "icm accel y"
ICM_ACCEL_Z_COL = "icm accel z"
ICM_GYRO_X_COL = "gyro x"
ICM_GYRO_Y_COL = "gyro y"
ICM_GYRO_Z_COL = "gyro z"
GPS_LAT_COL = "gps lat"
GPS_LON_COL = "gps long"
GPS_ALT_COL = "gps altitude"

############################## Read data ##############################

# Data
imu_accel_data = np.array(
    flight_data[[ICM_ACCEL_X_COL, ICM_ACCEL_Y_COL, ICM_ACCEL_Z_COL]]
)
imu_gyro_data = np.array(
    flight_data[[ICM_GYRO_X_COL, ICM_GYRO_Y_COL, ICM_GYRO_Z_COL]]
)  # * DEG2RAD
time_data_s = np.array(flight_data[TIME_MS_COL]) / 1000000
time_data_s = time_data_s - time_data_s[0]

# Time fix for microcontroller reset (anomaly)
time_offset = (
    time_data_s[491] - time_data_s[490] + (time_data_s[490] - time_data_s[489])
)
print(time_offset)
for i in range(491, len(time_data_s)):
    time_data_s[i] -= time_offset


bmp_data = np.array(flight_data[BMP_ALT_COL])
bmp_data -= bmp_data[0]
ms_data = np.array(flight_data[MS_ALT_COL])
ms_data -= ms_data[0]

# GPS data
gps_data = np.array(flight_data[[GPS_LON_COL, GPS_LAT_COL, GPS_ALT_COL]])
gps_data -= gps_data[0, :]
# gps_data[:,0] *= 111111 # Longitude Deg to m (approx)
# gps_data[:,1] *= 111111 # Latitude Deg to m (approx)

# pprint(gps_data)

############################## Some processing  ##############################


DATA_LENGTH = int(np.mean([np.argmax(bmp_data), np.argmax(ms_data)])) + 100
# DATA_LENGTH = len(flight_data[TIME_MS_COL])
print(f"Length of data: {DATA_LENGTH} data points")

IMU2BODY = quat_from_axis_rot(90 * DEG2RAD, [0, 1, 0])  # assumes perfect alignment

# Sensor frame to body frame
a_body = np.apply_along_axis(lambda x: quat_apply(IMU2BODY, x), 1, imu_accel_data)
w_body = np.apply_along_axis(lambda x: quat_apply(IMU2BODY, x), 1, imu_gyro_data)

# North is +X, West is +Y
# global_xy =

##############################  Pre-initialization  ##############################

step = 1
NEW_LENGTH = int(DATA_LENGTH / step)
# Arrays
q_launch = np.zeros((NEW_LENGTH, 4))
q_dot_launch = np.zeros((NEW_LENGTH, 4))

a_launch = np.zeros((NEW_LENGTH, 3))
w_launch = np.zeros((NEW_LENGTH, 3))

x_launch = np.zeros((NEW_LENGTH, 3))
v_launch = np.zeros((NEW_LENGTH, 3))
rocket_axis = np.zeros((NEW_LENGTH, 3))

initial_angle = 50 * DEG2RAD
initial_rotation = unit(quat_from_axis_rot(initial_angle, [0, 0, 1]))
# Initial values
q_launch[0] = initial_rotation  # No Z rotation
q_dot_launch[0] = np.array([0, 0, 0, 0])
rocket_axis[0] = quat_apply(q_launch[0], np.array([0, 0, 1]))

############################## Main loop  ##############################
print(DATA_LENGTH)

for i in range(step, DATA_LENGTH, step):
    launch_i = int(i / step)

    if launch_i >= NEW_LENGTH:
        break
    # print(launch_i-1)
    print(i)
    print(launch_i)
    dt = time_data_s[i] - time_data_s[i - step]
    # print(time_data_s[i])
    q_B2L = unit(q_launch[launch_i - 1])
    # print(q_launch[i-1])
    # q_B2L = np.array([1,0,0,0])

    # print(quat_apply(q_B2L, a_body[i]))

    # Putting into launch frame with trapezoidal integration
    a_launch[launch_i] = quat_apply(q_B2L, (a_body[i] + a_body[i - 1]) / 2) - np.array(
        [0, 0, GRAVITY]
    )
    # print(q_B2L)
    # print(a_launch[i])
    w_launch[launch_i] = quat_apply(q_B2L, (w_body[i] + w_body[i - 1]) / 2)

    # Newton
    v_launch[launch_i] = a_launch[launch_i] * dt + v_launch[launch_i - 1]
    x_launch[launch_i] = (
        v_launch[launch_i] * dt
        + 1 / 2 * a_launch[launch_i] * dt * dt
        + x_launch[launch_i - 1]
    )
    q_dot_launch[launch_i] = (
        quat_mult((w_launch[launch_i] + w_launch[launch_i - 1]) / 2, q_B2L) / 2.0
    )

    q_launch[launch_i] = q_dot_launch[launch_i] * dt + q_launch[launch_i - 1]

    rocket_axis[launch_i] = quat_apply(q_launch[launch_i], np.array([0, 0, 1]))


############################## Important values  ##############################

print(f"Max altitude: {max(x_launch[:DATA_LENGTH, 2] * M2FT)} ft")
print(f"Max vertical velocity: {max(v_launch[:DATA_LENGTH, 2] * M2FT)} ft/s")
print(f"Max vertical acceleration: {max(a_launch[: DATA_LENGTH - 150, 2] * M2FT)} ft/s")

apogee_index = np.argmax(x_launch[:DATA_LENGTH, 2])
print(f"Speed at apogee: {norm(v_launch[apogee_index, :]) * M2FT} ft/s")
print(f"Horizontal speed at apogee: {norm(v_launch[apogee_index, 0:2]) * M2FT} ft/s")
print()

############################## Position Graph ##############################

pitch = np.array([np.arccos(unit(v)[2]) for v in rocket_axis]) * RAD2DEG

# f = plt.figure(0)
plt.plot(time_data_s[: DATA_LENGTH - step + 1 : step], x_launch[:NEW_LENGTH, :] * M2FT)
plt.plot(time_data_s[:DATA_LENGTH], bmp_data[:DATA_LENGTH] * M2FT)
plt.plot(time_data_s[:DATA_LENGTH], ms_data[:DATA_LENGTH] * M2FT)
# plt.plot(time_data_s[:DATA_LENGTH], ms_data[:DATA_LENGTH] * M2FT)
# plt.plot(time_data_s[:NEW_LENGTH], gps_data[:NEW_LENGTH,2] * M2FT)
plt.title("Rocket Position (Launch Frame)")
plt.legend(["X", "Y", "Z", "BMP Z", "MS Z"])
plt.xlabel("Time (s)")
plt.ylabel("Distance (ft)")
plt.show()

############################## GPS Graph ##############################

x_launch_gps_coords = x_launch

# Convert loosely to degrees
x_launch_gps_coords[:, 0] /= 111111
x_launch_gps_coords[:, 1] /= 111111

# Add starting coordinates
x_launch_gps_coords[:, 0] -= gps_data[0, 0]
x_launch_gps_coords[:, 1] -= gps_data[0, 1]

# # map = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180, resolution='c')
# geometry = [Point(xy) for xy in zip(x_launch_gps_coords[:,0], x_launch_gps_coords[:,0])]
# geo_df = gpd.GeoDataFrame(df, #specify our data
#                           crs=crs #specify our coordinate reference system
#                           geometry=geometry) #specify the geometry list we created
# geo_df.head()
# gmap = GoogleMapPlotter(35.34619896926076, -117.80670287419308, 2)
# gmap.scatter( gps_data[:,0], gpdata[:,1], '# FF0000', size = 40, marker = False )
# gmap.draw(f"{os.path.dirname(__file__)}/out.html")
# Actually make this in lat/lon to put onto an actual map
# fig, ax = plt.subplots()
# ax.imshow(img, aspect='auto', extent=[0, 1, 0, 1])


# ax.plot(x_launch[:NEW_LENGTH,0], x_launch[:NEW_LENGTH,1] * M2FT, zorder=2)
# ax.scatter(gps_data[:,0], gps_data[:,1] * M2FT, color='r', marker='.', zorder=1)
# plt.imshow(img, zorder=0, extent=[-1500 + 156, 3000 + 156, -1500 - 478, 3000 - 478])
# # plt.plot(time_data_s[:NEW_LENGTH], ms_data[:NEW_LENGTH] * M2FT)
# # plt.plot(time_data_s[:NEW_LENGTH], gps_data[:NEW_LENGTH,2] * M2FT)
# plt.title("Rocket Position (North is up)")
# plt.legend(["IMU", "GPS"])
# plt.xlabel("m")
# plt.ylabel("Distance (ft)")
# plt.xlim(-1000, 3000)
# plt.ylim(-2000, 3000)
# plt.show()


############################## Velocity Graph ##############################

plt.plot(time_data_s[:NEW_LENGTH], v_launch[:NEW_LENGTH, :] * M2FT)
plt.title("Rocket Velocity (Launch Frame)")
plt.legend(["X", "Y", "Z"])
plt.xlabel("Time (s)")
plt.ylabel("Speed (ft/s)")
plt.show()

############################## Acceleration Graph ##############################

plt.plot(time_data_s[:NEW_LENGTH], a_launch[:NEW_LENGTH, :] * M2FT)
plt.title("Rocket Acceleration (Launch Frame)")
plt.legend(["X", "Y", "Z"])
plt.xlabel("Time (s)")
plt.ylabel("Accel (ft/s/s)")
plt.show()

############################## Pitch Graph ##############################

# plt.plot(time_data_s[1:NEW_LENGTH], rocket_axis[1:])
plt.plot(time_data_s[:NEW_LENGTH], pitch)
plt.title("Pitch (Launch Frame)")
plt.xlabel("Time (s)")
plt.ylabel("Pitch (\u00b0)")
plt.show()


# f = plt.figure(1)
# plt.plot(time_data_s[:NEW_LENGTH], v_launch[:NEW_LENGTH,:] * M2FT)
# plt.show()
