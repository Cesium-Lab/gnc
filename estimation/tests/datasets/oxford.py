## Code for parsing Oxford Inertial Odometry Dataset

"""

In each data fold, there is a raw data subfolder and a syn data subfolder,
which represent the raw data collection without synchronisation but with high precise timestep, 
and the synchronised data but without high precise timestep.

Here is the header of the sensor file and ground truth file.

vicon (vi*.csv)

- Time
- Header
- translation.x
- translation.y
- translation.z
- rotation.x
- rotation.y
- rotation.z
- rotation.w

Sensors (imu*.csv)

- Time
- attitude_roll(radians)
- attitude_pitch(radians)
- attitude_yaw(radians)
- rotation_rate_x(radians/s)
- rotation_rate_y(radians/s)
- rotation_rate_z(radians/s)
- gravity_x(G)
- gravity_y(G)
- gravity_z(G)
- user_acc_x(G)
- user_acc_y(G)
- user_acc_z(G)
- magnetic_field_x(microteslas)
- magnetic_field_y(microteslas)
- magnetic_field_z(microteslas)

"""

import pandas as pd
import numpy as np
from dataclasses import dataclass

G_TO_M_S2 = 9.80665

# ------------------------------
# Data Structures
# ------------------------------

@dataclass
class IMUData:
    t: np.ndarray
    attitude: np.ndarray      # (N, 3) roll, pitch, yaw
    gyro: np.ndarray          # (N, 3) rad/s
    accel: np.ndarray         # (N, 3) m/s^2 (user acceleration)
    gravity: np.ndarray       # (N, 3) m/s^2
    mag: np.ndarray           # (N, 3) microtesla

@dataclass
class ViconData:
    t: np.ndarray
    position: np.ndarray      # (N, 3)
    quaternion: np.ndarray    # (N, 4)

# ------------------------------
# IMU Parser
# ------------------------------

def parse_imu_csv(path: str) -> IMUData:
    data = np.loadtxt(path, delimiter=",")

    t = data[:, 0]

    attitude = data[:, 1:4]  # roll, pitch, yaw
    gyro = data[:, 4:7]  # wx, wy, wz (good)

    gravity = data[:, 7:10] * G_TO_M_S2
    accel   = data[:, 10:13] * G_TO_M_S2

    mag = data[:, 13:16]

    return IMUData(
        t=t,
        attitude=attitude,
        gyro=gyro,
        accel=accel,
        gravity=gravity,
        mag=mag
    )

# ------------------------------
# Vicon Parser
# ------------------------------

def parse_vicon_csv(path: str) -> ViconData:
    data = np.loadtxt(path, delimiter=",")

    t = data[:, 0]

    position = data[:, 1:4]
    quaternion = data[:, 4:8]

    return ViconData(
        t=t,
        position=position,
        quaternion=quaternion
    )

# ------------------------------
# Example Usage
# ------------------------------

if __name__ == "__main__":
    base = "Oxford Inertial Odometry Dataset/handbag/data1/raw"
    imu = parse_imu_csv(f"{base}/imu1.csv")
    vicon = parse_vicon_csv(f"{base}/vi1.csv")

    print("IMU samples:", len(imu.t))
    print("Vicon samples:", len(vicon.t))