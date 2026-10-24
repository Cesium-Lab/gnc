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

from dataclasses import dataclass

import numpy as np

G_TO_M_S2 = 9.80665

# ------------------------------
# Data Structures
# ------------------------------


@dataclass
class IMUData:
    """IMU samples from an Oxford Inertial Odometry Dataset `imu*.csv` file.

    Attributes
    ----------
    t : np.ndarray (N,)
        Sample timestamps [s].
    attitude : np.ndarray (N, 3)
        Device attitude as roll, pitch, yaw [rad].
    gyro : np.ndarray (N, 3)
        Body angular rates [rad/s].
    accel : np.ndarray (N, 3)
        User acceleration, with gravity removed [m/s2].
    gravity : np.ndarray (N, 3)
        Gravity vector in the device frame [m/s2].
    mag : np.ndarray (N, 3)
        Magnetic field [microtesla].
    """

    t: np.ndarray
    attitude: np.ndarray  # (N, 3) roll, pitch, yaw
    gyro: np.ndarray  # (N, 3) rad/s
    accel: np.ndarray  # (N, 3) m/s^2 (user acceleration)
    gravity: np.ndarray  # (N, 3) m/s^2
    mag: np.ndarray  # (N, 3) microtesla


@dataclass
class ViconData:
    """Ground-truth motion-capture samples from a `vi*.csv` file.

    Attributes
    ----------
    t : np.ndarray (N,)
        Sample timestamps [s].
    header : np.ndarray (N,)
        Vicon header (frame sequence) values.
    position : np.ndarray (N, 3)
        Position in the Vicon world frame [m].
    quaternion : np.ndarray (N, 4)
        Scalar-first attitude quaternions `[w, x, y, z]`.
    """

    t: np.ndarray
    header: np.ndarray
    position: np.ndarray  # (N, 3)
    quaternion: np.ndarray  # (N, 4)


# ------------------------------
# IMU Parser
# ------------------------------


def parse_imu_csv(path: str) -> IMUData:
    """Load an Oxford Inertial Odometry Dataset IMU file.

    Parameters
    ----------
    path : str
        Path to an `imu*.csv` file.

    Returns
    -------
    IMUData
        Parsed IMU samples, with accelerations converted from G to [m/s2].
    """
    data = np.loadtxt(path, delimiter=",")

    t = data[:, 0]

    attitude = data[:, 1:4]  # roll, pitch, yaw
    gyro = data[:, 4:7]  # wx, wy, wz (good)

    gravity = data[:, 7:10] * G_TO_M_S2
    accel = data[:, 10:13] * G_TO_M_S2

    mag = data[:, 13:16]

    return IMUData(
        t=t, attitude=attitude, gyro=gyro, accel=accel, gravity=gravity, mag=mag
    )


# ------------------------------
# Vicon Parser
# ------------------------------


def parse_vicon_csv(path: str) -> ViconData:
    """Load an Oxford Inertial Odometry Dataset Vicon ground-truth file.

    Parameters
    ----------
    path : str
        Path to a `vi*.csv` file.

    Returns
    -------
    ViconData
        Parsed ground truth, with quaternions reordered to scalar-first.
    """
    data = np.loadtxt(path, delimiter=",")

    t = data[:, 0]
    header = data[:, 1]

    position = data[:, 2:5]
    xyz = data[:, 5:8]  # (N,3)
    w = data[:, 8]  # (N,)

    return ViconData(
        t=t,
        header=header,
        position=position,
        quaternion=np.vstack((w, xyz.T)).T,  # numpy vectors suck
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
