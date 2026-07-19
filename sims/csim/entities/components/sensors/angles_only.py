import numpy as np
from numpy.linalg import norm

from .base import Sensor


class AnglesOnly(Sensor):
    """Camera/star-tracker style bearing-only sensor: measures the line-of-sight
    azimuth/elevation to the target, no range. This is the classic "angles-only"
    RPO navigation sensor -- relative range is unobservable without a maneuver to
    create observability (a well-known limitation, not a bug in this model).
    """

    def __init__(self, angle_noise_std_rad: float = 1e-4, fov_half_angle_rad: float = np.pi,
                 boresight: np.ndarray = np.array([-1.0, 0.0, 0.0]),
                 rng: np.random.Generator | None = None):
        """
        Args:
            angle_noise_std_rad (float, optional): 1-sigma az/el noise [rad]. Defaults to 1e-4.
            fov_half_angle_rad (float, optional): Half-angle of the sensor FOV about `boresight` [rad].
                Defaults to pi (unconstrained).
            boresight (np.ndarray, optional): Sensor boresight unit vector (Hill frame).
                Defaults to -x (chaser looking back at a target ahead on V-bar/R-bar).
            rng (np.random.Generator, optional): Defaults to `np.random.default_rng()`.
        """
        R = np.diag([angle_noise_std_rad**2, angle_noise_std_rad**2])
        super().__init__(R, rng)
        self.fov_half_angle_rad = fov_half_angle_rad
        self.boresight = boresight / norm(boresight)

    def in_view(self, rho: np.ndarray):
        los = rho[:3] / norm(rho[:3])
        angle_off_boresight = np.arccos(np.clip(np.dot(los, self.boresight), -1, 1))
        return angle_off_boresight <= self.fov_half_angle_rad

    def h(self, rho: np.ndarray):
        x, y, z = rho[:3]
        az = np.arctan2(y, x)
        el = np.arcsin(z / norm(rho[:3]))
        return np.array([az, el])

    def H(self, rho: np.ndarray):
        x, y, z = rho[:3]
        r = norm(rho[:3])
        rho_xy = np.hypot(x, y)

        d_az = np.array([-y, x, 0]) / rho_xy**2
        d_el = np.array([-z*x, -z*y, rho_xy**2]) / (r**2 * rho_xy)

        return np.array([
            [*d_az, 0, 0, 0],
            [*d_el, 0, 0, 0],
        ])
