import numpy as np
from numpy.linalg import norm

from .base import Sensor
from .angles_only import AnglesOnly


class Lidar(Sensor):
    """Scanning lidar: range + azimuth/elevation, valid over a bounded range and FOV,
    with range noise that grows with distance (`std = range_noise_floor + range_noise_scale * range`),
    typical of real rangefinders.
    """

    def __init__(self, range_noise_floor_m: float = 0.01, range_noise_scale: float = 1e-4,
                 angle_noise_std_rad: float = 1e-3,
                 min_range_m: float = 1.0, max_range_m: float = 2000.0,
                 fov_half_angle_rad: float = np.deg2rad(15),
                 boresight: np.ndarray = np.array([-1.0, 0.0, 0.0]),
                 rng: np.random.Generator | None = None):
        """
        Args:
            range_noise_floor_m (float, optional): Range noise std at zero range [m]. Defaults to 0.01.
            range_noise_scale (float, optional): Additional range noise std per meter of range. Defaults to 1e-4.
            angle_noise_std_rad (float, optional): 1-sigma az/el noise [rad]. Defaults to 1e-3.
            min_range_m (float, optional): Minimum valid range [m]. Defaults to 1.0.
            max_range_m (float, optional): Maximum valid range [m]. Defaults to 2000.0.
            fov_half_angle_rad (float, optional): Half-angle of the scan FOV about `boresight` [rad].
                Defaults to 15 degrees.
            boresight (np.ndarray, optional): Sensor boresight unit vector (Hill frame). Defaults to -x.
            rng (np.random.Generator, optional): Defaults to `np.random.default_rng()`.
        """
        # R is range-dependent, so it's rebuilt each measure() call -- placeholder here
        super().__init__(np.eye(3), rng)
        self.range_noise_floor_m = range_noise_floor_m
        self.range_noise_scale = range_noise_scale
        self.angle_noise_std_rad = angle_noise_std_rad
        self.min_range_m = min_range_m
        self.max_range_m = max_range_m
        self._angles = AnglesOnly(angle_noise_std_rad, fov_half_angle_rad, boresight, rng)

    def in_view(self, rho: np.ndarray):
        r = norm(rho[:3])
        in_range = self.min_range_m <= r <= self.max_range_m
        return in_range and self._angles.in_view(rho)

    def h(self, rho: np.ndarray):
        r = norm(rho[:3])
        return np.hstack(([r], self._angles.h(rho)))

    def H(self, rho: np.ndarray):
        r_vec = rho[:3]
        r = norm(r_vec)
        d_range = np.hstack((r_vec / r, [0, 0, 0]))
        return np.vstack((d_range, self._angles.H(rho)))

    def measure(self, rho_truth: np.ndarray):
        if not self.in_view(rho_truth):
            return None, False

        r = norm(rho_truth[:3])
        range_std = self.range_noise_floor_m + self.range_noise_scale * r
        self.R = np.diag([range_std**2, self.angle_noise_std_rad**2, self.angle_noise_std_rad**2])

        return super().measure(rho_truth)
