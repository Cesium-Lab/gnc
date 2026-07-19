import numpy as np
from numpy.linalg import norm

from .base import Sensor


class RFRanging(Sensor):
    """RF crosslink/radar ranging: range and range-rate only, no bearing.
    Classic "range-only" RPO sensor -- relative position direction is unobservable
    from this sensor alone.
    """

    def __init__(self, range_noise_std: float = 1.0, range_rate_noise_std: float = 0.01,
                 min_range_m: float = 0.0, max_range_m: float = np.inf,
                 rng: np.random.Generator | None = None):
        """
        Args:
            range_noise_std (float, optional): 1-sigma range noise [m]. Defaults to 1.0.
            range_rate_noise_std (float, optional): 1-sigma range-rate noise [m/s]. Defaults to 0.01.
            min_range_m (float, optional): Minimum valid range [m]. Defaults to 0.
            max_range_m (float, optional): Maximum valid range [m]. Defaults to inf.
            rng (np.random.Generator, optional): Defaults to `np.random.default_rng()`.
        """
        R = np.diag([range_noise_std**2, range_rate_noise_std**2])
        super().__init__(R, rng)
        self.min_range_m = min_range_m
        self.max_range_m = max_range_m

    def in_view(self, rho: np.ndarray):
        r = norm(rho[:3])
        return self.min_range_m <= r <= self.max_range_m

    def h(self, rho: np.ndarray):
        r_vec, v_vec = rho[:3], rho[3:]
        r = norm(r_vec)
        range_rate = np.dot(r_vec, v_vec) / r
        return np.array([r, range_rate])

    def H(self, rho: np.ndarray):
        x, y, z, xd, yd, zd = rho
        r_vec, v_vec = rho[:3], rho[3:]
        r = norm(r_vec)

        d_range = r_vec / r

        rdotv = np.dot(r_vec, v_vec)
        d_range_rate_dr = v_vec/r - rdotv*r_vec/r**3
        d_range_rate_dv = r_vec / r

        return np.array([
            [*d_range, 0, 0, 0],
            [*d_range_rate_dr, *d_range_rate_dv],
        ])
