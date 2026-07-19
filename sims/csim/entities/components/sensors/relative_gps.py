import numpy as np

from .base import Sensor


class RelativeGPS(Sensor):
    """Differential/relative GPS: direct noisy relative position + velocity, in the
    target's Hill frame. Valid at medium/far range wherever both vehicles have GPS lock
    (i.e. Earth orbit, not deep space) -- no FOV or range constraint modeled.
    """

    def __init__(self, pos_noise_std: float = 0.02, vel_noise_std: float = 0.001,
                 rng: np.random.Generator | None = None):
        """
        Args:
            pos_noise_std (float, optional): 1-sigma position noise [m]. Defaults to 0.02.
            vel_noise_std (float, optional): 1-sigma velocity noise [m/s]. Defaults to 0.001.
            rng (np.random.Generator, optional): Defaults to `np.random.default_rng()`.
        """
        R = np.diag([pos_noise_std**2]*3 + [vel_noise_std**2]*3)
        super().__init__(R, rng)

    def h(self, rho: np.ndarray):
        return rho[:6]

    def H(self, rho: np.ndarray):
        return np.eye(6)
