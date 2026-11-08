from dataclasses import dataclass

import numpy as np

from cesium.world.constants import G0


@dataclass
class Propulsion:
    I_sp: float
    """s"""
    thrust_max: float
    """N"""
    thrust_min: float = 0.0
    """N"""

    def __post_init__(self):
        if not (0 <= self.thrust_min <= self.thrust_max < np.inf):
            raise ValueError("Require 0 <= thrust_min <= thrust_max, thrust_max > 0")
        if not 0 < self.I_sp < np.inf:
            raise ValueError("Require 0 < I_sp < inf")

    def force(self, throttle=1.0):
        """Force in Newtons. Clamp to 0-1"""
        throttle = np.clip(throttle, 0, 1)
        force = throttle * (self.thrust_max - self.thrust_min) + self.thrust_min
        return force

    def m_dot(self, throttle=1.0):
        """Mass flow rate in kg/s"""
        force = self.force(throttle)
        m_dot = force / (self.I_sp * G0)
        return m_dot
