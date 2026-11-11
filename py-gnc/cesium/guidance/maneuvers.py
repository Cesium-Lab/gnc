"""Maneuver representations shared by guidance planners."""

from dataclasses import dataclass

import numpy as np
from numpy.linalg import norm


@dataclass(frozen=True)
class Burn:
    """Impulsive maneuver.

    `r` and `dv` are expressed in the axes named by `frame`. Times are relative
    to the epoch of the plan that holds the burn (e.g. periapsis arrival), which
    keeps full float precision compared with absolute Julian dates.
    """

    t_s: float
    """Time of the burn relative to the plan's epoch [s]."""
    r: np.ndarray
    """(3,) Position at the burn [m]."""
    dv: np.ndarray
    """(3,) Delta-v vector [m/s]."""
    frame: str = "PCI"
    """Axes of `r` and `dv`, e.g. "MCI", "ECI", "RSW"."""

    @property
    def dv_mag(self) -> float:
        """|dv| [m/s]."""
        return float(norm(self.dv))
