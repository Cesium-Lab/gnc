"""What the flight computer knows about its measurements.

Each sensor onboard has a `MeasModel` (the filter's model of it), stored in the flight
computer under a sensor id. Measurements arrive tagged with that id. Two sensors can
share `h` and `H` (e.g. a shock and a normal accelerometer measuring the same thing)
and differ only in `R`.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from cesium.navigation.kf import plain_residual


@dataclass(frozen=True)
class MeasModel:
    """The flight computer's model of one sensor."""

    h: Callable[[np.ndarray], np.ndarray]
    """Measurement function `h(x) -> (M,)`, from `cesium.navigation.measurements`."""
    H: Callable[[np.ndarray], np.ndarray]
    """Jacobian of `h` for the full state, `H(x) -> (M, N)`."""
    R: np.ndarray
    """Measurement noise covariance assumed by the filter (M, M)."""
    residual: Callable[[np.ndarray, np.ndarray], np.ndarray] = plain_residual
    """Innovation `residual(z, z_hat) -> (M,)`; override to wrap angles."""


@dataclass(frozen=True)
class Measurement:
    """One measurement, tagged with the id of the sensor that made it."""

    sensor_id: str
    """Key into the flight computer's `meas_models`, e.g. "rel-pos"."""
    z: np.ndarray
    """Measurement (M,)."""
