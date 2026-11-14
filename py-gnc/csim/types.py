from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class TruthPropType(Enum):
    """How the simulator propagates the true relative state."""

    STM = auto()
    """Closed-form exact HCW STM."""
    RK4 = auto()
    """Fixed-step RK4 on `hcw_dynamics`"""


@dataclass
class SimSensor:
    """Truth-side sensor: measures the true state with its own true noise.

    It knows nothing about the flight computer's model of it (`MeasModel`); the two
    only share the measurement function `h` and the `sensor_id`.
    `R_true=None` is a perfect sensor (no noise). Noise comes from an explicit,
    seedable generator so Monte Carlo runs are reproducible.
    """

    sensor_id: str
    """Tag sent with each measurement; must be a key in the flight computer's models."""
    h: Callable[[np.ndarray], np.ndarray]
    """Measurement function `h(x_true) -> (M,)`."""
    R_true: np.ndarray | None = None
    """True noise covariance (M, M), or None for no noise."""
    rng: np.random.Generator | None = None
    """Noise source; required when `R_true` is set (seed it for repeatable runs)."""
    every: int = 1
    """Measure on every `every`-th sim step (1 = every step)."""

    def __post_init__(self) -> None:
        if self.R_true is not None and self.rng is None:
            raise ValueError("a noisy sensor needs an rng")
        if self.every < 1:
            raise ValueError(f"every must be >= 1, got {self.every}")

    def fires(self, k: int) -> bool:
        """Whether this sensor measures on sim step `k`."""
        return k % self.every == 0

    def measure(self, x_true: np.ndarray) -> np.ndarray:
        z = np.array(self.h(x_true), dtype=float)
        if self.R_true is not None:
            assert self.rng is not None  # checked in __post_init__
            z += self.rng.multivariate_normal(np.zeros(len(z)), self.R_true)
        return z


################################################################################
#               Simulation
################################################################################


@dataclass(frozen=True)
class SimResult:
    """Time histories from `RelNavSim.run`"""

    t: np.ndarray
    """Time (K,) [s]."""
    x_true: np.ndarray
    """True relative states (K, 6) [m, m/s]."""
    z: dict[str, np.ndarray]
    """Measurements per sensor id, each (K, M); NaN rows on steps it didn't fire."""
    x_est: np.ndarray
    """Flight computer estimates after each step (K, 6)."""
    P: np.ndarray
    """Flight computer covariances after each step (K, 6, 6)."""

    @property
    def err(self) -> np.ndarray:
        """Estimation error `x_est - x_true` (K, 6)."""
        return self.x_est - self.x_true
