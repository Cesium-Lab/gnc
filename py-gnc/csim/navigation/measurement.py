# @dataclass
# class Sensor:
#     """
#     Sensor abstraction that encapsulates measurement function, noise, and Jacobian.
#     measurement_fn and noise_cov can depend on SensorEnvironment for time-varying data.
#     """
#     name: str
#     meas_fn: Callable[[float, np.ndarray], np.ndarray]
#     R: np.ndarray | Callable[[np.ndarray], np.ndarray]
#     rng = np.random.Generator

#     def __post_init__(self):
#         self.m: int = self.R.shape[0]
#         assert self.R.shape == (self.m, self.m)

#     def measure(self, t: float, state_truth: np.ndarray, env: "SensorEnvironment") -> np.ndarray:
#         """Compute measurement without noise."""
#         return self.meas_fn(t, state_truth) + self.rng.multivariate_normal(np.zeros_like(state), self.R)
