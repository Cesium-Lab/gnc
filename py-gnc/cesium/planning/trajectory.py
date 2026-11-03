import numpy as np
from scipy.interpolate import splev, splrep


class TrajectoryPlan:
    def __init__(
        self,
        path: np.ndarray | list[tuple],
        path_x_spline: tuple[np.ndarray, np.ndarray, int],
        path_y_spline: tuple[np.ndarray, np.ndarray, int],
        duration: float,
    ):
        self.path = path if isinstance(path, np.ndarray) else np.array(path)

        # Cubic splines
        self.path_x_spline = path_x_spline
        self.path_y_spline = path_y_spline
        self.duration = duration

    def smoothed_path(self, dt=0.1) -> np.ndarray:
        ts = np.arange(0.0, self.duration, dt)
        path = np.zeros((ts.shape[0], 2))
        path[:, 0] = splev(ts, self.path_x_spline, der=0)
        path[:, 1] = splev(ts, self.path_y_spline, der=0)

        return path


def path_length(path) -> float:
    path = np.asarray(path)
    return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))


def smooth_path(path: list[tuple], v_desired=0.15, spline_alpha=0.05):
    path: np.ndarray = np.asarray(path)  # want to do this for type checking

    total_dist = path_length(path)
    tspan = total_dist / v_desired
    nsteps = path.shape[0]
    ts = np.linspace(0, tspan, nsteps)

    path_x_spline = splrep(ts, path[:, 0], k=3, s=spline_alpha)
    path_y_spline = splrep(ts, path[:, 1], k=3, s=spline_alpha)

    return TrajectoryPlan(
        path=path,
        path_x_spline=path_x_spline,
        path_y_spline=path_y_spline,
        duration=ts[-1],
    )
