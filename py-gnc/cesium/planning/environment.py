import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def box(lo, hi):
    """Returns the four walls of an axis-aligned rectangle with corners lo and hi."""
    (x0, y0), (x1, y1) = lo, hi
    return [
        ((x0, y0), (x1, y0)),
        ((x1, y0), (x1, y1)),
        ((x1, y1), (x0, y1)),
        ((x0, y1), (x0, y0)),
    ]


class StaticOccupancyGrid2D:
    """
    A 2D state space with a set of rectangular obstacles on a grid

    AA 274A Function
    """

    def __init__(self, width: float, height: float, obstacles: list):
        """Grid

        Parameters
        ----------
        height, width: float
            Dimensions of grid
        obstacles : list (N,2,2)
            Obstacles with ((bottom left), (top right)) coordinates
        """
        self.width = width
        self.height = height
        self.obstacles = obstacles

        self.lines = box((0, 0), (width, height))

        for o in self.obstacles:
            assert len(o[0]) == 2
            assert len(o[1]) == 2

            self.lines = self.lines + box(o[0], o[1])

    def is_free(self, x: tuple):
        """Verifies that point is not inside any obstacles by some margin"""
        assert len(x) == 2
        for obs in self.obstacles:
            if (
                x[0] >= obs[0][0] - self.width * 0.01
                and x[0] <= obs[1][0] + self.width * 0.01
                and x[1] >= obs[0][1] - self.height * 0.01
                and x[1] <= obs[1][1] + self.height * 0.01
            ):
                return False
        return True

    def plot(self, fig_num=0):
        """Plots the space and its obstacles"""
        fig = plt.figure(fig_num)
        ax = fig.add_subplot(111, aspect="equal")
        for obs in self.obstacles:
            ax.add_patch(
                patches.Rectangle(
                    obs[0],
                    obs[1][0] - obs[0][0],
                    obs[1][1] - obs[0][1],
                )
            )
        ax.set(xlim=(0, self.width), ylim=(0, self.height))


def is_free_motion(obstacles: np.ndarray, x1, x2):
    motion = np.array([x1, x2])
    for line in obstacles:
        if line_line_intersection(line, motion):
            return False
    return True


class SegmentGraph2D:
    # TODO
    pass


def generate_planning_problem(
    width: float, height: float, num_obs: int, min_size: float, max_size: float
):
    x_margin = width * 0.1
    y_margin = height * 0.1
    obs_corners_x = np.random.uniform(-x_margin, width + x_margin, num_obs)
    obs_corners_y = np.random.uniform(-y_margin, height + y_margin, num_obs)
    obs_lower_corners = np.vstack([obs_corners_x, obs_corners_y]).T
    obs_sizes = np.random.uniform(min_size, max_size, (num_obs, 2))
    obs_upper_corners = obs_lower_corners + obs_sizes
    obstacles = list(zip(obs_lower_corners, obs_upper_corners))
    occupancy = StaticOccupancyGrid2D(width, height, obstacles)

    x_init = tuple(np.random.uniform(0, width - x_margin, 2).tolist())
    while not occupancy.is_free(x_init):
        x_init = tuple(np.random.randint(0, width - x_margin, 2).tolist())
    x_goal = x_init
    while (not occupancy.is_free(x_goal)) or (
        np.linalg.norm(np.array(x_goal) - np.array(x_init))
        < np.sqrt(width**2 + height**2) * 0.4
    ):
        x_goal = tuple(np.random.uniform(0, width - x_margin, 2).tolist())

    return occupancy, x_init, x_goal


def line_line_intersection(l1: np.ndarray, l2: np.ndarray):
    """Checks whether or not two 2D line segments `l1` and `l2` intersect.

    AA 274A Function

    Args:
        l1: A line segment in 2D, i.e., an array-like of two points `((x_start, y_start), (x_end, y_end))`.
        l2: A line segment in 2D, i.e., an array-like of two points `((x_start, y_start), (x_end, y_end))`.

    Returns:
        `True` iff `l1` and `l2` intersect.
    """
    l1 = np.asarray(l1)
    l2 = np.asarray(l2)
    assert l1.shape == (2, 2)
    assert l2.shape == (2, 2)

    def ccw(A, B, C):
        return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0]) > 0

    A, B = np.array(l1)
    C, D = np.array(l2)
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
