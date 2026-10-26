import numpy as np


class DetOccupancyGrid2D(object):
    """
    A 2D state space grid with a set of rectangular obstacles. The grid is
    fully deterministic
    """

    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def is_free(self, x):
        """Verifies that point is not inside any obstacles by some margin"""
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


def generate_planning_problem(width, height, num_obs, min_size, max_size):
    from P1_astar import DetOccupancyGrid2D

    x_margin = width * 0.1
    y_margin = height * 0.1
    obs_corners_x = np.random.uniform(-x_margin, width + x_margin, num_obs)
    obs_corners_y = np.random.uniform(-y_margin, height + y_margin, num_obs)
    obs_lower_corners = np.vstack([obs_corners_x, obs_corners_y]).T
    obs_sizes = np.random.uniform(min_size, max_size, (num_obs, 2))
    obs_upper_corners = obs_lower_corners + obs_sizes
    obstacles = list(zip(obs_lower_corners, obs_upper_corners))
    occupancy = DetOccupancyGrid2D(width, height, obstacles)

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


def line_line_intersection(l1, l2):
    """Checks whether or not two 2D line segments `l1` and `l2` intersect.

    Args:
        l1: A line segment in 2D, i.e., an array-like of two points `((x_start, y_start), (x_end, y_end))`.
        l2: A line segment in 2D, i.e., an array-like of two points `((x_start, y_start), (x_end, y_end))`.

    Returns:
        `True` iff `l1` and `l2` intersect.
    """

    def ccw(A, B, C):
        return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0]) > 0

    A, B = np.array(l1)
    C, D = np.array(l2)
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
