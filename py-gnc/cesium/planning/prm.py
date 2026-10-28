import matplotlib.pyplot as plt
import numpy as np

from .environment import StaticOccupancyGrid2D, is_free_motion
from .plotting import plot_line_segments


class PRM:
    """PRM

    AA 274A
    """

    def __init__(
        self,
        lower_bounds: tuple | np.ndarray,
        upper_bounds: tuple | np.ndarray,
        x_init: tuple,
        x_goal: tuple,
        r0: float,
        occupancy_grid: StaticOccupancyGrid2D,
    ):
        """Grid AStar with bounds checking

        Parameters
        ----------
        xlim, ylim: tuple | np.ndarray
            Edges of A* search space
        x_init, x_goal : tuple
            Initial and goal states
        occupancy_grid : StaticOccupancyGrid2D
            Object representing grid
        """
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.r = r0  # radius of connecting things
        assert r0 > 0

        if len(self.upper_bounds) != len(self.lower_bounds):
            raise ValueError(
                "Bounds not the same size:\n"
                + f"  Upper: {upper_bounds}\n"
                + f"  Lower: {lower_bounds}"
            )
        self.state_dim = len(lower_bounds)
        for i, (u, l) in enumerate(zip(upper_bounds, lower_bounds)):
            if u <= l:
                raise ValueError(
                    f"Bounds at index {i} not initialized correctly:\n"
                    + f"  Upper: {upper_bounds}\n"
                    + f"  Lower: {lower_bounds}"
                )
        self.x_init = self.validate_tuple(x_init)
        self.x_goal = self.validate_tuple(x_goal)

        self.grid = occupancy_grid

        self.points = self.lines = None

    @staticmethod
    def validate_tuple(x: tuple):
        if not isinstance(x, (tuple, np.ndarray)):
            raise ValueError(f"Invalid state/bounds type of {x} ({type(x)})")
        if len(x) != 2:
            raise ValueError(f"Input point not 2D: {x}")
        return np.array(x)

    ##################################################
    #           First query
    ##################################################

    def splat_points(self, n: int, seed: int = None):
        rng = np.random.default_rng(seed)

        # Random, but make sure is free
        points = [self.x_init, self.x_goal]
        for i in range(n):
            p = np.array(rng.uniform(self.lower_bounds, self.upper_bounds, 2))
            if self.grid.is_free(p):
                points.append(p)

        # Add lines stupidly
        lines = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if i == j:
                    continue
                if np.linalg.norm(points[i] - points[j]) < self.r and is_free_motion(
                    self.grid.obstacles, points[i], points[j]
                ):
                    lines.append((points[i], points[j]))

        self.points = np.asarray(points)
        self.lines = np.asarray(lines)

        if len(self.points) == 0:
            # raise RuntimeError("Did not add any points")
            return False
        if len(self.lines) == 0:
            # raise RuntimeError("Did not add any lines")
            return False

        return True

    def plot_first_query(self, fig_num=0, point_size=15):
        if self.points is None or self.lines is None:
            raise RuntimeError("Need to splat points before plotting")

        # Grid and obstacles
        self.grid.plot(fig_num)
        plot_line_segments(self.grid.lines, color="red", linewidth=2, label="obstacles")

        # Points
        plt.scatter(
            self.points[:, 0],
            self.points[:, 1],
            color="blue",
            marker=".",
            s=point_size,
            zorder=10,
            alpha=0.2,
        )

        # Lines
        plot_line_segments(
            self.lines,
            color="blue",
            linewidth=0.5,
            alpha=0.4,
            label="PRM edges",
        )
        # Boundary
        plt.scatter(
            [self.x_init[0], self.x_goal[0]],
            [self.x_init[1], self.x_goal[1]],
            color="green",
            s=30,
            zorder=10,
        )
        plt.annotate(r"$x_{init}$", self.x_init[:2] + [0.2, 0], fontsize=16)
        plt.annotate(r"$x_{goal}$", self.x_goal[:2] + [0.2, 0], fontsize=16)
        plt.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3
        )
        plt.axis("scaled")
