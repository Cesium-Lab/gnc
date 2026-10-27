from __future__ import annotations

from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

from .environment import StaticOccupancyGrid2D
from .plotting import plot_line_segments


class AStarHeuristic(Enum):
    l1 = 1
    l2 = 2
    linf = np.inf


# TODO: make it generalizable for a different graph too (SegmentGraph2D)


class AStarGrid:
    """Grid planning problem to be solved using A*

    AA 274A
    """

    def __init__(
        self,
        xlim: tuple,
        ylim: tuple,
        x_init: tuple,
        x_goal: tuple,
        occupancy_grid: StaticOccupancyGrid2D,
        resolution=1.0,
        heuristic=AStarHeuristic.l2,
        heuristic_weight=1.0,
    ):
        """Grid AStar with bounds checking

        Parameters
        ----------
        xlim, ylim: tuple
            Edges of A* search space
        x_init, x_goal : tuple
            Initial and goal states
        occupancy_grid : StaticOccupancyGrid2D
            Object representing grid
        resolution : float, optional
            Resolution of discretization of A* search space, by default 1.0
        heuristic : AStarHeuristic, optional
            Enum of heuristic, by default AStarHeuristic.l2
        heuristic_weight : float, optional
            Weight w >= 1 on the heuristic (w > 1 gives weighted A*), By default 1.0
        """

        self.xlim = self.validate_tuple(xlim)
        self.ylim = self.validate_tuple(ylim)
        if self.xlim[1] <= self.xlim[0]:
            raise ValueError(f"X min and max not initialized correctly: {self.xlim}")
        if self.ylim[1] <= self.ylim[0]:
            raise ValueError(f"Y min and max not initialized correctly: {self.ylim}")

        self.resolution = resolution

        self.center = self.validate_tuple(x_init)  # Center of where it is initialized
        self.x_init = self.validate_tuple(self.snap_to_grid(x_init))
        self.x_goal = self.validate_tuple(self.snap_to_grid(x_goal))

        self.grid = occupancy_grid

        self.heuristic_type = heuristic  # typing is enforced by type
        self.heuristic_weight = heuristic_weight
        assert self.heuristic_weight >= 1.0

        self.clear_sets()

    def clear_sets(self):
        # States visited and are candidates
        self.visited, self.candidates = set(), {self.x_init}

        # Cost tracking
        self.cost_to_arrive = {self.x_init: 0}
        self.est_cost_through = {self.x_init: self.heuristic(self.x_init)}

        # For reconstructing trajectory
        self.came_from = {}
        self.path = []

    @staticmethod
    def validate_tuple(x: tuple):
        if not isinstance(x, tuple):
            raise ValueError(f"Invalid state/bounds type of {x} ({type(x)})")
        if len(x) != 2:
            raise ValueError(f"Input point not 2D: {x}")
        return x

    ##################################################
    #           Grid helpers
    ##################################################

    def snap_to_grid(self, x: tuple) -> tuple:
        """Returns the closest point on a discrete state grid as an `np.ndarray`"""
        x = self.validate_tuple(x)

        # Can do another way probably
        return (
            self.resolution * round((x[0] - self.center[0]) / self.resolution)
            + self.center[0],
            self.resolution * round((x[1] - self.center[1]) / self.resolution)
            + self.center[1],
        )

    def is_free(self, x: tuple) -> bool:
        """Checks if state is inside the bounds and not inside any obstacle"""
        x = self.validate_tuple(x)
        in_bounds = (
            x[0] >= self.xlim[0]
            and x[0] <= self.xlim[1]
            and x[1] >= self.ylim[0]
            and x[1] <= self.ylim[1]
        )
        collision_free = self.grid.is_free(x)

        return collision_free and in_bounds

    def distance(self, x1: tuple, x2: tuple) -> float:
        """Euclidean distance"""
        x1 = self.validate_tuple(x1)
        x2 = self.validate_tuple(x2)
        return float(np.linalg.norm(np.array(x1) - np.array(x2)))

    def get_neighbors(self, x: tuple) -> list[tuple]:
        """Gets the FREE neighbor states of a given state x. Right now, assumed 8-connected-grid.

        Could be replaced by a function from the grid

        Parameters
        ----------
        x: state

        Returns
        ----------
        heuristic_value: np.ndarray (N,2)
            Neighbors that are free
        """
        x = self.validate_tuple(x)

        neighbors = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue

                candidate = self.snap_to_grid(
                    (x[0] + i * self.resolution, x[1] + j * self.resolution)
                )
                if not self.is_free(candidate):
                    continue

                neighbors.append(candidate)

        return neighbors

    ##################################################
    #           For solving
    ##################################################

    def heuristic(self, x: tuple) -> float:
        """
        Computes the weighted heuristic estimate of the cost-to-go from state x
        to self.x_goal

        i.e. `w * heuristic(x - x_goal), where w = self.heuristic_weight`

        Parameters
        ----------
        x: state

        Returns
        ----------
        heuristic_value: float
        """
        x = self.validate_tuple(x)

        if self.heuristic_type not in (
            AStarHeuristic.l1,
            AStarHeuristic.l2,
            AStarHeuristic.linf,
        ):
            raise ValueError(
                f"Heuristic of type {self.heuristic_type} not implemented yet"
            )

        return self.heuristic_weight * np.linalg.norm(
            np.asarray(self.x_goal) - np.asarray(x), ord=self.heuristic_type.value
        )

    def _find_best_est_cost_through(self) -> tuple:
        """
        Gets the state in candidates that has the lowest est_cost_through
        """
        return min(self.candidates, key=lambda x: self.est_cost_through[x])

    def _reconstruct_path(self) -> np.ndarray:
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location

        Returns
        ----------
        path : np.ndarray (N,2)
            Array of points from `x_init` to `x_goal`

        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return np.array(list(reversed(path)))

    def solve(self):
        """Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path

        Returns
        -------
        bool
            True if a solution was found
        """
        ########## Code starts here ##########

        self.clear_sets()

        while len(self.candidates) > 0:
            x_curr = self._find_best_est_cost_through()
            if x_curr == self.x_goal:
                self.path = self._reconstruct_path()
                return True

            self.candidates.remove(x_curr)
            self.visited.add(x_curr)

            for x_neigh in self.get_neighbors(x_curr):
                if x_neigh in self.visited:
                    continue

                tentative_cost_to_arrive = self.cost_to_arrive[x_curr] + self.distance(
                    x_curr, x_neigh
                )

                if x_neigh not in self.candidates:
                    self.candidates.add(x_neigh)
                elif tentative_cost_to_arrive > self.cost_to_arrive[x_neigh]:
                    continue

                self.came_from[x_neigh] = x_curr
                self.cost_to_arrive[x_neigh] = tentative_cost_to_arrive
                self.est_cost_through[x_neigh] = (
                    tentative_cost_to_arrive + self.heuristic(x_neigh)
                )

        return False

    ##################################################
    #           Plotting!
    ##################################################

    def plot_path(self, fig_num=0, show_init_label=True):
        """Plots the path found in self.path and the obstacles"""
        if self.path is None:
            return

        self.grid.plot(fig_num)

        solution_path = np.asarray(self.path)
        plt.plot(
            solution_path[:, 0],
            solution_path[:, 1],
            color="green",
            linewidth=2,
            label="A* solution path",
            zorder=10,
        )
        plt.scatter(
            [self.x_init[0], self.x_goal[0]],
            [self.x_init[1], self.x_goal[1]],
            color="green",
            s=30,
            zorder=10,
        )
        if show_init_label:
            plt.annotate(
                r"$x_{init}$", np.array(self.x_init) + np.array([0.2, 0.2]), fontsize=16
            )
        plt.annotate(
            r"$x_{goal}$", np.array(self.x_goal) + np.array([0.2, 0.2]), fontsize=16
        )
        plt.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3
        )

        plt.axis([0, self.grid.width, 0, self.grid.height])

    def plot_tree(self, point_size=15):
        plot_line_segments(
            [(x, self.came_from[x]) for x in self.candidates if x != self.x_init],
            linewidth=1,
            color="blue",
            alpha=0.2,
        )
        plot_line_segments(
            [(x, self.came_from[x]) for x in self.visited if x != self.x_init],
            linewidth=1,
            color="blue",
            alpha=0.2,
        )
        px = [
            x[0]
            for x in self.candidates | self.visited
            if x != self.x_init and x != self.x_goal
        ]
        py = [
            x[1]
            for x in self.candidates | self.visited
            if x != self.x_init and x != self.x_goal
        ]
        plt.scatter(px, py, color="blue", s=point_size, zorder=10, alpha=0.2)
