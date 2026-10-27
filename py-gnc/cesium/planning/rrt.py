import matplotlib.pyplot as plt
import numpy as np

from ..math.base import unit
from .environment import line_line_intersection
from .plotting import plot_line_segments, plot_tree


class RRT:
    """Grid planning problem to be solved using RRT

    AA 274A
    """

    def __init__(
        self,
        lower_bounds: tuple | np.ndarray,
        upper_bounds: tuple | np.ndarray,
        x_init: tuple,
        x_goal: tuple,
        obstacles: list[tuple] | np.ndarray,
    ):
        """Grid AStar with bounds checking

        Parameters
        ----------
        lower_bounds, upper_bounds: tuple (2,)
            Edges of A* search space
        x_init, x_goal : tuple (2,)
            Initial and goal states
        obstacles : np.ndarray
            Line segments
        """

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        if len(self.upper_bounds) != len(self.lower_bounds):
            raise ValueError(
                "Bounds not the same size:\n"
                + f"  Upper: {upper_bounds}"
                + f"  Lower: {lower_bounds}"
            )
        self.state_dim = len(lower_bounds)
        for i, (u, l) in enumerate(zip(upper_bounds, lower_bounds)):
            if u <= l:
                raise ValueError(
                    f"Bounds at index {i} not initialized correctly:\n"
                    + f"  Upper: {upper_bounds}"
                    + f"  Lower: {lower_bounds}"
                )

        self.x_init = self.validate_tuple(x_init)
        self.x_goal = self.validate_tuple(x_goal)

        # Making sure obstacles are defined correctly
        try:
            self.obstacle_segments = np.asarray(obstacles)
        except Exception as e:
            raise ValueError(
                f"Error when casting obstacles to array: {e}]n" | obstacles
            )
        if self.obstacle_segments.shape[1:] != (2, self.state_dim):
            raise ValueError(
                f"obstacle shape weird: {self.obstacle_segments.shape}")

        self.path = None

    ##################################################
    #           Motion helpers
    ##################################################

    def validate_tuple(self, x: tuple | np.ndarray | list) -> np.ndarray:
        if len(x) != self.state_dim:
            raise ValueError(f"Input point not 2D: {x}")
        return np.asarray(x)

    def is_free_motion(self, obstacles: np.ndarray, x1, x2):
        """
        Whether a point robot moving along the shortest path from x1 to x2 would collide with any obstacles

        Inputs:
            obstacles: list/np.array of line segments ("walls")
            x1: start state of motion (2,)
            x2: end state of motion (2,)
        Output:
            Boolean True/False
        """
        x1 = self.validate_tuple(x1)
        x2 = self.validate_tuple(x2)
        motion = np.array([x1, x2])
        for line in obstacles:
            if line_line_intersection(line, motion):
                return False

        return True

    def find_nearest(self, V: np.ndarray, x):
        """
        Index of V such that the steering distance from
        V[i] to x is minimized

        Inputs:
            V: list/np.array of states (N,2)
            x - query state (2,)
        Output:
            Integer index of nearest point in V to x
        """
        V = np.asarray(V)  # (N,2)
        x = self.validate_tuple(x)  # (2,)
        distances = np.linalg.norm(V - x[ None, :], axis=1)
        result = np.argmin(distances)

        return result

    def steer_towards(self, x1, x2, eps: float):
        """
        Steers from x1 towards x2 along the shortest path (subject to robot
        dynamics). Returns x2 if the length of this shortest path is less than
        eps, otherwise returns the point at distance eps along the path from
        x1 to x2.

        Parameters
        ----------
        x1, x2 : np.ndarray | list (2,)
            Start and current state
        eps : float
            maximum steering distance

        Result
        ----------
        state_new : np.ndarray (2,)
        """
        x1 = self.validate_tuple(x1)
        x2 = self.validate_tuple(x2)

        if np.linalg.norm(x2 - x1) < eps:
            return x2

        return x1 + eps * unit(x2 - x1)

    def solve(
        self, eps: float, max_iters=1000, goal_bias=0.05, shortcut=False, plot=True,seed = None, **scatter_args
    ):
        """Use RRT to produce dynamically-feasible and obstacle-free trajectory

        Parameters
        ----------
        eps: maximum steering distance
        max_iters: maximum number of RRT iterations (early termination
            is possible when a feasible solution is found)
        goal_bias: probability during each iteration of setting
            x_rand = self.x_goal (instead of uniformly randly sampling
            from the state space)
        shortcut: whether to call shortcut_path on the solution (and, if
            plot is True, also plot the path before and after shortcutting)
        plot: whether to plot the tree and solution (set False for
            batch experiments)
        Output:
            Boolean, True if a solution was found. See also the
            "Intermediate Outputs" descriptions below


        Parameters
        ----------
        eps : float
            maximum steering distance
        max_iters : int, optional
            Maximum number of RRT iterations, by default 1000
        goal_bias : float, optional
            Probability of setting next aiming state to goal state, by default 0.05
        shortcut : bool, optional
            Whether to shortcut the path after finding it, by default False
        plot : bool, optional
            Plot path (before and after shortcutting, if applicable), by default True
        """

        # States added to RRT
        V = np.zeros((max_iters + 1, self.state_dim))
        V[0] = self.x_init
        n = 1

        # Parents of states (indices) of those indices
        P = -np.ones(max_iters + 1, dtype=int)

        success = False
        self.num_iters = 0  # Attribute in case it fails

        # Start
        self.path = []

        rng = np.random.default_rng(seed)

        for iter in range(max_iters):
            self.num_iters = iter

            z = rng.uniform()

            x_rand = (
                self.x_goal
                if z < goal_bias
                else rng.uniform(self.lower_bounds, self.upper_bounds, self.state_dim)
            )

            x_nearest_idx = self.find_nearest(V[:n], x_rand)
            x_near = V[x_nearest_idx]

            x_new = self.steer_towards(x_near, x_rand, eps)

            # Add new point with its parent set to x_near
            if self.is_free_motion(self.obstacle_segments, x_near, x_new):
                V[n] = x_new
                P[n] = x_nearest_idx
                n += 1

                if np.allclose(x_new, self.x_goal):
                    success = True
                    break

        # Traverses down parents until it gets to first one (parent of -1)
        i = n - 1
        while i != -1:
            self.path.append(V[i])
            i = P[i]

        self.path = np.array(self.path[::-1])

        # Plot

        if not plot:
            if success and shortcut:
                self.shortcut_path()
            return success

        plt.figure()
        self.plot_problem()
        plot_tree(V, P, color="blue", linewidth=0.5, label="RRT tree", alpha=0.5)
        if success:
            if shortcut:
                plt.plot(
                    self.path[:, 0],
                    self.path[:, 1],
                    color="purple",
                    linewidth=2,
                    label="Original solution path",
                )
                self.shortcut_path()
                plt.plot(
                    self.path[:, 0],
                    self.path[:, 1],
                    color="green",
                    linewidth=2,
                    label="Shortcut solution path",
                )
            else:
                plt.plot(
                    self.path[:, 0],
                    self.path[:, 1],
                    color="green",
                    linewidth=2,
                    label="Solution path",
                )
            plt.legend(
                loc="upper center", bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3
            )
            plt.scatter(V[:n, 0], V[:n, 1], **scatter_args)
        else:
            print("Solution not found!")

    def plot_problem(self):
        plot_line_segments(
            self.obstacle_segments, color="red", linewidth=2, label="obstacles"
        )
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

    def shortcut_path(self):
        """
        Iteratively removes nodes from solution path to find a shorter path
        which is still collision-free.
        """
        valid = False

        while not valid:
            valid = True
            for i in range(1, len(self.path) - 1):
                # Don't shortcut endpoints
                if np.allclose(self.path[i], self.x_goal) or np.allclose(
                    self.path[i], self.x_init
                ):
                    continue

                if self.is_free_motion(
                    self.obstacle_segments, self.path[i - 1], self.path[i + 1]
                ):
                    self.path = np.delete(self.path, i, axis=0)
                    valid = False
                    break
