import matplotlib.pyplot as plt
import numpy as np


def plot_line_segments(segments, **kwargs):
    """AA 274A Function"""
    plt.plot(
        [x for tup in [(p1[0], p2[0], None) for (p1, p2) in segments] for x in tup],
        [y for tup in [(p1[1], p2[1], None) for (p1, p2) in segments] for y in tup],
        **kwargs,
    )


def plot_tree(V, P, **kwargs):
    """AA 274A Function"""
    plot_line_segments(
        [(V[P[i], :], V[i, :]) for i in range(V.shape[0]) if P[i] >= 0], **kwargs
    )


def plot_path(path, **kwargs):
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], **kwargs)
