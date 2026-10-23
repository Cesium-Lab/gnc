import numpy as np
import plotly.graph_objects as go

from .world import R_EARTH


def makesphere(x, y, z, radius, resolution=10):
    """Surface coordinates of a sphere, for plotting.

    Parameters
    ----------
    x, y, z : float
        Center of the sphere, in the plot frame.
    radius : float
        Sphere radius, in the same units as the center.
    resolution : int, optional
        Number of latitude samples; longitude uses twice as many. Default is 10.

    Returns
    -------
    X, Y, Z : np.ndarray (2 * resolution, resolution)
        Grid coordinates of the sphere surface.
    """
    u, v = np.mgrid[0 : 2 * np.pi : resolution * 2j, 0 : np.pi : resolution * 1j]
    X = radius * np.cos(u) * np.sin(v) + x
    Y = radius * np.sin(u) * np.sin(v) + y
    Z = radius * np.cos(v) + z
    return (X, Y, Z)


def plot_orbit(r: np.ndarray):
    """Plot a satellite trajectory around Earth in an interactive 3D figure.

    Parameters
    ----------
    r : np.ndarray (N, 3)
        Satellite position history in ECI [m], one position per row.
    """
    # plot orbits
    fig = go.Figure()

    # Plot trajectory
    fig.add_scatter3d(
        x=r[:, 0],
        y=r[:, 1],
        z=r[:, 2],
        mode="markers",
        marker=dict(size=1, color="blue"),
        name="Trajectory",
    )

    # Final Position
    fig.add_scatter3d(
        x=[r[-1, 0]],
        y=[r[-1, 1]],
        z=[r[-1, 2]],
        mode="markers",
        marker=dict(size=10, color="green"),
        name="Final Position",
    )

    # Initial Position
    fig.add_scatter3d(
        x=[r[0, 0]],
        y=[r[0, 1]],
        z=[r[0, 2]],
        mode="markers",
        marker=dict(
            size=10,
            color="red",
        ),
        name="Initial Position",
    )

    # Earth!
    X, Y, Z = makesphere(0, 0, 0, R_EARTH)
    fig.add_surface(
        x=X,
        y=Y,
        z=Z,
        colorscale=[[0, "blue"], [1, "blue"]],
        showlegend=False,
        showscale=False,
        name="Earth",
    )

    fig.update_layout(
        width=1000,
        height=800,
        title_font=dict(size=24, family="Garamond"),
        title_text="Satellite Trajectory",
        title_x=0.5,
        title_y=0.9,
    )
    fig.show()


def plot_rocket(r: np.ndarray):
    """Plot a rocket trajectory around Earth in an interactive 3D figure.

    Parameters
    ----------
    r : np.ndarray (N, 3)
        Rocket position history in an Earth-centered frame [m], one position per row.
    """
    # plot orbits
    fig = go.Figure()

    # Plot trajectory
    fig.add_scatter3d(
        x=r[:, 0],
        y=r[:, 1],
        z=r[:, 2],
        mode="markers",
        marker=dict(size=1, color="blue"),
        name="Trajectory",
    )

    # Final Position
    fig.add_scatter3d(
        x=[r[-1, 0]],
        y=[r[-1, 1]],
        z=[r[-1, 2]],
        mode="markers",
        marker=dict(size=10, color="green"),
        name="Final Position",
    )

    # Initial Position
    fig.add_scatter3d(
        x=[r[0, 0]],
        y=[r[0, 1]],
        z=[r[0, 2]],
        mode="markers",
        marker=dict(
            size=10,
            color="red",
        ),
        name="Initial Position",
    )

    # Earth!
    X, Y, Z = makesphere(0, 0, 0, R_EARTH)
    fig.add_surface(
        x=X,
        y=Y,
        z=Z,
        colorscale=[[0, "blue"], [1, "blue"]],
        showlegend=False,
        showscale=False,
        name="Earth",
    )

    fig.update_layout(
        width=1000,
        height=800,
        title_font=dict(size=24, family="Garamond"),
        title_text="Rocket Trajectory",
        title_x=0.5,
        title_y=0.9,
    )
    fig.show()
