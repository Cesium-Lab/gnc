"""TODO: organize"""

import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline

from .constants import AU_TO_KM

from plotly.colors import qualitative
import matplotlib.pyplot as plt

from .orbital.flyby import OrbitalFlyby

def plot_3d(
    traces,
    t_arrs=None,
    labels=None,
    title="3D Trajectories",
    bodies=None,
    t_max=None,
    show_start=False,
    show_end=True,
    interpolate=False,
    n_interp=1000,
    units="km",
    height=800,
    show=True,
):
    """Plot one or more 3D trajectories with optional spherical bodies.

    Parameters
    ----------
    traces : np.ndarray (N, 3) or list of np.ndarray
        One or more 3D position trajectories.
    t_arrs : np.ndarray (N,) or list of np.ndarray, optional
        Time arrays, either shared or one per trajectory.
    labels : str or list of str, optional
        Trajectory labels.
    title : str
        Figure title.
    bodies : dict or list of dict, optional
        Spherical bodies with keys:
        'name', 'radius', 'center', 'color', 'opacity'.
    t_max : float, optional
        Maximum time to display.
    show_start, show_end : bool
        Show markers at trajectory endpoints.
    interpolate : bool
        Apply cubic spline interpolation to trajectories.
    n_interp : int
        Number of points per interpolated trajectory.
    units : str
        Position units for axis labels.
    height : int
        Figure height in pixels.
    show : bool
        Display the figure immediately.

    Returns
    -------
    fig : go.Figure | None
        Plotly figure or nothing if `show=False`
    """

    fig = go.Figure()

    # --------------------------------------------------
    # 1. Normalize trajectory inputs
    # --------------------------------------------------
    if isinstance(traces, np.ndarray):
        if traces.ndim == 2:
            traces = [traces]
        elif traces.ndim == 3:
            traces = list(traces)
        else:
            raise ValueError("traces must have shape (N,3) or (K,N,3).")
    else:
        traces = list(traces)

    traces = [np.asarray(r, dtype=float) for r in traces]
    num_traces = len(traces)

    if num_traces == 0:
        raise ValueError("At least one trajectory is required.")

    # Labels
    if labels is None:
        labels = [f"Trajectory {i+1}" for i in range(num_traces)]
    elif isinstance(labels, str):
        labels = (
            [labels] if num_traces == 1
            else [f"{labels} {i+1}" for i in range(num_traces)]
        )

    if len(labels) != num_traces:
        raise ValueError("Number of labels must match trajectories.")

    # Time arrays: none, shared, or separate
    if t_arrs is None:
        times = [None] * num_traces
    elif len(t_arrs) > 0 and not np.isscalar(t_arrs[0]):
        times = [np.asarray(t) for t in t_arrs]
    else:
        times = [np.asarray(t_arrs)] * num_traces

    if len(times) != num_traces:
        raise ValueError("Number of time arrays must match trajectories.")

    if t_max is not None and t_arrs is None:
        raise ValueError("t_max requires time arrays.")

    # --------------------------------------------------
    # 2. Plot spherical bodies
    # --------------------------------------------------
    if bodies is not None:
        if isinstance(bodies, dict):
            bodies = [bodies]

        u = np.linspace(0, 2*np.pi, 60)
        v = np.linspace(0, np.pi, 30)

        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones_like(u), np.cos(v))

        for body in bodies:
            radius = float(body["radius"])
            center = np.asarray(body.get("center", [0, 0, 0]))
            color = body.get("color", "royalblue")
            opacity = body.get("opacity", 0.7)

            if radius <= 0 or center.shape != (3,):
                raise ValueError("Body requires radius > 0 and center (3,).")

            fig.add_trace(go.Surface(
                x=radius*sphere_x + center[0],
                y=radius*sphere_y + center[1],
                z=radius*sphere_z + center[2],
                name=body.get("name", "Body"),
                colorscale=[[0, color], [1, color]],
                showscale=False,
                opacity=opacity,
                showlegend=False,
                hoverinfo="skip"
            ))

    # --------------------------------------------------
    # 3. Plot trajectories
    # --------------------------------------------------
    colors = qualitative.Plotly

    for idx, (t, r, label) in enumerate(zip(times, traces, labels)):

        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError(f"{label}: expected (N,3), got {r.shape}")

        if len(r) == 0 or not np.all(np.isfinite(r)):
            raise ValueError(f"{label}: invalid trajectory data.")

        if t is None:
            t = np.arange(len(r))
        else:
            t = np.asarray(t, dtype=float)

        if len(t) != len(r):
            raise ValueError(f"{label}: time and position lengths differ.")

        if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
            raise ValueError(f"{label}: times must be finite and increasing.")

        # Clip time
        if t_max is not None:
            mask = t <= t_max
            t, r = t[mask], r[mask]

        if len(r) == 0:
            continue

        # Optional cubic spline interpolation
        if interpolate and len(r) >= 3:
            t_fine = np.linspace(t[0], t[-1], n_interp)
            r_plot = CubicSpline(t, r, axis=0)(t_fine)
        else:
            r_plot = r

        color = colors[idx % len(colors)]

        # Trajectory
        fig.add_trace(go.Scatter3d(
            x=r_plot[:, 0],
            y=r_plot[:, 1],
            z=r_plot[:, 2],
            mode="lines",
            name=label,
            legendgroup=label,
            line=dict(color=color, width=5)
        ))

        # Start marker
        if show_start:
            fig.add_trace(go.Scatter3d(
                x=[r_plot[0, 0]],
                y=[r_plot[0, 1]],
                z=[r_plot[0, 2]],
                mode="markers",
                name=f"{label} Start",
                legendgroup=label,
                showlegend=False,
                marker=dict(
                    size=6,
                    color=color,
                    symbol="diamond"
                )
            ))

        # End marker
        if show_end:
            fig.add_trace(go.Scatter3d(
                x=[r_plot[-1, 0]],
                y=[r_plot[-1, 1]],
                z=[r_plot[-1, 2]],
                mode="markers",
                name=f"{label} End",
                legendgroup=label,
                showlegend=False,
                marker=dict(size=6, color=color)
            ))

    # --------------------------------------------------
    # 4. Layout
    # --------------------------------------------------
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f"X [{units}]",
            yaxis_title=f"Y [{units}]",
            zaxis_title=f"Z [{units}]",
            aspectmode="data"
        ),
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
        height=height,
        margin=dict(l=0, r=0, b=0, t=50)
    )

    if show:
        fig.show()
        return None

    return fig

def plot_flybys(flybys: list[OrbitalFlyby], title: str, xlim = None, ylim = None, planet_num_to_colors: dict = None, scale = AU_TO_KM):

    fig, ax = plt.subplots(figsize=(9, 9))

    # Plot Sun
    ax.scatter(0, 0, color="gold", marker="o", s=100, label="Sun", zorder=5)

    # Plot waypoints
    for flyby in flybys:
        
        r = flyby.r_HCI * scale
        if planet_num_to_colors is not None:
            ax.scatter(r[0], r[1], color=planet_num_to_colors[flyby.planet_num], s=40, zorder=3, label = flyby.name)
        else:
            ax.scatter(r[0], r[1], s=40, zorder=3, label = flyby.name)
        
        # # Dumb dub dumb
        # name = flyby.name
        # if "Venus" in name:
        #     ax.annotate(name, (r[0], r[1]),xytext=(-75, 6),textcoords="offset points",fontsize=9)
        # elif "Earth Swingby" not in name:
        #     ax.annotate(name, (r[0], r[1]),xytext=(6, 6),textcoords="offset points",fontsize=9)
        # elif "#1" in name:
        #     ax.annotate("Earth Swingbys", (r[0], r[1]),xytext=(6, 6),textcoords="offset points",fontsize=9)

    ax.set_xlabel("Heliocentric X [AU]")
    ax.set_ylabel("Heliocentric Y [AU]")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend()

    plt.tight_layout()

def plot_hyperbolic_characteristics(flyby: OrbitalFlyby, color: str):
    r = flyby.r_HCI[:2] / AU_TO_KM

    scale = 0.2 / 40  # AU per (km/s)

    v_in = flyby.v_arrival_HCI[:2] * scale
    v_out = flyby.v_departure_HCI[:2] * scale

    plt.arrow(*(r - v_in), *v_in,
              width=0.005, color=color)

    plt.arrow(*r, *v_out,
              width=0.005, color=color)
###########################################################################
#                   Examples
###########################################################################

# if __name__ == "__main__":
#     plot_3d(
#         r_arr,
#         t_arrs=t_arr,
#         labels="Satellite",
#         bodies={"name": "Earth", "radius": R_EARTH},
#         title="Earth Orbit"
#     )
    
#     plot_3d(
#         [r1, r2, r3],
#         t_arrs=t_arr,
#         labels=["Trajectory 1", "Trajectory 2", "Trajectory 3"],
#         bodies={
#             "name": "Pluto",
#             "radius": R_PLUTO,
#             "color": "slategrey"
#         },
#         show_start=True,
#         show_end=True
#     )
    
#     plot_3d(
#         r_arr,
#         bodies=[
#             {
#                 "name": "Pluto",
#                 "radius": R_PLUTO,
#                 "color": "slategrey"
#             },
#             {
#                 "name": "Charon",
#                 "radius": R_CHARON,
#                 "center": [r_PLUTO_CHARON, 0, 0],
#                 "color": "lightgrey"
#             }
#         ]
#     )
    
#     fig = plot_3d(
#         relative_positions,
#         labels="Deputy",
#         title="HCW Relative Motion",
#         show_start=True,
#         show=False
#     )

#     fig.update_layout(scene_camera=dict(
#         eye=dict(x=0.5, y=0.5, z=2)
#     ))

#     fig.show()