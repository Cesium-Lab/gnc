"""TODO: organize"""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative
from scipy.interpolate import CubicSpline

from .constants import AU_TO_KM
from .orbital.interplanetary import OrbitalFlyby


def plot_sky(
    azimuth,
    elevation,
    times=None,
    *,
    elevation_mask=0.0,
    visible_mask=None,
    deg=True,
    title="Satellite sky track",
    time_label="Elapsed time",
    color="purple",
    cmap="viridis",
    figsize=(9, 8),
    show=True,
):
    """Plot a site's satellite sky track with a shaded elevation mask.

    Parameters
    ----------
    azimuth, elevation : array_like, shape (N,)
        Azimuth clockwise from north and elevation above the horizon.
        NaN angles are omitted. Compute these with geodetic_ECEF_to_SEZ.
    times : array_like, shape (N,), optional
        Numeric values used to color the track. Convert seconds to hours
        before passing them if time_label describes hours.
    elevation_mask : float, optional
        Minimum visible elevation, in the same angular units as the inputs.
        The excluded sky region is shaded gray.
    visible_mask : array_like of bool, shape (N,), optional
        Additional visibility conditions, combined with the elevation mask.
    deg : bool, optional
        Input angles and elevation_mask are degrees if True, radians otherwise.
    title, time_label : str
        Plot title and time colorbar label.
    color, cmap : str
        Marker color without times, or colormap when times are supplied.
    figsize : tuple[float, float]
        Figure width and height in inches.
    show : bool, optional
        Display immediately. The figure and axes are returned either way.

    Returns
    -------
    fig, ax : matplotlib Figure and polar Axes
        North is at the top; azimuth increases clockwise. The horizon is the
        outer circle and zenith is the center. Lines break across hidden or
        undefined samples rather than joining separate visible intervals.
    """
    az = np.asarray(azimuth, dtype=float)
    el = np.asarray(elevation, dtype=float)
    if az.ndim != 1 or el.shape != az.shape:
        raise ValueError(
            "Azimuth and elevation must be matching one-dimensional arrays"
        )
    el_deg = el if deg else np.rad2deg(el)
    mask_deg = elevation_mask if deg else np.rad2deg(elevation_mask)
    if not np.isfinite(mask_deg) or not 0 <= mask_deg <= 90:
        raise ValueError("Elevation mask must be between horizon and zenith")
    visible = np.isfinite(az) & np.isfinite(el) & (el_deg >= mask_deg) & (el_deg <= 90)
    if visible_mask is not None:
        keep = np.asarray(visible_mask)
        if keep.shape != az.shape or keep.dtype != np.bool_:
            raise ValueError("visible_mask must be a boolean array matching the angles")
        visible &= keep
    if times is not None:
        times = np.asarray(times, dtype=float)
        if times.shape != az.shape or not np.isfinite(times).all():
            raise ValueError("Times must be finite and match the angle arrays")
    theta = np.deg2rad(az) if deg else az
    radius = 90 - el_deg

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(
        np.arange(0, 360, 45), ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    )
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 30, 60, 90], labels=["90°", "60°", "30°", "0°"])
    ax.set_title(title, pad=25)
    ax.grid(True, alpha=0.4)
    mask_angles = np.linspace(0, 2 * np.pi, 361)
    ax.fill_between(mask_angles, 90 - mask_deg, 90, color="gray", alpha=0.25, zorder=0)
    ax.plot(
        mask_angles,
        np.full_like(mask_angles, 90 - mask_deg),
        color="gray",
        linestyle="--",
        linewidth=1,
    )
    ax.plot(
        np.where(visible, theta, np.nan),
        np.where(visible, radius, np.nan),
        color="gray",
        linewidth=1,
    )
    if visible.any():
        dots = ax.scatter(
            theta[visible],
            radius[visible],
            s=8,
            **(
                {"c": times[visible], "cmap": cmap}
                if times is not None
                else {"color": color}
            ),
        )
        if times is not None:
            fig.colorbar(dots, ax=ax, pad=0.1, label=time_label)
    else:
        ax.text(0.5, 0.5, "No visible samples", transform=ax.transAxes, ha="center")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_3d(
    traces=None,
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
    prev_fig=None,
    vectors=None,
    trace_colors=None,
    elev=None,
    azim=None,
    projection="ortho",
    camera_distance=None,
    scale=1.0,
    frames=None,
    points=None,
) -> go.Figure:
    """Plot 3D trajectories, spherical bodies, and vector arrows.

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
    scale : float
        Positive divisor converting input coordinates into displayed units.
        For meter inputs, use scale=AU_TO_M with units="AU", or scale=1000
        with units="km". Applies to new trajectories, body centers/radii, and
        vector origins/lengths; existing prev_fig traces and times are unchanged.
    height : int
        Figure height in pixels.
    show : bool
        Display the figure immediately.
    prev_fig : go.Figure | None
        Existing figure to add to, including its traces in the camera bounds.
    vectors : dict or list of dict, optional
        Arrows with nonzero direction (3,), optional origin (default zero),
        length (default direction magnitude, before scaling), name, color, width.
    frames : dict or list of dict, optional
        Coordinate triads with location (default zero), dcm_body_wrt_hci
        (default identity), length (default 1 in input units), and name.
        Despite the key's HCI suffix, the DCM must map components from the
        plot's coordinate frame into the displayed frame: v_body = DCM @
        v_plot. Its rows are the displayed X/Y/Z axes expressed in plot
        coordinates. For an ECI plot showing ECEF axes, pass an ECI-to-ECEF
        matrix, e.g. calc_dcm_gcrs_wrt_itrf(epoch_jd).T. Location is expressed
        in plot coordinates. Optional colors default to red/green/blue;
        width defaults to 4. Locations and lengths use scale.
    points : dict or list of dict, optional
        Named markers with location (3,) in plot-frame input units. Optional
        keys: name, color (default red), size (default 7), symbol (default
        circle), label (default True), and textposition (default top center).
        Locations are divided by scale and included in automatic axis bounds.
    trace_colors : str or list of str, optional
        One color for all trajectories, or one per trajectory.
    elev, azim : float, optional
        Camera elevation and azimuth in degrees, measured from +x toward +y.
    projection : {"ortho", "persp"}
        Camera projection.
    camera_distance : float, optional
        Positive camera distance; larger values zoom out. Orthographic zoom
        scales equal axis ranges rather than only moving the camera.

    Returns
    -------
    fig : go.Figure
        Figure, whether or not it is displayed.
    """

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    fig = prev_fig if prev_fig is not None else go.Figure()

    # --------------------------------------------------
    # 1. Normalize trajectory inputs
    # --------------------------------------------------
    if traces is None:
        traces = []
    elif isinstance(traces, np.ndarray):
        if traces.ndim == 2:
            traces = [traces]
        elif traces.ndim == 3:
            traces = list(traces)
        else:
            raise ValueError("traces must have shape (N,3) or (K,N,3).")
    else:
        traces = list(traces)

    traces = [np.asarray(r, dtype=float) / scale for r in traces]
    num_traces = len(traces)

    if isinstance(trace_colors, str):
        trace_colors = [trace_colors] * num_traces
    elif trace_colors is not None and len(trace_colors) != num_traces:
        raise ValueError("Number of trace colors must match trajectories.")

    # Labels
    if labels is None:
        labels = [f"Trajectory {i + 1}" for i in range(num_traces)]
    elif isinstance(labels, str):
        labels = (
            [labels]
            if num_traces == 1
            else [f"{labels} {i + 1}" for i in range(num_traces)]
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

        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 30)

        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones_like(u), np.cos(v))

        for body in bodies:
            radius = float(body["radius"]) / scale
            center = np.asarray(body.get("center", [0, 0, 0]), dtype=float) / scale
            color = body.get("color", "royalblue")
            opacity = body.get("opacity", 0.7)

            if radius <= 0 or center.shape != (3,):
                raise ValueError("Body requires radius > 0 and center (3,).")

            fig.add_trace(
                go.Surface(
                    x=radius * sphere_x + center[0],
                    y=radius * sphere_y + center[1],
                    z=radius * sphere_z + center[2],
                    name=body.get("name", "Body"),
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    opacity=opacity,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

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

        if t.ndim != 1:
            raise ValueError(f"{label}: times must be one-dimensional.")
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

        color = (
            trace_colors[idx] if trace_colors is not None else colors[idx % len(colors)]
        )

        # Trajectory
        fig.add_trace(
            go.Scatter3d(
                x=r_plot[:, 0],
                y=r_plot[:, 1],
                z=r_plot[:, 2],
                mode="lines",
                name=label,
                legendgroup=label,
                line={"color": color, "width": 5},
            )
        )

        # Start marker
        if show_start:
            fig.add_trace(
                go.Scatter3d(
                    x=[r_plot[0, 0]],
                    y=[r_plot[0, 1]],
                    z=[r_plot[0, 2]],
                    mode="markers",
                    name=f"{label} Start",
                    legendgroup=label,
                    showlegend=False,
                    marker={"size": 6, "color": color, "symbol": "diamond"},
                )
            )

        # End marker
        if show_end:
            fig.add_trace(
                go.Scatter3d(
                    x=[r_plot[-1, 0]],
                    y=[r_plot[-1, 1]],
                    z=[r_plot[-1, 2]],
                    mode="markers",
                    name=f"{label} End",
                    legendgroup=label,
                    showlegend=False,
                    marker={"size": 6, "color": color},
                )
            )

    # Coordinate triads reuse the same arrow geometry as other vectors.
    vector_specs = (
        []
        if vectors is None
        else ([vectors] if isinstance(vectors, dict) else list(vectors))
    )
    if frames is not None:
        for frame in [frames] if isinstance(frames, dict) else frames:
            dcm = np.asarray(frame.get("dcm_body_wrt_hci", np.eye(3)), dtype=float)
            if (
                dcm.shape != (3, 3)
                or not np.isfinite(dcm).all()
                or not np.allclose(dcm @ dcm.T, np.eye(3), atol=1e-6, rtol=0)
                or not np.isclose(np.linalg.det(dcm), 1, atol=1e-6, rtol=0)
            ):
                raise ValueError(
                    "dcm_body_wrt_hci must be a proper 3x3 rotation matrix"
                )
            frame_colors = frame.get("colors", ["red", "green", "blue"])
            if len(frame_colors) != 3:
                raise ValueError("Frame colors must contain one color per axis")
            for axis, direction, color in zip("XYZ", dcm, frame_colors):
                vector_specs.append(
                    {
                        "origin": frame.get("location", [0, 0, 0]),
                        "direction": direction,
                        "length": frame.get("length", 1.0),
                        "name": f"{frame.get('name', 'Frame')} {axis}",
                        "color": color,
                        "width": frame.get("width", 4),
                        "label": True,
                    }
                )

    # Vector shafts and arrowheads share a legend entry.
    if vector_specs:
        for idx, vector in enumerate(vector_specs):
            origin = np.asarray(vector.get("origin", [0, 0, 0]), dtype=float)
            direction = np.asarray(vector["direction"], dtype=float)
            if (
                origin.shape != (3,)
                or direction.shape != (3,)
                or not np.all(np.isfinite(origin))
                or not np.all(np.isfinite(direction))
            ):
                raise ValueError(
                    "Vector origin and direction must be finite (3,) arrays."
                )
            magnitude = np.linalg.norm(direction)
            length = float(vector.get("length", magnitude))
            if (
                not np.isfinite(magnitude)
                or magnitude == 0
                or not np.isfinite(length)
                or length <= 0
            ):
                raise ValueError(
                    "Vector direction must be nonzero and length positive and finite."
                )
            direction = direction / magnitude
            origin = origin / scale
            length = length / scale
            endpoint = origin + length * direction
            color = vector.get("color", colors[idx % len(colors)])
            name = vector.get("name", f"Vector {idx + 1}")
            group = f"vector-{len(fig.data)}"
            fig.add_trace(
                go.Scatter3d(
                    x=[origin[0], endpoint[0]],
                    y=[origin[1], endpoint[1]],
                    z=[origin[2], endpoint[2]],
                    mode="lines+text" if vector.get("label", False) else "lines",
                    text=["", name] if vector.get("label", False) else None,
                    textposition="top center",
                    name=name,
                    legendgroup=group,
                    line={"color": color, "width": vector.get("width", 5)},
                )
            )
            fig.add_trace(
                go.Cone(
                    x=[endpoint[0]],
                    y=[endpoint[1]],
                    z=[endpoint[2]],
                    u=[direction[0]],
                    v=[direction[1]],
                    w=[direction[2]],
                    anchor="tip",
                    sizemode="absolute",
                    sizeref=0.06 * length,
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    name=f"{name} arrowhead",
                    legendgroup=group,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    if points is not None:
        for index, point in enumerate([points] if isinstance(points, dict) else points):
            location = np.asarray(point["location"], dtype=float)
            if location.shape != (3,) or not np.isfinite(location).all():
                raise ValueError("Point location must be a finite (3,) array")
            location = location / scale
            name = point.get("name", f"Point {index + 1}")
            labeled = point.get("label", True)
            fig.add_trace(
                go.Scatter3d(
                    x=[location[0]],
                    y=[location[1]],
                    z=[location[2]],
                    mode="markers+text" if labeled else "markers",
                    name=name,
                    text=[name] if labeled else None,
                    textposition=point.get("textposition", "top center"),
                    marker={
                        "size": point.get("size", 7),
                        "color": point.get("color", "red"),
                        "symbol": point.get("symbol", "circle"),
                    },
                )
            )

    # Equal axis ranges keep planar trajectories and spherical bodies undistorted.
    if projection not in {"ortho", "persp"}:
        raise ValueError('projection must be "ortho" or "persp"')
    plotted_points = []
    for trace in fig.data:
        if not all(getattr(trace, axis, None) is not None for axis in "xyz"):
            continue
        xyz = [np.asarray(getattr(trace, axis), dtype=float).ravel() for axis in "xyz"]
        if len({len(a) for a in xyz}) != 1:
            continue
        points = np.column_stack(xyz)
        plotted_points.append(points[np.all(np.isfinite(points), axis=1)])
    auto_distance = max(3.0, 3.0 * (height - 50) / 800)
    if camera_distance is None:
        camera_distance = auto_distance
    if not np.isfinite(camera_distance) or camera_distance <= 0:
        raise ValueError("camera_distance must be positive and finite")
    zoom_scale = camera_distance / auto_distance if projection == "ortho" else 1.0
    axis_ranges = {}
    if plotted_points and any(len(p) for p in plotted_points):
        points = np.concatenate(plotted_points)
        low, high = points.min(axis=0), points.max(axis=0)
        center = (low + high) / 2
        half_span = max(float(np.max(high - low)) / 2, 1e-9) * 1.10 * zoom_scale
        axis_ranges = {
            f"{axis}axis": {"range": [c - half_span, c + half_span]}
            for axis, c in zip("xyz", center)
        }
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": f"X [{units}]",
            "yaxis_title": f"Y [{units}]",
            "zaxis_title": f"Z [{units}]",
            "aspectmode": "cube",
            **axis_ranges,
            "camera": _plotly_camera(
                np.rad2deg(np.arctan(1 / np.sqrt(2))) if elev is None else elev,
                45 if azim is None else azim,
                projection,
                auto_distance if projection == "ortho" else camera_distance,
            ),
        },
        template="plotly_white",
        legend={"x": 0.02, "y": 0.98, "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0},
        height=height,
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
    )
    if show:
        fig.show()
    return fig


def _plotly_camera(elev, azim, projection, distance):
    """Convert elevation and azimuth in degrees to a Plotly camera."""
    if not np.all(np.isfinite([elev, azim])):
        raise ValueError("Camera elevation and azimuth must be finite")
    elevation, azimuth = np.deg2rad([elev, azim])
    return {
        "eye": {
            "x": distance * np.cos(elevation) * np.cos(azimuth),
            "y": distance * np.cos(elevation) * np.sin(azimuth),
            "z": distance * np.sin(elevation),
        },
        "up": {"x": 0, "y": 0, "z": 1},
        "projection": {
            "type": "orthographic" if projection == "ortho" else "perspective"
        },
    }


def plot_flybys(
    flybys: list[OrbitalFlyby],
    title: str,
    xlim=None,
    ylim=None,
    planet_num_to_colors: dict = None,
    scale=AU_TO_KM,
):

    _fig, ax = plt.subplots(figsize=(9, 9))

    # Plot Sun
    ax.scatter(0, 0, color="gold", marker="o", s=100, label="Sun", zorder=5)

    # Plot waypoints
    for flyby in flybys:
        r = flyby.r_HCI * scale
        if planet_num_to_colors is not None:
            ax.scatter(
                r[0],
                r[1],
                color=planet_num_to_colors[flyby.planet_num],
                s=40,
                zorder=3,
                label=flyby.name,
            )
        else:
            ax.scatter(r[0], r[1], s=40, zorder=3, label=flyby.name)

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

    plt.arrow(*(r - v_in), *v_in, width=0.005, color=color)

    plt.arrow(*r, *v_out, width=0.005, color=color)


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
