"""Plot scalar departure/arrival grids with Plotly.

Example
-------
>>> fig = plot_porkchop(  # doctest: +SKIP
...     departure_jd, arrival_jd, c3_grid,
...     zmin=0, zmax=100, contour_min=0, contour_max=40, contour_step=2,
...     title="Earth-to-Mars departure C3", zlabel="C3 [km²/s²]",
... )
>>> fig.show()  # doctest: +SKIP
"""

import numpy as np
import plotly.graph_objects as go


def plot_porkchop(
    departure,
    arrival,
    values,
    *,
    mask=None,
    zmin=None,
    zmax=None,
    contour_min=None,
    contour_max=None,
    contour_step=None,
    xrange=None,
    yrange=None,
    title="Porkchop plot",
    xlabel="Departure [JD]",
    ylabel="Arrival [JD]",
    zlabel="",
    colorscale="Viridis",
    coloring="fill",
    showlabels=True,
    line_width=1,
    width=1200,
    height=800,
    hovertemplate=None,
    crop=True,
    trace_kwargs=None,
    layout_kwargs=None,
    show=False,
):
    """Return a configurable Plotly contour figure without changing the inputs.

    ``values`` has shape ``(len(departure), len(arrival))``: pass the
    departure/arrival grid directly, without transposing it. Any scalar field
    works, including C3, excess speed, right ascension, and declination.
    NaNs, infinities, and NumPy masked entries are hidden. An optional boolean
    ``mask`` of the same shape uses True for points to *keep*.

    ``zmin``/``zmax`` control the color scale only. ``contour_min``,
    ``contour_max``, and ``contour_step`` independently control contour levels;
    they do not mask data. Use ``mask=values <= limit`` to exclude values.
    Contour bounds default to the color bounds, or the finite data bounds.
    The default step divides that interval into ten parts.

    ``xrange``/``yrange`` set visible axis bounds. By default the full input
    coordinate extent is retained even when ``crop=True`` removes empty outer
    rows/columns (with one cell of padding) to avoid Plotly's NaN interpolation
    failure. Pass offset coordinates and matching labels to plot days from an
    epoch. ``coloring`` accepts Plotly's fill, lines, none, or heatmap modes.

    ``trace_kwargs`` and ``layout_kwargs`` override other Plotly settings.
    An entirely masked grid returns an empty, labeled figure. The returned
    figure supports further customization and ``write_image``/``write_html``.
    """
    x = np.asarray(departure, dtype=float)
    y = np.asarray(arrival, dtype=float)
    z = np.ma.asarray(values, dtype=float).filled(np.nan).copy()
    for name, coordinates in (("departure", x), ("arrival", y)):
        if coordinates.ndim != 1 or coordinates.size < 2:
            raise ValueError(f"{name} must be a one-dimensional array of length >= 2")
        if not np.all(np.isfinite(coordinates)) or not np.all(np.diff(coordinates) > 0):
            raise ValueError(
                f"{name} coordinates must be finite and strictly increasing"
            )
    if z.shape != (x.size, y.size):
        raise ValueError("values must have shape (len(departure), len(arrival))")
    if mask is not None:
        keep = np.asarray(mask)
        if keep.shape != z.shape or keep.dtype != np.bool_:
            raise ValueError(
                "mask must be a boolean array with the same shape as values"
            )
        z[~keep] = np.nan
    z[~np.isfinite(z)] = np.nan
    if zmin is not None and zmax is not None and zmin >= zmax:
        raise ValueError("zmin must be less than zmax")
    if contour_step is not None and (
        not np.isfinite(contour_step) or contour_step <= 0
    ):
        raise ValueError("contour_step must be finite and positive")

    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis={
            "title": xlabel,
            "range": list(xrange) if xrange is not None else [x[0], x[-1]],
        },
        yaxis={
            "title": ylabel,
            "range": list(yrange) if yrange is not None else [y[0], y[-1]],
        },
        width=width,
        height=height,
    )
    valid = np.isfinite(z)
    if valid.any():
        lo = contour_min if contour_min is not None else zmin
        hi = contour_max if contour_max is not None else zmax
        lo = float(np.nanmin(z)) if lo is None else float(lo)
        hi = float(np.nanmax(z)) if hi is None else float(hi)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
            raise ValueError(
                "contour bounds must be finite with contour_min <= contour_max"
            )
        step = contour_step if contour_step is not None else (hi - lo) / 10 or 1.0
        if crop:
            dep, arr = np.where(valid)
            ds = slice(max(0, dep.min() - 1), min(x.size, dep.max() + 2))
            ars = slice(max(0, arr.min() - 1), min(y.size, arr.max() + 2))
            x, y, z = x[ds], y[ars], z[ds, ars]
        options = {
            "x": x.tolist(),
            "y": y.tolist(),
            "z": z.T.tolist(),
            "autocontour": False,
            "contours": {
                "start": lo,
                "end": hi,
                "size": step,
                "coloring": coloring,
                "showlabels": showlabels,
            },
            "zmin": zmin,
            "zmax": zmax,
            "colorscale": colorscale,
            "line": {"width": line_width, "smoothing": 0},
            "connectgaps": False,
            "hoverongaps": False,
            "colorbar": {"title": zlabel},
            "hovertemplate": hovertemplate
            or ("Departure: %{x}<br>Arrival: %{y}<br>Value: %{z:.2f}<extra></extra>"),
        }
        options.update(trace_kwargs or {})
        fig.add_trace(go.Contour(**options))
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    if show:
        fig.show()
    return fig
