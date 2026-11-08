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

from datetime import UTC, date, datetime

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import distance_transform_edt


def _coordinates(values, date_axes):
    """Normalize numeric coordinates or calendar dates for Plotly."""
    values = np.asarray(values)
    is_date = values.dtype.kind in "MUS" or (
        values.dtype.kind == "O"
        and any(
            isinstance(v, (date, datetime, np.datetime64, str)) for v in values.flat
        )
    )
    if is_date:
        # Normalize aware Python datetimes to UTC before removing their timezone.
        if values.dtype.kind == "O":
            values = np.asarray(
                [
                    v.astimezone(UTC).replace(tzinfo=None)
                    if isinstance(v, datetime) and v.tzinfo is not None
                    else v
                    for v in values.flat
                ]
            ).reshape(values.shape)
        dates = values.astype("datetime64[ms]")
    elif date_axes:
        jd = values.astype(float)
        if not np.all(np.isfinite(jd)):
            raise ValueError("Julian dates must be finite")
        # JD 2440587.5 is 1970-01-01 00:00, rather than noon.
        dates = np.datetime64("1970-01-01", "ms") + np.rint(
            (jd - 2440587.5) * 86400000
        ).astype("timedelta64[ms]")
    else:
        return values.astype(float), False
    if np.any(np.isnat(dates)):
        raise ValueError("Date coordinates cannot contain NaT")
    return dates, True


def _plot_coordinates(values, is_date):
    return values.astype(str).tolist() if is_date else values.tolist()


def _julian_dates(values):
    coordinates, is_date = _coordinates(values, False)
    if is_date:
        return coordinates.astype("int64") / 86400000 + 2440587.5
    return coordinates


def _axis_coordinates(values, mode):
    return _coordinates(_julian_dates(values), mode == "gregorian")


def _time_of_flight(departure, arrival, values):
    """Resample each departure column without interpolating across masked gaps."""
    start = arrival[0] - departure[-1]
    end = arrival[-1] - departure[0]
    count = int(np.ceil((end - start) / np.min(np.diff(arrival)))) + 1
    duration = np.linspace(start, end, count)
    result = np.full((departure.size, count), np.nan)
    for i, dep in enumerate(departure):
        samples = arrival - dep
        # Keeping NaNs in the interpolant prevents bridging masked intervals.
        result[i] = np.interp(duration, samples, values[i], left=np.nan, right=np.nan)
    return duration, result


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
    xlabel=None,
    ylabel=None,
    zlabel="",
    colorscale="Viridis",
    coloring="fill",
    showlabels=True,
    line_width=1,
    width=1200,
    height=800,
    hovertemplate=None,
    xaxis="jd",
    yaxis="jd",
    date_axes=None,
    date_format="%Y-%m-%d",
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
    failure. ``coloring`` accepts Plotly's fill, lines, none, or heatmap modes.
    Large grids with missing values use a single contour surface with nearest
    values extended into gaps, then an opaque background-colored mask hides
    those gaps. This avoids Plotly NaN failures and section seams. Mask edges
    follow grid cells; hidden regions cover background grid lines as well.

    Datetime/date arrays, NumPy datetime64 arrays, and ISO date strings are
    accepted as inputs, alongside numeric Julian dates. ``xaxis`` accepts
    "jd" or "gregorian"; ``yaxis`` also accepts "time-of-flight" (days).
    Gregorian axes use millisecond precision. No time-scale conversion is
    performed; timezone-aware Python datetimes are normalized to UTC.
    Date axis ranges accept Julian dates or calendar dates. Time-of-flight
    ranges use days. Time-of-flight mode linearly resamples each departure's
    values onto a shared duration grid at approximately the arrival sampling
    interval, without extrapolation or interpolation across masked gaps.
    ``date_axes=True`` is a compatibility alias for two Gregorian axes.
    ``date_format`` controls date ticks and default date hover formatting.

    ``trace_kwargs`` and ``layout_kwargs`` override other Plotly settings.
    An entirely masked grid returns an empty, labeled figure. The returned
    figure supports further customization and ``write_image``/``write_html``.
    """
    if date_axes:
        xaxis = yaxis = "gregorian"
    if xaxis not in {"jd", "gregorian"}:
        raise ValueError('xaxis must be "jd" or "gregorian"')
    if yaxis not in {"jd", "gregorian", "time-of-flight"}:
        raise ValueError('yaxis must be "jd", "gregorian", or "time-of-flight"')
    dep_jd = _julian_dates(departure)
    arr_jd = _julian_dates(arrival)
    x, xdate = _axis_coordinates(dep_jd, xaxis)
    y, ydate = _axis_coordinates(arr_jd, yaxis)
    z = np.ma.asarray(values, dtype=float).filled(np.nan).copy()
    for name, coordinates in (("departure", x), ("arrival", y)):
        if coordinates.ndim != 1 or coordinates.size < 2:
            raise ValueError(f"{name} must be a one-dimensional array of length >= 2")
        numeric = (
            coordinates.astype("int64")
            if coordinates.dtype.kind == "M"
            else coordinates
        )
        if not np.all(np.isfinite(numeric)) or not np.all(np.diff(numeric) > 0):
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
    if yaxis == "time-of-flight":
        y, z = _time_of_flight(dep_jd, arr_jd, z)
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
            "title": xlabel
            if xlabel is not None
            else "Departure" + ("" if xdate else " [JD]"),
            "range": _plot_coordinates(_axis_coordinates(xrange, xaxis)[0], xdate)
            if xrange is not None
            else _plot_coordinates(x[[0, -1]], xdate),
            **({"type": "date", "tickformat": date_format} if xdate else {}),
        },
        yaxis={
            "title": ylabel
            if ylabel is not None
            else (
                "Time of flight [days]"
                if yaxis == "time-of-flight"
                else "Arrival" + ("" if ydate else " [JD]")
            ),
            "range": _plot_coordinates(_axis_coordinates(yrange, yaxis)[0], ydate)
            if yrange is not None
            else _plot_coordinates(y[[0, -1]], ydate),
            **({"type": "date", "tickformat": date_format} if ydate else {}),
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
            "x": _plot_coordinates(x, xdate),
            "y": _plot_coordinates(y, ydate),
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
            or (
                "Departure: %{x"
                + ("|" + date_format if xdate else "")
                + (
                    "}<br>Time of flight [days]: %{y"
                    if yaxis == "time-of-flight"
                    else "}<br>Arrival: %{y"
                )
                + ("|" + date_format if ydate else "")
                + "}<br>Value: %{z:.2f}<extra></extra>"
            ),
        }
        options.update(trace_kwargs or {})
        missing = ~np.isfinite(z)
        if missing.any() and max(z.shape) > 256:
            # Supply Plotly a finite surface so its internal NaN interpolation
            # cannot underflow. The mask below hides all extended values.
            nearest = distance_transform_edt(
                missing, return_distances=False, return_indices=True
            )
            surface = z[tuple(nearest)]
            options["z"] = surface.T.tolist()
            # Missing cells cannot supply meaningful hover values.
            options["customdata"] = z.T.tolist()
            options["hovertemplate"] = options["hovertemplate"].replace(
                "%{z", "%{customdata"
            )
            fig.add_trace(go.Contour(**options))
            if layout_kwargs:
                fig.update_layout(**layout_kwargs)
            background = (
                fig.layout.plot_bgcolor
                or fig.layout.template.layout.plot_bgcolor
                or "white"
            )
            fig.add_trace(
                go.Heatmap(
                    x=_plot_coordinates(x, xdate),
                    y=_plot_coordinates(y, ydate),
                    z=np.where(missing.T, 1.0, np.nan).tolist(),
                    colorscale=[[0, background], [1, background]],
                    zmin=0,
                    zmax=1,
                    showscale=False,
                    hoverinfo="skip",
                    connectgaps=False,
                    zsmooth=False,
                )
            )
        else:
            fig.add_trace(go.Contour(**options))
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    if show:
        fig.show()
    return fig
