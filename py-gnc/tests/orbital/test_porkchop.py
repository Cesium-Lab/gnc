"""

CAUTION: AI SLOP FOR TESTING PLOTTING

Coordinate conversions preserve the original departure/arrival grid.
"""

import numpy as np
import pytest

pytest.importorskip("plotly")

from cesium.orbital.porkchop import plot_porkchop


@pytest.mark.parametrize("xaxis", ["jd", "gregorian"])
@pytest.mark.parametrize("yaxis", ["jd", "gregorian", "time-of-flight"])
def test_axis_modes(xaxis, yaxis):
    dep = np.array([2451544.5, 2451545.5])
    arr = np.arange(2451546.5, 2451550.5)
    tof = arr[None, :] - dep[:, None]
    fig = plot_porkchop(
        dep,
        arr,
        tof,
        xaxis=xaxis,
        yaxis=yaxis,
        crop=False,
        zmin=0,
        zmax=100,
        contour_min=0,
        contour_max=40,
    )
    trace = fig.data[0]
    if xaxis == "gregorian":
        assert trace.x[0] == "2000-01-01T00:00:00.000"
        assert fig.layout.xaxis.type == "date"
    else:
        assert trace.x[0] == dep[0]
    if yaxis == "gregorian":
        assert trace.y[0] == "2000-01-03T00:00:00.000"
    elif yaxis == "time-of-flight":
        z = np.asarray(trace.z)
        expected = np.broadcast_to(np.asarray(trace.y)[:, None], z.shape)
        np.testing.assert_allclose(z[np.isfinite(z)], expected[np.isfinite(z)])
        assert fig.layout.yaxis.title.text == "Time of flight [days]"
    else:
        assert trace.y[0] == arr[0]
    assert trace.zmax == 100
    assert trace.contours.end == 40


def test_time_of_flight_preserves_masked_gaps_and_inputs():
    dep = np.array([0.0, 1.0])
    arr = np.arange(2.0, 7.0)
    values = np.ones((2, 5))
    values[:, 2] = np.nan
    original = values.copy()
    fig = plot_porkchop(dep, arr, values, yaxis="time-of-flight", crop=False)
    z = np.asarray(fig.data[0].z)
    duration = np.asarray(fig.data[0].y)
    assert np.isnan(z[duration == 4, 0]).all()
    assert np.isnan(z[duration == 3, 1]).all()
    np.testing.assert_equal(values, original)


def test_calendar_inputs_and_independent_axis_ranges():
    fig = plot_porkchop(
        ["2000-01-01", "2000-01-02"],
        np.array(["2000-01-03", "2000-01-04"], dtype="datetime64[D]"),
        np.ones((2, 2)),
        xaxis="gregorian",
        yaxis="time-of-flight",
        xrange=[2451544.5, 2451545.5],
        yrange=[1, 5],
    )
    assert fig.layout.xaxis.range[0] == "2000-01-01T00:00:00.000"
    assert fig.layout.yaxis.range == (1.0, 5.0)
    with pytest.raises(ValueError, match="xaxis"):
        plot_porkchop([1, 2], [3, 4], np.ones((2, 2)), xaxis="time-of-flight")


def test_large_time_of_flight_grid_uses_one_surface_and_mask():
    dep = np.arange(300.0)
    arr = np.arange(100.0, 500.0)
    values = arr[None, :] - dep[:, None]
    values[:, 20] = np.nan
    fig = plot_porkchop(
        dep,
        arr,
        values,
        yaxis="time-of-flight",
        crop=False,
        contour_min=0,
        contour_max=400,
        contour_step=20,
    )
    assert len(fig.data) == 2
    surface, mask = fig.data
    assert surface.type == "contour" and mask.type == "heatmap"
    assert np.isfinite(np.asarray(surface.z)).all()
    original = np.asarray(surface.customdata)
    hidden = np.isfinite(np.asarray(mask.z))
    np.testing.assert_equal(hidden, ~np.isfinite(original))
    assert surface.contours.end == 400 and surface.contours.size == 20
    assert mask.showscale is False and mask.hoverinfo == "skip"
    for departure in (0.0, 126.0, 127.0, 253.0, 254.0, 298.0):
        col = list(surface.x).index(departure)
        row = list(surface.y).index(150.0)
        assert surface.z[row][col] == 150.0
        assert not hidden[row, col]


@pytest.mark.parametrize("xaxis", ["jd", "gregorian"])
@pytest.mark.parametrize("yaxis", ["jd", "gregorian", "time-of-flight"])
def test_named_marker_uses_selected_axis_coordinates(xaxis, yaxis):
    fig = plot_porkchop(
        [2451544.5, 2451545.5],
        [2451546.5, 2451547.5],
        np.ones((2, 2)),
        xaxis=xaxis,
        yaxis=yaxis,
        markers={"departure_jd": 2451545.5, "arrival_jd": 2451547.5, "name": "Chosen"},
    )
    marker = fig.data[-1]
    assert marker.type == "scatter" and marker.text == ("Chosen",)
    assert marker.x[0] == (
        "2000-01-02T00:00:00.000" if xaxis == "gregorian" else 2451545.5
    )
    expected_y = {
        "jd": 2451547.5,
        "gregorian": "2000-01-04T00:00:00.000",
        "time-of-flight": 2.0,
    }
    assert marker.y[0] == expected_y[yaxis]
    assert marker.zorder == 2
