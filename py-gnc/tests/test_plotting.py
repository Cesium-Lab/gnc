"""

CAUTION: AI SLOP FOR TESTING PLOTTING

3D plotting geometry, camera controls, and existing figure support.
"""

import numpy as np
import pytest

go = pytest.importorskip("plotly.graph_objects")
pytest.importorskip("matplotlib")

from cesium.plotting import plot_3d, plot_sky


def test_vector_only_geometry_and_shared_legend():
    fig = plot_3d(
        vectors={"origin": [1, 2, 3], "direction": [0, 3, 4], "length": 10},
        show=False,
    )
    shaft, head = fig.data
    np.testing.assert_allclose([shaft.x[-1], shaft.y[-1], shaft.z[-1]], [1, 8, 11])
    np.testing.assert_allclose([head.u[0], head.v[0], head.w[0]], [0, 0.6, 0.8])
    assert shaft.legendgroup == head.legendgroup
    assert head.showlegend is False


def test_existing_figure_colors_clipping_and_interpolation():
    previous = go.Figure(go.Scatter3d(x=[-10], y=[0], z=[0]))
    positions = np.column_stack([np.arange(4.0), np.zeros((4, 2))])
    fig = plot_3d(
        positions,
        t_arrs=np.arange(4.0),
        t_max=2,
        interpolate=True,
        n_interp=11,
        trace_colors="red",
        show_start=True,
        prev_fig=previous,
        show=False,
    )
    assert fig is previous
    trajectory, start, end = fig.data[1:]
    assert len(trajectory.x) == 11
    assert trajectory.x[-1] == pytest.approx(2)
    assert trajectory.line.color == start.marker.color == end.marker.color == "red"
    assert fig.layout.scene.xaxis.range[0] < -10


def test_camera_direction_and_orthographic_zoom():
    positions = np.array([[0.0, 0, 0], [100.0, 2, 0]])
    base = plot_3d(positions, elev=0, azim=90, camera_distance=3, show=False)
    zoomed = plot_3d(positions, elev=0, azim=90, camera_distance=6, show=False)
    scene = base.layout.scene
    assert scene.camera.projection.type == "orthographic"
    assert scene.camera.eye.x == pytest.approx(0, abs=1e-12)
    assert scene.camera.eye.y == pytest.approx(3)
    spans = [np.ptp(getattr(scene, f"{axis}axis").range) for axis in "xyz"]
    np.testing.assert_allclose(spans, spans[0])
    assert np.ptp(zoomed.layout.scene.xaxis.range) == pytest.approx(2 * spans[0])
    perspective = plot_3d(positions, projection="persp", camera_distance=6, show=False)
    assert perspective.layout.scene.camera.projection.type == "perspective"
    eye = perspective.layout.scene.camera.eye
    assert np.linalg.norm([eye.x, eye.y, eye.z]) == pytest.approx(6)


def test_bodies_only_and_show_returns_figure(monkeypatch):
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self: shown.append(self))
    fig = plot_3d(bodies={"radius": 2, "center": [10, 0, 0]})
    assert shown == [fig]
    assert fig.data[0].type == "surface"
    assert fig.layout.scene.xaxis.range[0] < 8
    assert fig.layout.scene.xaxis.range[1] > 12


def test_scale_divides_new_geometry_without_changing_inputs_or_previous_traces():
    positions = np.array([[1000.0, 0, 0], [2000.0, 0, 0]])
    original = positions.copy()
    previous = go.Figure(go.Scatter3d(x=[-1], y=[0], z=[0]))
    fig = plot_3d(
        positions,
        scale=1000,
        units="km",
        prev_fig=previous,
        show_end=False,
        bodies={"radius": 1000, "center": [3000, 0, 0]},
        vectors={"origin": [1000, 0, 0], "direction": [0, 3000, 4000]},
        show=False,
    )
    existing, sphere, trajectory, shaft, head = fig.data
    np.testing.assert_equal(existing.x, [-1])
    np.testing.assert_allclose(trajectory.x, [1, 2])
    assert np.max(sphere.z) == pytest.approx(1)
    assert np.max(sphere.x) > 3.9
    np.testing.assert_allclose([shaft.x[-1], shaft.y[-1], shaft.z[-1]], [1, 3, 4])
    assert head.sizeref == pytest.approx(0.3)
    assert fig.layout.scene.xaxis.title.text == "X [km]"
    np.testing.assert_equal(positions, original)


def test_frame_axes_use_dcm_rows_and_share_vector_scaling():
    from cesium.math.rotations import dcm_z

    dcm = dcm_z(np.pi / 2)
    fig = plot_3d(
        frames={
            "location": [1000, 2000, 3000],
            "dcm_body_wrt_hci": dcm,
            "length": 2000,
            "name": "ECI",
        },
        vectors={"direction": [1000, 0, 0]},
        scale=1000,
        show=False,
    )
    assert len(fig.data) == 8
    for i, axis in enumerate("XYZ"):
        shaft, head = fig.data[2 + 2 * i : 4 + 2 * i]
        np.testing.assert_allclose(
            [shaft.x[-1], shaft.y[-1], shaft.z[-1]], np.array([1, 2, 3]) + 2 * dcm[i]
        )
        assert shaft.name == f"ECI {axis}"
        assert shaft.text[-1] == f"ECI {axis}"
        assert head.sizeref == pytest.approx(0.12)


@pytest.mark.parametrize("dcm", [np.ones((3, 3)), np.diag([1, 1, -1]), np.eye(2)])
def test_invalid_frame_rotations(dcm):
    with pytest.raises(ValueError, match="rotation matrix"):
        plot_3d(frames={"dcm_body_wrt_hci": dcm}, show=False)


def test_named_points_scaled_and_included_in_bounds():
    location = np.array([1000.0, 2000.0, 3000.0])
    fig = plot_3d(
        points=[
            {"location": location, "name": "Cape", "color": "red"},
            {"location": [4000.0, 0, 0], "name": "Periapsis", "label": False},
        ],
        scale=1000,
        show=False,
    )
    cape, periapsis = fig.data
    assert np.allclose([cape.x[0], cape.y[0], cape.z[0]], [1, 2, 3])
    assert cape.name == "Cape" and cape.text == ("Cape",)
    assert cape.marker.color == "red"
    assert periapsis.mode == "markers" and periapsis.text is None
    assert fig.layout.scene.xaxis.range[1] > 4
    assert np.allclose(location, [1000, 2000, 3000])


def test_sky_plot_visibility_and_coordinates():
    import matplotlib.pyplot as plt

    fig, ax = plot_sky(
        [0, 90, 180, 270],
        [0, 45, 90, 5],
        [0, 1, 2, 3],
        elevation_mask=10,
        visible_mask=[True, True, False, True],
        show=False,
    )
    assert ax.get_theta_direction() == -1
    assert np.allclose(ax.get_ylim(), [0, 90])
    track = ax.lines[-1]
    assert np.isnan(track.get_xdata()[0])
    assert np.allclose(track.get_xdata()[1], np.pi / 2)
    assert np.isnan(track.get_xdata()[2])
    assert np.allclose(track.get_ydata()[1], 45)
    assert len(fig.axes) == 2
    assert np.allclose(ax.collections[-1].get_array(), [1])
    plt.close(fig)


def test_sky_plot_radians_and_empty_visibility():
    import matplotlib.pyplot as plt

    fig, ax = plot_sky(
        [np.pi / 2], [np.pi / 4], deg=False, elevation_mask=np.pi / 6, show=False
    )
    assert np.allclose(ax.lines[-1].get_xdata(), [np.pi / 2])
    assert np.allclose(ax.lines[-1].get_ydata(), [45])
    plt.close(fig)
    fig, ax = plot_sky([0], [-10], show=False)
    assert ax.texts[-1].get_text() == "No visible samples"
    plt.close(fig)


@pytest.mark.parametrize(
    "options",
    [
        {"vectors": {"direction": [0, 0, 0]}},
        {"projection": "invalid"},
        {"camera_distance": 0},
        {"azim": np.nan},
        {"scale": 0},
        {"scale": -1},
        {"scale": np.nan},
    ],
)
def test_invalid_options(options):
    with pytest.raises(ValueError):
        plot_3d(show=False, **options)
