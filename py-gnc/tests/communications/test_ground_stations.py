import matplotlib.pyplot as plt
import numpy as np
import pytest

from cesium.communications import GroundStation, GroundStationNetwork
from cesium.orbital.transformations import geodetic_LLA_to_ECEF
from cesium.world.constants import R_EARTH


def test_station_uses_geodetic_location_and_altitude():
    station = GroundStation("Site", 45, 30, 100)
    lat, lon = np.deg2rad([45, 30])
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    position = geodetic_LLA_to_ECEF(45, 30, 100) + 1000 * up
    angles, visible = station.sat_pass(np.array([np.r_[position, np.zeros(3)]]), 10)

    assert np.allclose(angles[0, 1:3], [90, 1000])  # elevation and range
    assert visible.tolist() == [True]


def test_network_windows_and_station_identity():
    stations = [GroundStation("ONE", 0, 0, 0), GroundStation("West", 0, 20, 0)]
    one = np.array([R_EARTH + 100, 0, 0])
    two = np.array([R_EARTH, R_EARTH, 0])  # Hella 45
    both = np.array([R_EARTH * 10, R_EARTH * 10, 0])  # should be in view of both
    hidden = np.array([0, 0, R_EARTH + 1000])
    dates = 2464492.5 + np.arange(6) / 1440
    network = GroundStationNetwork(stations).compute_visibility(
        np.column_stack(
            (np.array([one, both, two, two, hidden, one]), np.zeros((6, 3)))
        ),
        dates,
        [10, 20],
    )

    assert network.visible_mask.shape == (2, 6)
    assert network.az_el_rho.shape == (2, 6, 6)
    assert network.available_stations == [
        ["ONE"],
        ["ONE", "West"],
        ["West"],
        ["West"],
        [],
        ["ONE"],
    ]
    assert network.station_count.tolist() == [1, 2, 1, 1, 0, 1]
    assert network.stations_available(dates[2]) == ["West"]
    assert np.allclose(
        network.windows_jd["ONE"],
        [[dates[0], dates[1]], [dates[5], dates[5]]],  # for ONE ground station
    )
    assert np.allclose(network.windows_jd["West"], [[dates[1], dates[3]]])
    assert np.allclose(
        network.network_windows_jd,
        [[dates[0], dates[3]], [dates[5], dates[5]]],  # FULL windows
    )


def test_network_rejects_mismatched_inputs():
    network = GroundStationNetwork([GroundStation("Site", 0, 0, 0)])
    dates = np.array([2464492.5, 2464493.5])
    positions = np.array([[7e6, 0, 0, 0, 0, 0], [7e6, 0, 0, 0, 0, 0]])

    with pytest.raises(ValueError, match="threshold "):
        network.compute_visibility(positions, dates, [10, 20])
    with pytest.raises(ValueError, match="position"):  # Not enough positions for dates
        network.compute_visibility(positions[:1], dates, [10])
    with pytest.raises(ValueError, match="increasing"):  # reject increasing dates
        network.compute_visibility(positions, dates[::-1], [10])


def test_network_plot_time_axes_and_hidden_cells():
    station = GroundStation("Site", 0, 0, 0)
    dates = np.array([2464492.5, 2464492.5 + 1 / 24])
    network = GroundStationNetwork([station]).compute_visibility(
        np.array([[7e6, 0, 0, 0, 0, 0], [-7e6, 0, 0, 0, 0, 0]]),
        dates,
        [10],
    )
    for mode in ["gregorian", "jd", "hours"]:
        fig, axes = network.plot_visibility(time_axis=mode, show=False)
        heat = axes[0].collections[0].get_array()
        assert np.array_equal(np.ma.getmaskarray(heat), ~network.visible_mask)
        assert axes[1].lines[0].get_drawstyle() == "steps-mid"
        if mode == "hours":
            assert np.allclose(axes[1].lines[0].get_xdata(), [0, 1])
        plt.close(fig)


def test_network_single_sample_pass_boundaries():
    network = GroundStationNetwork([GroundStation("Site", 0, 0, 0)])
    network.t_jd = np.array([2464492.5, 2464493.5, 2464494.5])
    assert np.array_equal(
        network._windows(np.array([False, True, False])),
        [[network.t_jd[1], network.t_jd[1]]],
    )
