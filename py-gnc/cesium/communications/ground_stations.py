"""Ground-station look angles and sampled network visibility windows."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..constants import DAY_TO_SEC, HOUR_TO_SEC
from ..math.time import jd_to_greg
from ..orbital.transformations import geodetic_ECEF_to_SEZ


@dataclass
class GroundStation:
    """Station geodetic latitude/longitude in degrees and ellipsoid height in meters."""

    name: str
    lat_deg: float
    lon_deg: float
    alt_m: float

    def sat_pass(self, rv_ecef_m: np.ndarray, elevation_mask_deg: float):
        """Return azimuth/elevation/range [deg, deg, m] and a threshold visibility mask."""
        az_el_rho = np.empty((rv_ecef_m.shape[0], 6))

        for i in tqdm(range(rv_ecef_m.shape[0]), desc=f"{self.name} az/el calculation"):
            az_el_rho[i] = geodetic_ECEF_to_SEZ(
                rv_ecef_m[i,0:3], rv_ecef_m[i,3:6],
                self.lat_deg,
                self.lon_deg,
                alt_m=self.alt_m,
                deg=True,
            )

        visible_mask = np.isfinite(az_el_rho[:, 1]) & (
            az_el_rho[:, 1] >= elevation_mask_deg
        )
        return az_el_rho, visible_mask


class GroundStationNetwork:
    """Station visibility on a shared Julian-date grid, using degree thresholds.

    visible_mask[j, i] says whether station j sees the satellite at t_jd[i].
    windows_jd maps station names to (start JD, end JD) rows. Boundaries are
    first/last visible samples; no crossing-time interpolation is performed.
    """

    def __init__(self, stations: list[GroundStation]):
        self.stations = list(stations)
        self.station_names = np.array([station.name for station in self.stations])

        if not self.stations or len(set(self.station_names)) != len(self.stations):
            raise ValueError("Provide stations with unique names")

    def compute_visibility(
        self, r_ecef_m: np.ndarray, t_jd: np.ndarray, elevation_masks_deg: list[float]
    ):
        """Calculate per-station visibility; thresholds follow station order. indexed by (station, time)"""
        self.t_jd = np.asarray(t_jd, dtype=float).copy()
        r_ecef_m = np.asarray(r_ecef_m, dtype=float)
        self.elevation_masks_deg = np.asarray(elevation_masks_deg, dtype=float)
        if (
            self.t_jd.ndim != 1
            or not self.t_jd.size
            or not np.isfinite(self.t_jd).all()
            or np.any(np.diff(self.t_jd) <= 0)
        ):
            raise ValueError("Julian dates must be finite and strictly increasing")
        if r_ecef_m.shape != (len(self.t_jd), 3) or not np.isfinite(r_ecef_m).all():
            raise ValueError(
                "Provide one finite ECEF position in meters per Julian date"
            )
        if self.elevation_masks_deg.shape != (len(self.stations),):
            raise ValueError("Provide one elevation threshold per station")

        # Bounds
        if not np.isfinite(elevation_masks_deg).all() or np.any(
            (self.elevation_masks_deg < 0) | (self.elevation_masks_deg > 90)
        ):
            raise ValueError(
                f"Elevation thresholds must be between 0 and 90 degrees: {elevation_masks_deg}"
            )

        self.az_el_rho = np.empty((len(self.stations), len(self.t_jd), 3))
        self.visible_mask = np.empty((len(self.stations), len(self.t_jd)), dtype=bool)

        self.windows_jd = {}
        for station_i, station in enumerate(self.stations):
            self.az_el_rho[station_i], self.visible_mask[station_i] = station.sat_pass(
                r_ecef_m,
                self.elevation_masks_deg[station_i],
            )
            self.windows_jd[station.name] = self._windows(self.visible_mask[station_i])

        self.available_stations = [
            self.station_names[mask].tolist() for mask in self.visible_mask.T
        ]
        self.station_count = self.visible_mask.sum(axis=0)
        self.network_visible_mask = self.station_count > 0
        self.network_windows_jd = self._windows(self.network_visible_mask)
        return self

    def _windows(self, mask):
        """Return inclusive sampled JD endpoints for contiguous True runs."""
        edges = np.diff(np.r_[False, mask, False].astype(int))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1) - 1
        return np.column_stack((self.t_jd[starts], self.t_jd[ends]))

    def stations_available(self, epoch_jd):
        """Return station names available at the nearest sampled Julian date."""
        if not np.isfinite(epoch_jd) or not self.t_jd[0] <= epoch_jd <= self.t_jd[-1]:
            raise ValueError("Requested date is outside the computed interval")
        i = int(np.argmin(abs(self.t_jd - epoch_jd)))
        return self.available_stations[i]

    def plot_visibility(
        self,
        time_axis="gregorian",
        title="Ground station availability",
        figsize=(12, 6),
        cmap="tab10",
        show=True,
    ):
        """Plot visible station samples and a station-count step function.

        time_axis accepts "gregorian", "jd", or "hours" since the first sample.
        Hidden cells are NaN/transparent. Each row has its own station color.
        Cells and count steps meet halfway between samples; exact acquisition
        and loss times are not inferred from the sampled visibility masks.
        Returns the Matplotlib figure and two axes, whether displayed or not.
        """

        if time_axis not in {"gregorian", "jd", "hours"}:
            raise ValueError('time_axis must be "gregorian", "jd", or "hours"')
        dates = np.asarray(self.t_jd)
        x = (
            (dates - dates[0]) * DAY_TO_SEC / HOUR_TO_SEC
            if time_axis == "hours"
            else dates
        )
        if len(x) > 1:
            midpoints = (x[:-1] + x[1:]) / 2
            edges = np.r_[
                x[0] - (x[1] - x[0]) / 2, midpoints, x[-1] + (x[-1] - x[-2]) / 2
            ]
        else:
            half_width = 0.5 / HOUR_TO_SEC if time_axis == "hours" else 0.5 / DAY_TO_SEC
            edges = np.array([x[0] - half_width, x[0] + half_width])

        n_stations = len(self.stations)
        availability = np.where(
            self.visible_mask, np.arange(n_stations)[:, None], np.nan
        )
        colors = plt.get_cmap(cmap, n_stations).copy()
        colors.set_bad(alpha=0)
        fig, axes = plt.subplots(
            2,
            1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
            constrained_layout=True,
        )
        axes[0].pcolormesh(
            edges,
            np.arange(n_stations + 1) - 0.5,
            np.ma.masked_invalid(availability),
            cmap=colors,
            vmin=-0.5,
            vmax=n_stations - 0.5,
            shading="flat",
        )
        axes[0].set_yticks(np.arange(n_stations), labels=self.station_names)
        axes[0].invert_yaxis()
        axes[0].set_title(title)
        axes[0].set_ylabel("Station")
        axes[1].step(x, self.station_count, where="mid", color="black")
        axes[1].set_yticks(np.arange(n_stations + 1))
        axes[1].set_ylim(-0.1, n_stations + 0.1)
        axes[1].set_ylabel("Available\nstations")
        for ax in axes:
            ax.grid(True, alpha=0.25)
        axes[1].set_xlim(edges[0], edges[-1])
        axes[1].locator_params(axis="x", nbins=8)
        if time_axis == "gregorian":
            axes[1].xaxis.set_major_formatter(
                lambda epoch, _: "{:04.0f}-{:02.0f}-{:02.0f}\n{:02.0f}:{:02.0f}".format(
                    *jd_to_greg(epoch)[:5]
                )
            )
            axes[1].set_xlabel("Gregorian date/time")
        elif time_axis == "jd":
            axes[1].ticklabel_format(axis="x", style="plain", useOffset=False)
            axes[1].set_xlabel("Julian date")
        else:
            axes[1].set_xlabel("Hours since propagation start")
        if show:
            plt.show()
        return fig, axes
