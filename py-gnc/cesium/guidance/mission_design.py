"""Interplanetary mission design: transfer-grid search, launch windows, reference trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import juliandate as jd
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

from ..constants import DAY_TO_SEC, M_TO_KM, RAD_TO_DEG
from ..orbital.ephemeris import planet_elements_and_rv
from ..orbital.interplanetary import calc_transfer, hyperbolic_characteristics
from ..orbital.transformations import calc_dcm_eci_wrt_hci
from ..physics.dynamics import propagate_two_body
from ..world.constants import MU_SUN


class InterplanetaryTransferGrid:
    """Search heliocentric Lambert transfers on a departure/arrival date grid.

    Parameters
    ----------
    name : str
        Name used in progress messages.
    departure_start_jd, departure_end_jd : float
        Departure interval in Julian days; the end is excluded.
    tof_bounds : tuple[float, float]
        Inclusive minimum and maximum flight duration, in days.
    planet_nums : tuple[int, int]
        Departure and arrival planet identifiers for ``planet_elements_and_rv``.
    resolution_days : float, optional
        Spacing of both date axes in days.
    verbose : bool, optional
        Print initialization and computation summaries.

    Attributes
    ----------
    t_dep_jd, t_arr_jd : ndarray
        One-dimensional date axes. Arrival dates cover the departure interval
        shifted by the flight-duration bounds; not every date pair is valid.
    tof_grid_days : ndarray, shape (n_dep, n_arr)
        Arrival minus departure in days
    rv_dep_body_hci, rv_arr_body_hci : ndarray
        Planet states on their respective date axes, shaped (n_dep, 6) and
        (n_arr, 6). Positions are meters and velocities are m/s.
    C3_grid_km2 : ndarray, shape (n_dep, n_arr)
        Departure planet-relative excess speed squared, in km²/s². Created
        by ``compute_grid`` along with the velocity arrays.
    v_dep_pci_km, v_arr_pci_km : ndarray, shape (n_dep, n_arr, 3)
        Planet-relative excess velocities in km/s, expressed in HCI axes.
    v1_dep_hci_km, v2_arr_hci_km : ndarray, shape (n_dep, n_arr, 3)
        Heliocentric Lambert departure and arrival velocities in km/s.
    failed_transfers : list[tuple]
        Departure JD, arrival JD, and error text for caught solver failures.

    Notes
    -----
    Result grids use [departure, arrival] indexing. Excluded and failed pairs
    remain NaN. These epochs describe the heliocentric transfer; they do not
    specify liftoff, injection-burn, or planetary SOI-crossing times.
    """

    def __init__(
        self,
        name: str,
        departure_start_jd: float,
        departure_end_jd: float,
        tof_bounds: tuple,
        planet_nums: tuple[int, int],
        resolution_days: float = 1.0,
        verbose=True,
    ):
        """Build the date axes, duration grid, and SI planet states without solving transfers."""
        self.name = name
        self.verbose = verbose
        # Time
        self.departure_start_jd = departure_start_jd
        self.departure_end_jd = departure_end_jd
        self.tof_bounds = tof_bounds

        self.t_dep_jd = np.arange(
            self.departure_start_jd, self.departure_end_jd, resolution_days
        )
        self.t_arr_jd = np.arange(
            self.departure_start_jd + self.tof_bounds[0],
            self.departure_end_jd + self.tof_bounds[1],
            resolution_days,
        )
        self.n_dep = len(self.t_dep_jd)
        self.n_arr = len(self.t_arr_jd)

        self.t_tot_jd = np.sort(np.concatenate((self.t_dep_jd, self.t_arr_jd)))

        dep_grid_jd, arr_grid_jd = np.meshgrid(
            self.t_dep_jd, self.t_arr_jd, indexing="ij"
        )
        self.tof_grid_days = arr_grid_jd - dep_grid_jd

        # Planets (indexed on arrival and departure date lengths)
        self.rv_dep_body_hci = np.array(
            [
                np.ravel(planet_elements_and_rv(planet_nums[0], t)[1:])
                for t in self.t_dep_jd
            ]
        )
        self.rv_arr_body_hci = np.array(
            [
                np.ravel(planet_elements_and_rv(planet_nums[1], t)[1:])
                for t in self.t_arr_jd
            ]
        )

        if verbose:
            print(f"'{name}' - Initialized with:")
            print(
                f"\tSearch space of: ({self.n_dep}, {self.n_arr}) with {resolution_days} days resolution"
            )

    def compute_grid(self):
        # Solutions
        """Populate C3 and velocity grids with zero-revolution Lambert solutions.

        Only pairs within ``self.tof_bounds`` are solved. Each call replaces the
        previous result arrays and failure list. Caught ``ValueError`` failures
        are recorded in ``failed_transfers``; unsolved entries retain NaN.
        Planet states and Lambert inputs use SI units; stored transfer velocities
        are converted to km/s and C3 to km²/s².

        Returns
        -------
        None
            Results are stored on this object.
        """
        self.C3_grid_km2 = np.full((self.n_dep, self.n_arr), np.nan)

        self.v_dep_pci_km = np.full(
            (self.n_dep, self.n_arr, 3), np.nan
        )  # subtracting body velocities
        self.v_arr_pci_km = np.full(
            (self.n_dep, self.n_arr, 3), np.nan
        )  # subtracting body velocities

        self.v1_dep_hci_km = np.full(
            (self.n_dep, self.n_arr, 3), np.nan
        )  # from lambert
        self.v2_arr_hci_km = np.full(
            (self.n_dep, self.n_arr, 3), np.nan
        )  # from lambert

        # Calculation
        self.failed_transfers = []

        if self.verbose:
            print(f"`{self.name}` - Computing grid:")
        for dep_i in tqdm(range(self.n_dep)):
            rv_dep = self.rv_dep_body_hci[dep_i]

            for arr_i in range(self.n_arr):
                rv_arr = self.rv_arr_body_hci[arr_i]
                tof = self.tof_grid_days[dep_i, arr_i]

                if not self.tof_bounds[0] <= tof <= self.tof_bounds[1]:
                    continue
                try:
                    v1, v2 = calc_transfer(rv_dep[:3], rv_arr[:3], tof * DAY_TO_SEC)
                except ValueError as e:
                    self.failed_transfers.append(
                        (self.t_dep_jd[dep_i], self.t_arr_jd[arr_i], str(e))
                    )
                    print(
                        f"Failed transfer: departure JD {self.t_dep_jd[dep_i]}, arrival JD {self.t_arr_jd[arr_i]}: {e}"
                    )
                    continue

                v1_hci = v1 - rv_dep[3:6]
                v2_hci = v2 - rv_arr[3:6]

                C3 = norm(v1_hci) ** 2
                self.C3_grid_km2[dep_i, arr_i] = C3 * M_TO_KM * M_TO_KM
                self.v_dep_pci_km[dep_i, arr_i] = v1_hci * M_TO_KM
                self.v_arr_pci_km[dep_i, arr_i] = v2_hci * M_TO_KM

                self.v1_dep_hci_km[dep_i, arr_i] = v1 * M_TO_KM
                self.v2_arr_hci_km[dep_i, arr_i] = v2 * M_TO_KM

        if self.verbose and len(self.failed_transfers) > 1:
            print(
                f"{len(self.failed_transfers)} failed transfers; see failed_transfers for dates and errors."
            )


class InterplanetaryTrajectory:
    """Propagate one heliocentric transfer and sample both planets with it.

    Parameters
    ----------
    name : str
        Name used in printed summaries.
    departure_jd, arrival_jd : float
        Heliocentric Lambert reference epochs in Julian days. Arrival must
        follow departure; these are not surface-launch or SOI-crossing epochs.
    planet_nums : tuple[int, int]
        Departure and arrival planet identifiers for the existing ephemeris.
    resolution_days : float, optional
        Positive sampling interval in days. The exact arrival epoch is
        appended, so the final interval can be shorter.
    verbose : bool, optional
        Print dates and resolution during initialization.

    Attributes
    ----------
    t_eval_jd : ndarray, shape (N,)
        Shared sample dates, including exact departure and arrival.
    tof_s : float
        Requested arrival minus departure in seconds.
    rv_dep_body_hci, rv_arr_body_hci : ndarray, shape (N, 6)
        Planet position/velocity states in HCI, in meters and m/s.
    r_sat_hci, v_sat_hci : ndarray, shape (N, 3)
        Propagated spacecraft positions in meters and velocities in m/s;
        populated by ``launch_that_babyy`` when integration reaches every sample.
    C3 : float
        Currently stores squared planet-relative departure speed in m²/s².

    Notes
    -----
    Initialization samples ephemerides; propagation is a separate operation.
    The spacecraft coast model includes solar two-body gravity only.
    """

    def __init__(
        self,
        name: str,
        departure_jd: float,
        arrival_jd: float,
        planet_nums: tuple[int, int],
        resolution_days: float = 0.1,
        verbose=True,
    ):
        """Sample both planets on a shared date grid that includes the exact arrival epoch."""
        self.name = name
        self.verbose = verbose

        # Time
        self.departure_jd = departure_jd
        self.arrival_jd = arrival_jd
        self.tof_s = (arrival_jd - departure_jd) * DAY_TO_SEC

        if (
            not np.isfinite(resolution_days)
            or resolution_days <= 0
            or arrival_jd <= departure_jd
        ):
            raise ValueError("Need positive resolution and arrival after departure")

        self.t_eval_jd = np.append(
            departure_jd + np.arange(0, arrival_jd - departure_jd, resolution_days),
            arrival_jd,  # Include the exact Mars arrival epoch.
        )
        self.N = len(self.t_eval_jd)

        # Planets (indexed on arrival and departure date lengths)
        self.rv_dep_body_hci = np.array(
            [
                np.ravel(planet_elements_and_rv(planet_nums[0], t)[1:])
                for t in self.t_eval_jd
            ]
        )
        self.rv_arr_body_hci = np.array(
            [
                np.ravel(planet_elements_and_rv(planet_nums[1], t)[1:])
                for t in self.t_eval_jd
            ]
        )

        if verbose:
            print(f"'{name}' - Initialized trajectory with:")
            d1 = datetime(*jd.to_gregorian(departure_jd))
            d2 = datetime(*jd.to_gregorian(arrival_jd))
            print(f"\tDeparture: \t{d1.strftime('%Y/%m/%d\t%T')}")
            print(f"\tArrival: \t{d2.strftime('%Y/%m/%d\t%T')}")
            print(f"\tResolution: {resolution_days} days")

    def launch_that_babyy(self):
        """Solve the endpoint transfer and propagate the spacecraft under solar gravity.

        Uses ``calc_transfer`` to obtain the initial heliocentric velocity, then
        integrates ``two_body`` with ``MU_SUN`` at ``t_eval_jd``. Stores position
        and velocity arrays in meters and m/s and prints the arrival position
        residual in meters. The current ``C3`` attribute Earth-relative excess speed squared.

        Returns
        -------
        None
            The propagated arrays and ``C3`` are assigned to this object.
        """
        try:
            v1, v2 = calc_transfer(
                self.rv_dep_body_hci[0, 0:3],
                self.rv_arr_body_hci[-1, 0:3],
                self.tof_s,
                MU_SUN,
            )
            if np.isnan(v1).any() or np.isnan(v2).any():
                raise ValueError("One of v1 or v2 was nan")
        except ValueError as e:
            print(f"Failed transfer: {e}")

        state0_hci = np.concatenate((self.rv_dep_body_hci[0, 0:3], v1))

        # Relative to departure: absolute JD seconds (~2e11 s) waste float precision.
        t_eval_s = (self.t_eval_jd - self.t_eval_jd[0]) * DAY_TO_SEC
        rv_sat_hci = propagate_two_body(state0_hci, t_eval_s, MU_SUN)

        self.r_sat_hci = rv_sat_hci[:, 0:3]
        self.v_sat_hci = rv_sat_hci[:, 3:6]

        self.C3 = norm(self.v_sat_hci[0] - self.rv_dep_body_hci[0, 0:3]) ** 2

        print(
            f"Position difference (m): {norm(self.r_sat_hci[-1, 0:3] - self.rv_arr_body_hci[-1, 0:3])}"
        )


@dataclass
class EarthLaunchWindow:
    """Maintain a feasible subset of an Earth-departure transfer grid.

    Parameters
    ----------
    grid : InterplanetaryTransferGrid or EarthDepartureGrid
        A computed transfer grid, or its already calculated Earth-departure
        characteristics. An EarthDepartureGrid is constructed when needed.

    Attributes
    ----------
    transfer : InterplanetaryTransferGrid
        Underlying date axes, planet states, and Lambert results.
    departure : EarthDepartureGrid
        C3, equatorial excess velocity, RLA, and DLA used for filtering.
    mask : ndarray of bool, shape (n_dep, n_arr)
        Starts with finite transfer solutions and narrows with each filter.

    Notes
    -----
    Indices returned by this object refer to ``transfer``, not to compacted
    arrays. Filtering leaves the underlying transfer results unchanged.
    """

    grid: InterplanetaryTransferGrid | EarthDepartureGrid

    def __post_init__(self):
        """Attach Earth departure characteristics and copy their valid-solution mask."""
        self.departure = (
            self.grid
            if isinstance(self.grid, EarthDepartureGrid)
            else EarthDepartureGrid(self.grid)
        )
        self.transfer = self.departure.grid
        self.mask = self.departure.valid_mask.copy()

    def filter(
        self,
        *,
        c3=None,
        rla=None,
        dla=None,
        departure_bounds_jd=None,
        arrival_bounds_jd=None,
    ) -> EarthLaunchWindow:
        """Narrow the current selection using inclusive lower/upper bounds.

        Parameters
        ----------
        c3 : tuple[float | None, float | None], optional
            Characteristic-energy bounds in km²/s².
        rla, dla : tuple[float | None, float | None], optional
            Right ascension and declination bounds in degrees. RLA bounds with
            lower > upper cross zero, e.g. (350, 10). Reversed bounds for other
            quantities are invalid.
        departure_bounds_jd, arrival_bounds_jd : tuple[float | None, float | None], optional
            Bounds on the heliocentric departure and arrival reference dates in
            Julian days

        Returns
        -------
        EarthLaunchWindow
            This object, allowing chained filter calls.

        Raises
        ------
        ValueError
            If a bound is NaN or a non-RLA lower bound exceeds its upper bound.

        Notes
        -----
        Omitted filters do nothing, and None endpoints are unbounded. Calls
        accumulate rather than reset the mask. Stored bound attributes reflect
        arguments to the latest call, not the intersection of all prior calls.
        """
        mask = self.mask.copy()
        for values, bounds, wraps in (
            (self.departure.C3_grid_km2, c3, False),
            (self.departure.rla_grid, rla, True),
            (self.departure.dla_grid, dla, False),
            (self.transfer.t_dep_jd[:, None], departure_bounds_jd, False),
            (self.transfer.t_arr_jd[None, :], arrival_bounds_jd, False),
        ):
            if bounds is None:
                continue
            low, high = bounds
            low = -np.inf if low is None else low
            high = np.inf if high is None else high
            if np.isnan(low) or np.isnan(high):
                raise ValueError("Bounds cannot be NaN")
            if low > high and not wraps:
                raise ValueError("Lower bound must not exceed upper bound")
            # RA may cross zero, e.g. (350, 10) keeps 350–360 and 0–10 degrees.
            inside = (
                ((values >= low) | (values <= high))
                if low > high
                else ((values >= low) & (values <= high))
            )
            mask &= np.isfinite(values) & inside
        self.mask = mask

        self.c3_bounds = c3
        self.rla_bounds = rla
        self.dla_bounds = dla
        self.dla_bounds = dla
        self.departure_bounds_jd = departure_bounds_jd
        self.arrival_bounds_jd = arrival_bounds_jd
        return self

    def select(self, departure_jd, arrival_jd):
        """Find the nearest sampled pair and require it to pass the current mask.

        Parameters
        ----------
        departure_jd, arrival_jd : float
            Requested Julian dates; they need not lie exactly on sample points.

        Returns
        -------
        dep_i, arr_i : int
            Indices into ``transfer`` date axes and [departure, arrival] arrays.

        Raises
        ------
        ValueError
            If a date is non-finite, is outside its sampled axis (allowing 1e-6
            day endpoint tolerance), or the nearest pair is excluded.

        Notes
        -----
        Returns the nearest pair, not the nearest feasible alternative. Use the
        returned indices to inspect the actual sampled dates before propagation.
        """
        for requested, dates in (
            (departure_jd, self.transfer.t_dep_jd),
            (arrival_jd, self.transfer.t_arr_jd),
        ):
            if (
                not np.isfinite(requested)
                or not dates[0] - 1e-6 <= requested <= dates[-1] + 1e-6
            ):
                raise ValueError("Requested date is outside the sampled grid")
        dep_i = int(np.argmin(abs(self.transfer.t_dep_jd - departure_jd)))
        arr_i = int(np.argmin(abs(self.transfer.t_arr_jd - arrival_jd)))
        if not self.mask[dep_i, arr_i]:
            raise ValueError(
                "Selected transfer is excluded by the launch-window filters"
            )
        return dep_i, arr_i

    def arrival_bounds_for_departure(self, departure_jd):
        """Return range of feasible arrival JDs for the nearest departure.

        Parameters
        ----------
        departure_jd : float
            Requested departure Julian date, snapped to the nearest sample.

        Returns
        -------
        arrival_start_jd, arrival_end_jd : float
            Minimum and maximum arrival dates retained by that departure row's
            mask. THESE ENDPOINTS MAY ENCLOSE GAPS OF INFEASIBLE ARRIVALS.

        Raises
        ------
        ValueError
            If departure is non-finite or outside the sampled grid, or no
            arrivals remain feasible for the selected departure.
        """
        if (
            not np.isfinite(departure_jd)
            or not self.transfer.t_dep_jd[0] - 1e-6
            <= departure_jd
            <= self.transfer.t_dep_jd[-1] + 1e-6
        ):
            raise ValueError("Departure is outside the sampled grid")
        dep_i = np.argmin(abs(self.transfer.t_dep_jd - departure_jd))
        arrivals = self.transfer.t_arr_jd[self.mask[dep_i]]
        if not arrivals.size:
            raise ValueError("No feasible arrivals for this departure")
        return arrivals.min(), arrivals.max()

    @property
    def C3_grid_km2(self):
        """Return a new [departure, arrival] C3 grid in km²/s², with excluded entries NaN."""
        return np.where(self.mask, self.departure.C3_grid_km2, np.nan)

    @property
    def indices(self):
        """Return paired departure/arrival index arrays for the currently retained transfers."""
        return np.where(self.mask)


@dataclass
class EarthDepartureGrid:
    """Derive ECI equatorial departure characteristics from a transfer grid.

    Parameters
    ----------
    grid : InterplanetaryTransferGrid
        A computed grid whose departure planet is Earth. Its planet-relative
        excess velocities must be in km/s and expressed in HCI axes.

    Attributes
    ----------
    v_dep_eci_km : ndarray, shape (n_dep, n_arr, 3)
        Outgoing Earth-relative excess velocities in ECI axes, in km/s.
    v_inf_dep_norm_km_s : ndarray, shape (n_dep, n_arr)
        Excess-speed magnitudes in km/s.
    C3_grid_km2 : ndarray, shape (n_dep, n_arr)
        Reference to the underlying grid's C3 array in km²/s².
    rla_grid, dla_grid : ndarray, shape (n_dep, n_arr)
        Asymptote right ascension and declination in degrees. Invalid or
        zero-speed directions retain NaN.
    valid_mask : ndarray of bool, shape (n_dep, n_arr)
        True where C3 and the excess-velocity vector are finite.

    Notes
    -----
    Characteristics are calculated on initialization. Recreate this object
    if the underlying grid is recomputed; the arrays are not live properties.
    """

    grid: InterplanetaryTransferGrid

    def __post_init__(self):
        # The transfer grid already subtracted Earth's velocity.
        """Rotate Earth-relative excess velocities into ECI and initialize angular grids."""
        self.v_dep_eci_km = (
            self.grid.v_dep_pci_km @ calc_dcm_eci_wrt_hci().T
        )  # Numpy arrays are the transposed version of normal ones
        self.v_inf_dep_norm_km_s = norm(self.v_dep_eci_km, axis=2)

        self.C3_grid_km2 = self.grid.C3_grid_km2
        self.valid_mask = np.isfinite(self.C3_grid_km2) & np.all(
            np.isfinite(self.v_dep_eci_km), axis=2
        )

        self.calc_ra_dec()

    def calc_ra_dec(self):
        """Recalculate RLA/DLA in degrees using ``hyperbolic_characteristics``.

        Only valid, nonzero excess-velocity entries are evaluated. Replaces
        ``rla_grid`` and ``dla_grid``; all other entries remain NaN.
        """
        self.rla_grid = np.full(self.C3_grid_km2.shape, np.nan)
        self.dla_grid = np.full(self.C3_grid_km2.shape, np.nan)

        for dep_i, arr_i in zip(
            *np.where(self.valid_mask & (self.v_inf_dep_norm_km_s > 0))
        ):
            _, rla, dla = hyperbolic_characteristics(self.v_dep_eci_km[dep_i, arr_i])
            self.rla_grid[dep_i, arr_i] = rla * RAD_TO_DEG
            self.dla_grid[dep_i, arr_i] = dla * RAD_TO_DEG
