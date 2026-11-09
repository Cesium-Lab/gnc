from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import juliandate as jd
import numpy as np
from lamberthub import izzo2015
from numpy.linalg import norm
from scipy.integrate import solve_ivp
from tqdm import tqdm

from ..constants import C_LIGHT, DAY_TO_SEC, M_TO_KM, RAD_TO_DEG
from ..math.rotations import wrap_to_2pi, wrap_to_pi
from ..math.time import greg_to_jd
from ..physics.dynamics import two_body
from ..world.constants import MU_SUN
from .ephemeris import planet_elements_and_rv
from .transformations import ECI_from_HCI


###########################################################################
#                               Flyby
###########################################################################
class OrbitalFlyby:
    def __init__(
        self,
        name: str,
        *,
        datetime_ymdhms_utc: tuple[int, int, int, int, int, float] = None,
        jd: float = None,
        planet_num: int = None,
        r_HCI: np.ndarray = None,
        mu: float = None,
    ):
        self.name = name
        self.date = datetime_ymdhms_utc
        self.planet_num = planet_num
        self.mu = mu

        if jd:
            self.jd = jd
        elif datetime_ymdhms_utc:
            self.jd = greg_to_jd(*datetime_ymdhms_utc)
        else:
            raise ValueError("Must input jd or greg_date")

        if planet_num is None:
            self.r_HCI = r_HCI
            self.v_HCI = np.zeros(3)

        self.hyperbolic_turning_angle = None
        self.hyperbolic_eccentricity = None

        self.v_departure_HCI, self.v_arrival_HCI = np.zeros(3), np.zeros(3)
        self.v_departure_PCI, self.v_arrival_PCI = np.zeros(3), np.zeros(3)

    def add_rv_HCI(self, r_HCI: np.ndarray, v_HCI: np.ndarray):
        if self.planet_num is not None:
            self.r_HCI, self.v_HCI = r_HCI, v_HCI


def sph_of_infl(r_sat: float, mu_sat: float, mu_cb: float):
    return r_sat * (mu_sat / mu_cb) ** (2 / 5)


###########################################################################
#                               Planning
###########################################################################


class InterplanetaryTransferGrid:
    """Everything in KM and KM/S. TODO: class methods for different initialization"""

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
        self.tof_grid_jd = arr_grid_jd - dep_grid_jd

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
                tof = self.tof_grid_jd[dep_i, arr_i]

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


###########################################################################
#                               Coast
###########################################################################


def calc_transfer(r1_intrl: np.ndarray, r2_intrl: np.ndarray, tof_s: float, mu=MU_SUN):
    # Time and speed of light check
    if tof_s <= 0 or (norm(r1_intrl - r2_intrl) / tof_s) > C_LIGHT:
        return np.nan, np.nan

    result = izzo2015(mu, r1_intrl, r2_intrl, float(tof_s), M=0)
    v1_intrl, v2_intrl = (
        np.array(result[0]).reshape((3,)),
        np.array(result[1]).reshape((3,)),
    )

    # Catching dumb stuff
    velocities = np.concatenate((v1_intrl, v2_intrl))
    if not np.all(np.isfinite(velocities)):
        raise ValueError("Lambert returned non-finite velocities")

    # Could be numerically weird, so "real" it out
    imag_max = np.max(np.abs(velocities.imag))
    if imag_max > 1e-9:
        raise ValueError(
            f"Lambert returned complex velocities: max |imag| = {imag_max:.3g} km/s"
        )

    return v1_intrl.real, v2_intrl.real


class InterplanetaryTrajectory:
    """Everything in M and M/S. TODO: class methods for different initialization"""

    def __init__(
        self,
        name: str,
        departure_jd: float,
        arrival_jd: float,
        planet_nums: tuple[int, int],
        resolution_days: float = 0.1,
        verbose=True,
    ):

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

        # Spacecraft
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

        t_eval_s = self.t_eval_jd * DAY_TO_SEC
        sat_coast = solve_ivp(
            lambda _, s: two_body(s, MU_SUN),
            (t_eval_s[0], t_eval_s[-1]),
            state0_hci,
            t_eval=t_eval_s,
            rtol=1e-9,
            atol=1e-12,
        )

        self.r_sat_hci = sat_coast.y.T[:, 0:3]
        self.v_sat_hci = sat_coast.y.T[:, 3:6]

        self.C3 = norm(self.v_sat_hci[0]) ** 2

        print(
            f"Position difference (m): {norm(self.r_sat_hci[-1, 0:3] - self.rv_arr_body_hci[-1, 0:3])}"
        )


###########################################################################
#                          Earth Departure
###########################################################################


def departure_characteristics(v_dep_eci: np.ndarray):
    C3 = norm(v_dep_eci) ** 2
    rla = wrap_to_2pi(np.arctan2(v_dep_eci[1], v_dep_eci[0]))
    dla = wrap_to_pi(np.arcsin(v_dep_eci[2] / norm(v_dep_eci)))
    return C3, rla, dla


@dataclass
class EarthLaunchWindow:
    """Filter a computed transfer grid; C3 in km²/s², RA/dec in degrees, dates in JD."""

    grid: InterplanetaryTransferGrid | EarthDepartureGrid

    def __post_init__(self):
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
        launch_bounds_jd=None,
        arrival_bounds_jd=None,
    ):
        """Keep inclusive (min, max) bounds; None means unbounded. Calls accumulate."""
        mask = self.mask.copy()
        for values, bounds, wraps in (
            (self.departure.C3_grid_km2, c3, False),
            (self.departure.rla_grid, rla, True),
            (self.departure.dla_grid, dla, False),
            (self.transfer.t_dep_jd[:, None], launch_bounds_jd, False),
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
        self.launch_bounds_jd = launch_bounds_jd
        self.arrival_bounds_jd = arrival_bounds_jd
        return self

    def select(self, departure_jd, arrival_jd):
        """Return nearest (departure, arrival) indices within the filtered window."""
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
        return np.where(self.mask, self.departure.C3_grid_km2, np.nan)

    @property
    def indices(self):
        return np.where(self.mask)


@dataclass
class EarthDepartureGrid:
    """Earth departure arrays indexed [departure, arrival]; km/s and degrees."""

    grid: InterplanetaryTransferGrid

    def __post_init__(self):
        # The transfer grid already subtracted Earth's velocity.
        self.v_dep_eci_km = (
            self.grid.v_dep_pci_km @ ECI_from_HCI().T
        )  # Numpy arrays are the transposed version of normal ones
        self.v_inf_dep_norm_km_s = norm(self.v_dep_eci_km, axis=2)

        self.C3_grid_km2 = self.grid.C3_grid_km2
        self.valid_mask = np.isfinite(self.C3_grid_km2) & np.all(
            np.isfinite(self.v_dep_eci_km), axis=2
        )

        self.calc_ra_dec()

    def calc_ra_dec(self):
        self.rla_grid = np.full(self.C3_grid_km2.shape, np.nan)
        self.dla_grid = np.full(self.C3_grid_km2.shape, np.nan)

        for dep_i, arr_i in zip(
            *np.where(self.valid_mask & (self.v_inf_dep_norm_km_s > 0))
        ):
            _, rla, dla = departure_characteristics(self.v_dep_eci_km[dep_i, arr_i])
            self.rla_grid[dep_i, arr_i] = rla * RAD_TO_DEG
            self.dla_grid[dep_i, arr_i] = dla * RAD_TO_DEG
