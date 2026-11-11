"""Earth departure: parking orbit, escape burn, and midcourse correction."""

import numpy as np
from numpy.linalg import norm

from ..constants import DAY_TO_SEC, HOUR_TO_SEC, KM_TO_M, M_TO_KM
from ..math.base import unit
from ..math.time import jd_to_greg
from ..orbital.ephemeris import planet_elements_and_rv
from ..orbital.interplanetary import sph_of_infl
from ..orbital.transformations import calc_dcm_eci_wrt_hci, calc_dcm_gcrs_wrt_itrf
from ..physics.dynamics import propagate_two_body
from ..world.constants import MU_EARTH, MU_SUN
from .interplanetary import interplanetary_circ_injection
from .maneuvers import Burn


class EarthDepartureTrajectory:
    def __init__(
        self,
        launch_date_jd: float,
        v_inf_eci_km: np.ndarray,
        periapse_m: float,
        r_site_ecef_m: np.ndarray,
    ):
        self.dcm_ecef_wrt_eci0 = calc_dcm_gcrs_wrt_itrf(launch_date_jd).T
        self.launch_date_jd = launch_date_jd
        self.v_inf_eci_km = v_inf_eci_km
        self.periapse_m = periapse_m
        self.r_site_ecef_m = r_site_ecef_m
        print(f"Launching on {jd_to_greg(launch_date_jd)}")

        # Departure characteristics 2D and 3D
        self.dv_periapse_m, self.ta_asymptote_rad, self.aiming_radius_m, self.e = (
            interplanetary_circ_injection(
                norm(v_inf_eci_km * KM_TO_M), periapse_m, MU_EARTH, deg=False
            )
        )
        self.v_vec = unit(v_inf_eci_km)
        self.n_vec = unit(
            np.cross(v_inf_eci_km, self.dcm_ecef_wrt_eci0.T @ r_site_ecef_m)
        )
        self.right_vec = unit(np.cross(self.n_vec, v_inf_eci_km))
        self.dcm_eci_wrt_orbit = np.stack(
            (self.v_vec, self.right_vec, self.n_vec), axis=1
        )
        if abs(np.linalg.det(self.dcm_eci_wrt_orbit) - 1.0) > 1e-9:
            print(
                f"dcm_eci_wrt_orbit det > 1: ({np.linalg.det(self.dcm_eci_wrt_orbit)})"
            )

        r_p_orbit = (
            np.array([np.cos(self.ta_asymptote_rad), -np.sin(self.ta_asymptote_rad), 0])
            * periapse_m
        )
        self.r_p_eci = self.dcm_eci_wrt_orbit @ r_p_orbit
        along_track_eci = unit(np.cross(self.n_vec, self.r_p_eci))
        self.v_p_eci = along_track_eci * (
            np.sqrt(MU_EARTH / norm(r_p_orbit)) + self.dv_periapse_m
        )

        # Tangential escape burn from the circular parking orbit; t = 0 is this burn.
        self.burns = [
            Burn(
                t_s=0.0,
                r=self.r_p_eci,
                dv=along_track_eci * self.dv_periapse_m,
                frame="ECI",
            )
        ]

    @property
    def dv_total(self) -> float:
        """Sum of burn magnitudes [m/s]."""
        return sum(burn.dv_mag for burn in self.burns)

    def launch_that_babyy(self, tspan_s: float = 12 * HOUR_TO_SEC, npoints=100000):
        rv_eci_0 = np.concatenate((self.r_p_eci, self.v_p_eci))

        self.t_traj_s = np.linspace(0, tspan_s, npoints)
        self.rv_traj_eci_m = propagate_two_body(rv_eci_0, self.t_traj_s, MU_EARTH)

        return self.rv_traj_eci_m

    def evaluate(self, verbose=True):
        self.impact_parameter = norm(
            np.cross(self.rv_traj_eci_m[:, 0:3], self.rv_traj_eci_m[:, 3:6]), axis=1
        ) / norm(self.v_inf_eci_km * KM_TO_M)
        diff = self.impact_parameter - self.aiming_radius_m
        if verbose:
            print("evaluate()")
            print("  Perpendicular distance to v_inf comparison (km)")
            print(
                f"  h/v_inf: {self.impact_parameter.mean() * M_TO_KM:.6f} ({self.impact_parameter.std() * M_TO_KM:.6f} km s.d. very small)"
            )
            print(
                f"  Theoretical aiming radius [km]: {self.aiming_radius_m * M_TO_KM:.6f}"
            )
            print(f"  diff [m]: {diff.mean()} ({diff.std()} s.d. very small)")

    def midcourse_correction(
        self,
        lambert_dep_jd: np.ndarray,
        time_s: np.ndarray,
        rv_sat_eci_km: np.ndarray,
        rv_earth_hci_km: np.ndarray,
        verbose=True,
    ):
        # Nearest Lambert position in the same Earth-relative ECI axes (but the index)
        lambert_distances_km = norm(
            rv_sat_eci_km[:, 0:3] - (self.rv_traj_eci_m[-1, 0:3] * M_TO_KM), axis=1
        )  # distances from lam
        closest_lambert_i = int(np.argmin(lambert_distances_km))

        ##### From `interplanetary.ipynb`
        v_lambert_hci_km = (
            rv_earth_hci_km[closest_lambert_i, 3:6]
            + calc_dcm_eci_wrt_hci().T @ rv_sat_eci_km[closest_lambert_i, 3:6]
        )
        # v_lambert_hci_km = mission["satellite_rv_hci_km"][closest_lambert_i, 3:6]

        ##### From this notebook
        # Add Earth's velocity at the propagated endpoint to get actual HCI velocity.
        final_epoch_jd = self.launch_date_jd + self.t_traj_s[-1] / DAY_TO_SEC
        v_earth_final_hci_km = planet_elements_and_rv(3, final_epoch_jd)[2] * M_TO_KM

        v_final_hci_km = (
            calc_dcm_eci_wrt_hci().T @ (self.rv_traj_eci_m[-1, 3:6] * M_TO_KM)
            + v_earth_final_hci_km
        )
        self.dv_hci_km = v_lambert_hci_km - v_final_hci_km

        # Velocity-only correction at the end of the propagated escape leg.
        # A velocity difference needs only the axis rotation (no origin shift).
        self.burns.append(
            Burn(
                t_s=self.t_traj_s[-1],
                r=self.rv_traj_eci_m[-1, 0:3],
                dv=calc_dcm_eci_wrt_hci() @ (self.dv_hci_km * KM_TO_M),
                frame="ECI",
            )
        )

        self.closest_lambert_jd = (
            lambert_dep_jd + time_s[closest_lambert_i] / DAY_TO_SEC
        )
        epoch_diff_jd = self.closest_lambert_jd - final_epoch_jd

        if verbose:
            print("midcourse_correction()")
            print(
                f"  losest Lambert sample: {closest_lambert_i}; date: {jd_to_greg(self.closest_lambert_jd)}"
            )
            print(
                f"  Earth-relative position separation: {lambert_distances_km[closest_lambert_i]:.3f} km"
            )
            print(
                f"  Lambert sample time seconds behind ECI propagated time: {epoch_diff_jd * DAY_TO_SEC:.1f} s"
            )
            print(
                f"  Velocity-only correction HCI [km/s]: {self.dv_hci_km}; magnitude: {norm(self.dv_hci_km) * KM_TO_M:.3f} m/s"
            )
            print(
                "  This compares the nearest stored sample; a continuous handoff also requires matching position and epoch."
            )
            earth_soi_m = sph_of_infl(
                norm(rv_earth_hci_km[0, 0:3] * KM_TO_M), MU_EARTH, MU_SUN
            )
            print(
                f"Crossed sphere of influence: {norm(self.rv_traj_eci_m[-1, 0:3]) > earth_soi_m}"
            )
