import numpy as np
from numpy.linalg import norm
from scipy.integrate import solve_ivp

from ..constants import DAY_TO_SEC, HOUR_TO_SEC, KM_TO_M, M_TO_KM, RAD_TO_DEG
from ..math.base import unit
from ..math.time import jd_to_greg
from ..orbital.ephemeris import planet_elements_and_rv
from ..orbital.interplanetary import sph_of_infl
from ..orbital.transformations import ECI_from_HCI, itrf_to_gcrs_matrix
from ..physics.dynamics import two_body
from ..world.constants import MU_EARTH, MU_SUN


def interplanetary_circ_injection(
    v_inf_PCI: float, r_p: float, mu: float, deg=True
) -> tuple[float, float, float, float]:
    """Interplanetary Intercept along planet velocity vector

    Parameters
    ----------
    v_inf_PCI : float
        Planet centered departure or arrival velocity in inertial frame [m/s]
    r_p : float
        Periapse radius [m]
    mu : float
        Central body gravitational parameter [m3/s2]

    Returns
    -------
    dv_p : float
        Delta-v at periapsis for circular capture
    ta_asymptote : float
        Arrival true anomaly angle to asymptote [rad]
    aiming_radius : float
        Distance between asymptote and periapsis [m]
    e : float
        Eccentricity


    References
    -------
    AA 279B Lecture 9 p. 23
    """
    v_p = np.sqrt(v_inf_PCI**2 + 2 * mu / r_p)
    v_circ = np.sqrt(mu / r_p)
    dv_p = norm(v_circ - v_p)

    e = 1 + r_p * v_inf_PCI**2 / mu
    assert e > 1

    ta_asymptote = np.arccos(-1 / e)

    aiming_radius = mu / (v_inf_PCI**2) * np.sqrt(e * e - 1)

    if deg:
        ta_asymptote *= RAD_TO_DEG

    return dv_p, ta_asymptote, aiming_radius, e


# eps=1e-8rad
def flyby_shoot(
    v_arrival_PCI: float,
    v_departure_PCI: float,
    delta_des: float,
    mu: float,
    deg=False,
    eps=1e-8,
    d=10,
    verbose=False,
) -> tuple[float, float, float, float, float]:

    delta_des = delta_des / RAD_TO_DEG if deg else delta_des

    def delta_tot(r_p):
        return np.arcsin(1 / (1 + r_p * v_arrival_PCI**2 / mu)), np.arcsin(
            1 / (1 + r_p * v_departure_PCI**2 / mu)
        )

    # Guess if circular velocity with e=2
    e = 2
    r_p = mu * (e - 1) / (v_arrival_PCI**2)

    iter = 20
    for i in range(iter):
        delta1, delta2 = delta_tot(r_p)

        d_delta = delta_des - (delta1 + delta2)

        if verbose:
            print(f"d_delta: {d_delta:.5f}m")

        if norm(d_delta) < eps:
            break

        J = (sum(delta_tot(r_p + d)) - sum(delta_tot(r_p - d))) / (2 * d)

        step = d_delta / J

        # Make sure periapsis is positive
        alpha = 1.0

        while r_p + alpha * step <= 0:
            alpha *= 0.5

        r_p += alpha * step

    else:
        raise ValueError

    # dv
    v_p1 = np.sqrt(v_arrival_PCI**2 + 2 * mu / r_p)
    v_p2 = np.sqrt(v_departure_PCI**2 + 2 * mu / r_p)
    dv = abs(v_p2 - v_p1)
    # e
    e1 = 1 + r_p * v_arrival_PCI**2 / mu
    e2 = 1 + r_p * v_departure_PCI**2 / mu

    return dv, v_p1, v_p2, r_p, e1, e2, delta1, delta2


class EarthDepartureTrajectory:
    def __init__(
        self,
        launch_date_jd: float,
        v_inf_eci_km: np.ndarray,
        periapse_m: float,
        r_site_ecef_m: np.ndarray,
    ):
        self.dcm_ecef_wrt_eci0 = itrf_to_gcrs_matrix(launch_date_jd).T
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
            print(f"dcm_eci_wrt_orbit det > 1: ({np.linalg.det(self.dcm_eci_wrt_orbit)})")

        r_p_orbit = (
            np.array([np.cos(self.ta_asymptote_rad), -np.sin(self.ta_asymptote_rad), 0])
            * periapse_m
        )
        self.r_p_eci = self.dcm_eci_wrt_orbit @ r_p_orbit
        self.v_p_eci = unit(np.cross(self.n_vec, self.r_p_eci)) * (
            np.sqrt(MU_EARTH / norm(r_p_orbit)) + self.dv_periapse_m
        )

    def launch_that_babyy(self, tspan_s: float = 12 * HOUR_TO_SEC, npoints=100000):
        rv_eci_0 = np.concatenate((self.r_p_eci, self.v_p_eci))

        self.t_traj_s = np.linspace(0, tspan_s, npoints)
        departure_eci = solve_ivp(
            lambda _, s: two_body(s, MU_EARTH),
            (self.t_traj_s[0], self.t_traj_s[-1]),
            rv_eci_0,
            t_eval=self.t_traj_s,
            rtol=1e-9,
            atol=1e-12,
        )
        assert departure_eci.success

        self.rv_traj_eci_m = departure_eci.y.T

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
            + ECI_from_HCI().T @ rv_sat_eci_km[closest_lambert_i, 3:6]
        )
        # v_lambert_hci_km = mission["satellite_rv_hci_km"][closest_lambert_i, 3:6]

        ##### From this notebook
        # Add Earth's velocity at the propagated endpoint to get actual HCI velocity.
        final_epoch_jd = self.launch_date_jd + self.t_traj_s[-1] / DAY_TO_SEC
        v_earth_final_hci_km = planet_elements_and_rv(3, final_epoch_jd)[2] * M_TO_KM

        v_final_hci_km = (
            ECI_from_HCI().T @ (self.rv_traj_eci_m[-1, 3:6] * M_TO_KM)
            + v_earth_final_hci_km
        )
        self.dv_hci_km = v_lambert_hci_km - v_final_hci_km

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
            earth_soi_m = sph_of_infl(norm(rv_earth_hci_km[0, 0:3] * KM_TO_M), MU_EARTH, MU_SUN)
            print(f"Crossed sphere of influence: {norm(self.rv_traj_eci_m[-1, 0:3]) > earth_soi_m}")


# def flyby_dv(v_arrival_PCI: float, b_arrival: float,
#              v_departure_PCI: float, b_departure: float,
#              mu: float, dv: float = 0, deg=True) -> tuple[float, float, float]:

# e1 = np.sqrt((b_arrival * v_arrival_PCI**2 / mu)**2  + 1)
# e2 = np.sqrt((b_arrival * v_arrival_PCI**2 / mu)**2  + 1)
# assert e1 > 1
# assert e2 > 1

# r_p = mu*(e-1) / (v_arrival_PCI**2)
# v_p1 = np.sqrt(v_arrival_PCI**2 - 2 * mu / r_p)
# v_p2 = v_p1 + dv

# v_departure_PCI = np.sqrt(v_arrival_PCI**2 - 2 * mu / r_p)

# dv = norm()


# v_p = np.sqrt(v_arrival_PCI**2 + 2 * mu / r_p)
# v_circ = np.sqrt(mu / r_p)
# dv_p = norm(v_circ - v_p)

# delta = angle_between()
