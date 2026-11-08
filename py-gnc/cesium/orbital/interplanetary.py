import numpy as np
from lamberthub import izzo2015
from numpy.linalg import norm

from ..constants import C_LIGHT
from ..math.rotations import wrap_to_2pi, wrap_to_pi
from ..math.time import greg_to_jd
from ..world.constants import MU_SUN


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


###########################################################################
#                          Earth Departure
###########################################################################


def departure_characteristics(v_dep_eci: np.ndarray):
    C3 = norm(v_dep_eci) ** 2
    rla = wrap_to_2pi(np.arctan2(v_dep_eci[1], v_dep_eci[0]))
    dla = wrap_to_pi(np.arcsin(v_dep_eci[2] / norm(v_dep_eci)))
    return C3, rla, dla
