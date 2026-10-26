import numpy as np

from ..math.time import greg_to_jd


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


def sph_of_infl(r_sat: float, mu_sat: float, mu_cb: float):
    return r_sat * (mu_sat / mu_cb) ** (2 / 5)
