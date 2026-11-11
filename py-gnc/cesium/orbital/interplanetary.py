from __future__ import annotations

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
    """Solve a prograde, zero-revolution heliocentric Lambert transfer in SI units.

    Parameters
    ----------
    r1_intrl, r2_intrl : ndarray, shape (3,)
        Departure and arrival positions in the same inertial frame, in meters.
    tof_s : float
        Flight duration in seconds.
    mu : float, optional
        Central-body gravitational parameter in m³/s²; defaults to the Sun.

    Returns
    -------
    v1_intrl, v2_intrl : ndarray, shape (3,)
        Real inertial departure and arrival velocities in m/s. Returns two
        scalar NaNs instead if duration is nonpositive or chord speed exceeds
        the speed of light.

    Raises
    ------
    ValueError
        If Lambert fails, returns non-finite velocities, or has an imaginary
        velocity component larger than 1e-9 m/s.

    Notes
    -----
    Uses ``izzo2015`` prograde with M=0 and its default direction/branch settings.
    """
    if tof_s <= 0 or (norm(r1_intrl - r2_intrl) / tof_s) > C_LIGHT:
        return np.nan, np.nan

    result = izzo2015(mu, r1_intrl, r2_intrl, float(tof_s), prograde=True, M=0)
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


def hyperbolic_characteristics(v_dep_intrl: np.ndarray):
    """Calculate characteristic energy and asymptote angles from excess velocity.

    Parameters
    ----------
    v_dep_intrl : ndarray, shape (3,)
        Nonzero, planet-relative hyperbolic excess velocity expressed in
        Earth's equatorial inertial axes. Subtract Planet's velocity BEFORE
        rotating from HCI; do not supply heliocentric spacecraft velocity.

    Returns
    -------
    C3 : float
        Squared excess speed. Units follow the input: m²/s² for m/s, or
        km²/s² for km/s.
    rla : float
        Right ascension of the outgoing asymptote in radians, in [0, 2*pi).
    dla : float
        Declination of the outgoing asymptote in radians, in [-pi/2, pi/2].
    """
    C3 = norm(v_dep_intrl) ** 2
    rla = wrap_to_2pi(np.arctan2(v_dep_intrl[1], v_dep_intrl[0]))
    dla = wrap_to_pi(np.arcsin(v_dep_intrl[2] / norm(v_dep_intrl)))
    return C3, rla, dla
