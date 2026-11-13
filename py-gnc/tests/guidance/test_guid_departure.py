import numpy as np
import pytest
from numpy.linalg import norm

from cesium.guidance.departure import EarthDepartureTrajectory
from cesium.math.time import greg_to_jd
from cesium.orbital.transformations import geodetic_LLA_to_ECEF
from cesium.world import MU_EARTH, R_EARTH

R_P = R_EARTH + 300e3  # m
V_INF_ECI_KM = np.array([2.1, -1.4, 0.9])  # km/s


# From the Mars courier example notebook
@pytest.fixture(scope="module")
def departure():
    return EarthDepartureTrajectory(
        greg_to_jd(2035, 6, 14, 12, 0, 0),
        V_INF_ECI_KM,
        R_P,
        geodetic_LLA_to_ECEF(28.48889, -80.57778),
    )


def test_escape_burn_is_tangential_from_parking_orbit(
    departure: EarthDepartureTrajectory,
):
    (burn,) = departure.burns

    assert burn.t_s == 0.0
    assert burn.frame == "ECI"
    assert np.array_equal(burn.r, departure.r_p_eci)
    assert np.dot(burn.r, burn.dv) == pytest.approx(0.0, abs=1e-3)
    assert burn.dv_mag == pytest.approx(departure.dv_periapse_m, rel=1e-12)
    assert departure.dv_total == pytest.approx(burn.dv_mag)


def test_escape_burn_starts_from_circular_speed(departure):
    """Pre-burn velocity (v_p - dv) is circular; post-burn has the target v_inf."""
    (burn,) = departure.burns
    v_before = departure.v_p_eci - burn.dv

    assert norm(v_before) == pytest.approx(np.sqrt(MU_EARTH / R_P), rel=1e-12)
    v_inf_sq = norm(departure.v_p_eci) ** 2 - 2 * MU_EARTH / R_P
    assert np.sqrt(v_inf_sq) == pytest.approx(norm(V_INF_ECI_KM) * 1e3, rel=1e-9)
