import numpy as np
import pytest
from numpy.linalg import norm

import cesium.guidance as guid
from cesium.constants import HOUR_TO_SEC, KM_TO_M
from cesium.world import MU_EARTH, R_EARTH


class Test_hcw_two_impulse:
    def test_curtis_example_7_2(_):
        """Curtis Example 7.2 p. 331"""

        r0 = np.array([20, 20, 20]) * KM_TO_M
        v0 = np.array([-0.02, 0.02, -0.005]) * KM_TO_M
        t_trans = 8 * HOUR_TO_SEC
        w = 0.00115697

        dv1, dv2, _ = guid.hcw_two_impulse(r0, v0, t_trans, w)
        assert dv1 == pytest.approx([29.3608, -66.7514, 13.0326], abs=5e-3)
        assert dv2 == pytest.approx([25.8223, 0.472444, 22.2449], rel=2e-1)  # tf
        dv_tot = np.linalg.norm(dv1) + np.linalg.norm(dv2)

        # Seems like not very good approx but I sort of got the right answer after lots of rounding
        assert dv_tot == pytest.approx(109.7, rel=1e-3)

    def test_curtis_example_7_3_behind_track(_):
        """Curtis Example 7.3 p. 335"""
        alt = 300 * KM_TO_M  # km
        r_ECI = R_EARTH + alt
        v_ECI = np.sqrt(MU_EARTH / r_ECI)
        n = v_ECI / r_ECI
        t_trans = 1.49 * HOUR_TO_SEC

        r0 = np.array([0, -2, 0]) * KM_TO_M
        v0 = np.zeros(3)

        dv1, dv2, _ = guid.hcw_two_impulse(r0, v0, t_trans, n)
        dv = norm(dv1) + norm(dv2)
        assert dv1 == pytest.approx([-9.4824e-3, -1.2225e-1, 0], rel=5e-3)
        assert dv2 == pytest.approx([-9.4824e-3, 1.2225e-1, 0], rel=5e-3)
        assert dv == pytest.approx(0.2452, abs=1e-4)
