import numpy as np
import pytest
from numpy.linalg import norm

from cesium.constants import DEG_TO_RAD, KM_TO_M, RAD_TO_DEG
from cesium.guidance.noncoplanar import (
    guid_combined_i_raan_change,
    guid_i_change,
    guid_raan_change_circular,
    node_line_finder,
)
from cesium.orbital.coes import coes_to_rv


class Test_coplanar:
    def test_guid_i_change_vallado_6_4(self):
        """Vallado 4e. Example 6-4. p. 344"""
        di = 15 * DEG_TO_RAD
        v_i_m = 5.892311 * KM_TO_M  # km and m agnostic

        # circular e=0 (flight path angle is 0)
        assert guid_i_change(di, 0, v_i_m) == pytest.approx(1538.2021, abs=1e-3)

        # other e=0 (flight path angle is 0)
        _, v = coes_to_rv(
            a=19625.037 * KM_TO_M,  # km
            e=0.3,
            i=47 * DEG_TO_RAD,  # arbitrary with the di
            aop=30 * DEG_TO_RAD,
            raan=30 * DEG_TO_RAD,  # arbitrary
            ta=330 * DEG_TO_RAD,
        )

        # TODO: flight path angle calculation
        phi_fpa = np.arctan2(
            0.3 * np.sin(330 * DEG_TO_RAD), 1 + 0.3 * np.cos(330 * DEG_TO_RAD)
        )
        assert phi_fpa * RAD_TO_DEG == pytest.approx(-6.79, abs=1e-3)
        assert guid_i_change(di, phi_fpa, norm(v)) == pytest.approx(1553.727, abs=1e-3)

        # TODO: flight path angle calculation
        _, v = coes_to_rv(
            a=19625.037 * KM_TO_M,  # km
            e=0.3,
            i=47 * DEG_TO_RAD,  # arbitrary with the di
            aop=30 * DEG_TO_RAD,
            raan=30 * DEG_TO_RAD,  # arbitrary
            ta=150 * DEG_TO_RAD,
        )
        phi_fpa = np.arctan2(
            0.3 * np.sin(150 * DEG_TO_RAD), 1 + 0.3 * np.cos(150 * DEG_TO_RAD)
        )
        assert phi_fpa * RAD_TO_DEG == pytest.approx(11.4558, abs=1e-4)
        assert guid_i_change(di, phi_fpa, norm(v)) == pytest.approx(912.883, abs=1e-3)

    def test_guid_raan_change_vallado_6_5(self):
        """Vallado 4e. Example 6-5. p. 347"""
        i = 55 * DEG_TO_RAD
        v_i_m = 5.892311 * KM_TO_M  # km and m agnostic
        draan = 45 * DEG_TO_RAD

        # circular e=0 (flight path angle is 0)
        assert guid_raan_change_circular(draan, i, v_i_m) == pytest.approx(
            3694.195, abs=1e-3
        )

    def test_guid_i_raan_combined_vallado_6_6(self):
        """Vallado 4e. Example 6-6. p. 349"""
        i_i = 55 * DEG_TO_RAD
        i_f = 40 * DEG_TO_RAD
        v_i_m = 5.892311 * KM_TO_M  # km and m agnostic
        draan = 45 * DEG_TO_RAD

        # circular e=0 (flight path angle is 0)
        assert guid_combined_i_raan_change(i_i, i_f, draan, v_i_m) == pytest.approx(
            3615.925, abs=1e-3
        )


class Test_node_line_finder:
    MU = 3.986e14

    def rv(self, i, raan, ta):
        return np.concatenate(coes_to_rv(7000e3, 0.0, i, raan, 0.0, ta, self.MU))

    def test_equatorial_target_node_is_chaser_ascending_node(self):
        # Chaser at 30 deg past its ascending node (+x); target in the equator
        n_hat, d_ta, h_c, h_t = node_line_finder(
            self.rv(0.5, 0.0, 30 * DEG_TO_RAD), self.rv(0.0, 0.0, 0.0)
        )
        assert abs(n_hat @ [1.0, 0.0, 0.0]) == pytest.approx(1.0)
        assert n_hat @ h_c == pytest.approx(
            0.0, abs=1e-6
        ) and n_hat @ h_t == pytest.approx(0.0, abs=1e-6)
        # Next crossing of n_hat: either 150 deg (descending, -x) or 330 deg (ascending, +x) away
        expected = 330 * DEG_TO_RAD if n_hat[0] > 0 else 150 * DEG_TO_RAD
        assert d_ta == pytest.approx(expected)

    def test_sweep_lands_on_node(self):
        rv_c = self.rv(0.7, 0.4, 2.0)
        n_hat, d_ta, _, _ = node_line_finder(rv_c, self.rv(0.6, 0.5, 0.0))
        r_node = coes_to_rv(7000e3, 0.0, 0.7, 0.4, 0.0, 2.0 + d_ta, self.MU)[0]
        assert r_node / norm(r_node) == pytest.approx(n_hat, abs=1e-9)
