import numpy as np
import pytest
from numpy.linalg import norm

from cesium.physics.energy import (
    calc_jacobi_constant,
    calc_kinetic_energy,
    calc_potential_energy,
    calc_total_energy,
)
from cesium.world import MU_EARTH
from cesium.world.constants import A_MOON, MU_MOON


def test_PE_values_of_0():
    r = np.array([7000, 0, 0])
    mass = 100

    assert calc_potential_energy(r, 0, MU_EARTH) == 0
    assert calc_potential_energy(np.array([0, 0, 0]), mass, MU_EARTH) == 0
    assert calc_potential_energy(r, mass, 0) == 0


def test_PE_reasonable_values():
    mass = 100.0

    r = np.array([7000, 0, 0]) * 1e6
    assert calc_potential_energy(r, mass, MU_EARTH) == pytest.approx(-5694292, rel=1e-6)

    r = np.array([8e9, 4e10, 3e8]) * 1e6
    assert calc_potential_energy(r, mass, MU_EARTH) == pytest.approx(
        -0.9771233, rel=1e-6
    )

    mass = 1e12
    assert calc_potential_energy(r, mass, MU_EARTH) == pytest.approx(
        -9.771233e9, rel=1e-6
    )

    r = np.array([0.1, 0.1, 0.1])
    assert calc_potential_energy(r, mass, MU_EARTH) == pytest.approx(
        -2.3013207e27, rel=1e-6
    )


def test_KE_values_of_0():

    v = np.array([7, 1, 0])
    w = np.array([0.1, 0.1, 0.1])
    I = np.array([[1, 2, 0], [0, 2, 0], [0, 0.9, 1]])
    mass = 100.0

    assert calc_kinetic_energy(np.array([0, 0, 0]), np.array([0, 0, 0]), mass, I) == 0
    assert calc_kinetic_energy(v, np.array([0, 0, 0]), mass, I) == 2500.0
    assert calc_kinetic_energy(np.array([0, 0, 0]), w, mass, I) == 3.45e-2
    assert calc_kinetic_energy(v, w, 0, I) == 3.45e-2
    assert calc_kinetic_energy(v, w, mass, np.zeros((3, 3))) == 2500.0


def test_KE_reasomable_values():
    v = np.array([7, 1, 0])
    w = np.array([1, 1, 1])
    I = np.array([[1, 2, 0], [0, 2, 0], [0, 0.9, 1]])
    mass = 100.0

    assert calc_kinetic_energy(v, w, mass, I) == 2500.0 + 3.45

    v = np.array([8e5, 4e6, 3e5])
    w = np.array([1e5, 1e5, 1e5])

    assert calc_kinetic_energy(v, w, mass, I) == 8.365e14 + 3.45e10


def test_total_energy():
    I = np.array([[1, 2, 0], [2, 2, 0.9], [0, 0.9, 1]])
    mass = 100.0

    v = np.array([8e5, 4e6, 3e5])
    w = np.array([1e5, 1e5, 1e5])
    r = np.array([7000, 0, 0]) * 1e4

    #                                                       KE_trans     KE_rot           PE_grav
    assert (
        calc_total_energy(mass, I, r, v, w, MU_EARTH)
        == 8.365e14 + 49000000000.0 - 5.6942920257e8
    )


class Test_calc_jacobi_constant:
    L = A_MOON
    R1_BC = L * MU_MOON / (MU_EARTH + MU_MOON)
    R2_BC = L * MU_EARTH / (MU_EARTH + MU_MOON)
    W_S = np.sqrt((MU_EARTH + MU_MOON) / L**3)

    def test_at_rest_at_l4_matches_literature(self):
        """Nondimensional C(L4) = 3 - mu + mu^2 (~2.988 for Earth-Moon; Koon, Lo, Marsden & Ross).

        Lengths scale by L and time by 1/w_s, so the SI value is C_nd * (w_s L)^2.
        """
        mu = MU_MOON / (MU_EARTH + MU_MOON)  # mass ratio, ~0.01215
        x, y = self.L / 2 - self.R1_BC, np.sqrt(3) / 2 * self.L
        c = calc_jacobi_constant(
            np.array([x, y, 0.0, 0.0, 0.0, 0.0]),
            MU_EARTH,
            MU_MOON,
            self.R1_BC,
            self.R2_BC,
            self.W_S,
        )
        c_nd = c / (self.W_S * self.L) ** 2
        assert c_nd == pytest.approx(3 - mu + mu**2, rel=1e-12)
        assert c_nd == pytest.approx(2.988, abs=1e-3)  # tabulated Earth-Moon value

    def test_no_secondary_is_energy_and_angular_momentum(self):
        """With mu2 = 0 and Earth at the origin: C = -2 (E - w_s h_z), from inertial E and h."""
        w_s = np.sqrt(MU_EARTH / self.L**3)
        rv_S = np.array([2.0e7, -3.0e7, 5.0e6, 800.0, 1200.0, -300.0])
        r, v_S = rv_S[0:3], rv_S[3:6]
        v_I = v_S + np.cross([0.0, 0.0, w_s], r)  # rotating -> inertial velocity

        energy = 0.5 * np.dot(v_I, v_I) - MU_EARTH / norm(r)
        h_z = np.cross(r, v_I)[2]
        c = calc_jacobi_constant(rv_S, MU_EARTH, 0.0, 0.0, self.L, w_s)
        assert c == pytest.approx(-2 * (energy - w_s * h_z), rel=1e-12)
