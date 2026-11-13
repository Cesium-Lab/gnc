import numpy as np
import pytest
from numpy.linalg import norm

from cesium.physics.energy import (
    calc_jacobi_constant,
    calc_kinetic_energy,
    calc_potential_energy,
    calc_specific_energy,
    calc_total_energy,
)
from cesium.physics.gravity import grav_accel
from cesium.world import MU_EARTH
from cesium.world.constants import A_MOON, J2_EARTH, MU_MOON, R_EARTH


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

    assert calc_kinetic_energy(np.array([0, 0, 0]), mass, np.array([0, 0, 0]), I) == 0
    assert calc_kinetic_energy(v, mass, np.array([0, 0, 0]), I) == 2500.0
    assert calc_kinetic_energy(np.array([0, 0, 0]), mass, w, I) == 3.45e-2
    assert calc_kinetic_energy(v, 0, w, I) == 3.45e-2
    assert calc_kinetic_energy(v, mass, w, np.zeros((3, 3))) == 2500.0


def test_KE_reasomable_values():
    v = np.array([7, 1, 0])
    w = np.array([1, 1, 1])
    I = np.array([[1, 2, 0], [0, 2, 0], [0, 0.9, 1]])
    mass = 100.0

    assert calc_kinetic_energy(v, mass, w, I) == 2500.0 + 3.45

    v = np.array([8e5, 4e6, 3e5])
    w = np.array([1e5, 1e5, 1e5])

    assert calc_kinetic_energy(v, mass, w, I) == 8.365e14 + 3.45e10


def test_KE_without_inertia_is_translational_only():
    """Point mass unless both w and I are given. 0.5 * 2 kg * |[3, 4, 0]|^2 = 25 J."""
    v = np.array([3.0, 4.0, 0.0])
    w = np.array([5.0, -2.0, 1.0])
    assert calc_kinetic_energy(v, 2.0) == 25.0  # 3DOF: no w or I
    assert calc_kinetic_energy(v, 2.0, w=w) == 25.0  # spin, but no inertia
    assert calc_kinetic_energy(v, 2.0, I=np.eye(3)) == 25.0  # inertia, but no spin
    assert calc_kinetic_energy(v, 2.0, w, np.eye(3)) == 25.0 + 15.0  # 0.5 * |w|^2 = 15


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


class Test_calc_specific_energy:
    def test_circular_orbit_is_minus_half_mu_over_r(self):
        """Hand calc: v^2 = mu / r on a circle, so E = mu/(2r) - mu/r = -mu / (2r)."""
        r = 7000e3
        rv = np.array([r, 0.0, 0.0, 0.0, np.sqrt(MU_EARTH / r), 0.0])
        assert calc_specific_energy(rv, MU_EARTH) == pytest.approx(-MU_EARTH / (2 * r))

    def test_j2_term_is_potential_of_grav_accel(self):
        """Independent of both formulas' algebra: a = -grad(U), by central differences."""
        r0 = np.array([4.1e6, -3.3e6, 4.6e6])

        def potential(r):  # energy at rest is the potential
            return calc_specific_energy(
                np.concatenate((r, np.zeros(3))), MU_EARTH, J2_EARTH, R_EARTH
            )

        h = 1.0  # m
        grad = np.array(
            [
                (potential(r0 + h * e) - potential(r0 - h * e)) / (2 * h)
                for e in np.eye(3)
            ]
        )
        a = grav_accel(r0, MU_EARTH, J2_EARTH, R_EARTH)
        assert -grad == pytest.approx(a, rel=1e-6)

    def test_batched_states(self):
        rv = np.array([[7000e3, 0, 0, 0, 7500.0, 0], [0, 8000e3, 0, -7000.0, 0, 0]])
        expected = [calc_specific_energy(row, MU_EARTH) for row in rv]
        assert calc_specific_energy(rv, MU_EARTH) == pytest.approx(expected)
