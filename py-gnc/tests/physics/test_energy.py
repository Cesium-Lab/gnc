import sys
import numpy as np
import pytest

sys.path.append(".")

from csim.physics.energy import (
    calc_kinetic_energy,
    calc_potential_energy,
    calc_total_energy,
)
from csim.world import MU_EARTH


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
