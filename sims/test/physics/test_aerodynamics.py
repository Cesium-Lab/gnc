# ruff: noqa: E741
import sys
sys.path.append(".")
import pytest

import numpy as np

from csim.world import W_EARTH

from csim.physics import drag_v_rel, drag_accel

class TestDragVRel:
    def test_default_val(self):
        r = np.array([1,1,1])
        v = np.array([1,1,1])

        v_rel = drag_v_rel(r,v)

        assert np.array_equal(v_rel, [1+W_EARTH,1-W_EARTH,1])

    def test_array_list(self):
        w = [1,0,0]
        r = np.array([1,1,1])
        v = np.array([1,1,1])

        v_rel = drag_v_rel(r,v,w)

        assert np.array_equal(v_rel, [1, 1+1, 1-1])

    def test_float(self):
        w = 1
        r = np.array([1,1,1])
        v = np.array([1,1,1])

        v_rel = drag_v_rel(r,v,w)

        assert np.array_equal(v_rel, [1+1,1-1,1])

class TestVanillaDrag:

    def test_no_velocity(self):
        v_rel = np.array([0,0,0])
        m = 100
        rho = 1
        A = 0.02

        drag = drag_accel(v_rel, rho, A, m)
        assert np.array_equal(drag, [0,0,0])

    def test_no_cD_specified(self):
        v_rel = np.array([1,1,1])
        m = 100
        rho = 1
        A = 0.02

        drag = drag_accel(v_rel, rho, A, m)

        a = -3.81051e-4 # Hand calc
        assert np.allclose(drag, [a,a,a])

    def test_drag_with_cD_specified(self):
        v_rel = np.array([1,1,1])
        m = 100
        rho = 1
        A = 0.02
        Cd = 1

        drag = drag_accel(v_rel, rho, A, m, Cd)

        a = -1.73205e-4 # Hand calc
        assert np.allclose(drag, [a,a,a])

    def test_drag_with_BC(self):
        v_rel = np.array([1,1,1])
        rho = 1
        BC = 50

        drag = drag_accel(v_rel, rho, 0, 0, BC=BC)

        a = -0.01732051 # Hand calc
        assert np.allclose(drag, [a,a,a])