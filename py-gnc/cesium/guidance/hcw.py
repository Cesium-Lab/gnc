import numpy as np

from cesium.orbital.hcw import calc_Phi


def hcw_two_impulse(r0: np.ndarray, v0: np.ndarray, t_trans: float, n: float):
    Phi_rr, Phi_rv, Phi_vr, Phi_vv = calc_Phi(t_trans, n)

    v_i = np.linalg.solve(Phi_rv, -Phi_rr @ r0)
    dv1_RSW = v_i - v0

    v_f = Phi_vr @ r0 + Phi_vv @ v_i
    dv2_RSW = -v_f

    dr = Phi_rr @ r0 + Phi_rv @ v0

    return dv1_RSW, dv2_RSW, dr
