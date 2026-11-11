import numpy as np

from cesium.orbital.hcw import calc_Phi

from .maneuvers import Burn


def hcw_two_impulse(r0: np.ndarray, v0: np.ndarray, t_trans: float, n: float):
    Phi_rr, Phi_rv, Phi_vr, Phi_vv = calc_Phi(t_trans, n)

    v_i = np.linalg.solve(Phi_rv, -Phi_rr @ r0)
    dv1_RSW = v_i - v0

    v_f = Phi_vr @ r0 + Phi_vv @ v_i
    dv2_RSW = -v_f

    dr = Phi_rr @ r0 + Phi_rv @ v0

    return dv1_RSW, dv2_RSW, dr


def plan_hcw_two_impulse(
    r0_rsw: np.ndarray, v0_rsw: np.ndarray, t_trans: float, n: float
) -> list[Burn]:
    """Two-impulse HCW rendezvous to the target (RSW origin) as `Burn`s.

    Wraps `hcw_two_impulse`: the first burn at `t_s = 0` puts the chaser on the
    transfer, the second at `t_s = t_trans` cancels its relative velocity.

    Parameters
    ----------
    r0_rsw, v0_rsw : np.ndarray (3,)
        Chaser position and velocity relative to the target in RSW [m, m/s].
    t_trans : float
        Transfer time [s].
    n : float
        Target mean motion [rad/s].

    Returns
    -------
    list[Burn]
        Two burns with `frame="RSW"`; `r` is the chaser's relative position
        (the target, at the origin, for the second burn).
    """
    dv1_rsw, dv2_rsw, _ = hcw_two_impulse(r0_rsw, v0_rsw, t_trans, n)
    # After the transfer the chaser is at the target, the RSW origin.
    return [
        Burn(t_s=0.0, r=np.asarray(r0_rsw, dtype=float), dv=dv1_rsw, frame="RSW"),
        Burn(t_s=float(t_trans), r=np.zeros(3), dv=dv2_rsw, frame="RSW"),
    ]
