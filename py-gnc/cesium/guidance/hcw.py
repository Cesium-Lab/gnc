import numpy as np
from numpy.linalg import norm

from ..constants import DAY_TO_SEC, M_TO_KM, RAD_TO_DEG
from ..math.base import unit
from ..math.rotations import wrap_to_2pi, wrap_to_pi
from ..orbital.coes import COES, orbit_sweep
from ..orbital.hcw import calc_Phi
from ..world.constants import J2_MARS, MU_MARS, R_MARS
from .coplanar import guid_coaxial_hohmann, guid_phasing_same_orbit
from .maneuvers import Burn
from .noncoplanar import guid_plane_change_3d, node_line_finder
from .timeline import RpoTimeline


def hcw_two_impulse(r0: np.ndarray, v0: np.ndarray, t_trans: float, n: float):
    Phi_rr, Phi_rv, Phi_vr, Phi_vv = calc_Phi(t_trans, n)

    v_i = np.linalg.solve(Phi_rv, -Phi_rr @ r0)
    dv1_RSW = v_i - v0

    v_f = Phi_vr @ r0 + Phi_vv @ v_i
    dv2_RSW = -v_f

    # Troubleshooting
    # dr = Phi_rr @ r0 + Phi_rv @ v_i
    # dv = Phi_vr @ r0 + Phi_vv @ v_i
    # print(dr)
    # print(dv)

    return dv1_RSW, dv2_RSW


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
    dv1_rsw, dv2_rsw = hcw_two_impulse(r0_rsw, v0_rsw, t_trans, n)
    # After the transfer the chaser is at the target, the RSW origin.
    return [
        Burn(t_s=0.0, r=np.asarray(r0_rsw, dtype=float), dv=dv1_rsw, frame="RSW"),
        Burn(t_s=float(t_trans), r=np.zeros(3), dv=dv2_rsw, frame="RSW"),
    ]


def coast_to_nearer_apsis(tl: RpoTimeline) -> bool:
    """Coast the chaser to its nearer apsis; returns True if that apsis is periapsis."""
    c, _ = tl.coes()
    peri = c.ta > np.pi
    sweep = (2 * np.pi - c.ta) if peri else wrap_to_2pi(np.pi - c.ta)
    tl.coast(orbit_sweep(c, sweep, tl.mu)[1])
    return peri


def run_rpo(
    chaser0: COES,
    target0: COES,
    k_phase: int,
    hcw_thresh: float,
    dock_thresh: float,
    t_rendezvous: float,
    *,
    hcw_iter: int = 3,
    raan_bias: float = 0.0,
    epoch0_tdb_s: float = 0.0,
    mu: float = MU_MARS,
    j2: float | None = J2_MARS,
    r_cb: float = R_MARS,
    frame: str = "MCI",
    verbose: bool = False,
):
    """Full RPO timeline. `raan_bias` [rad] pre-biases the Burn 1 plane against J2 nodal drift."""
    tl = RpoTimeline(
        chaser0.to_rv(mu),
        target0.to_rv(mu),
        mu,
        j2=j2,
        r_cb=r_cb,
        epoch0_tdb_s=epoch0_tdb_s,
        frame=frame,
    )
    info = {}

    # Burn 1: coast to the node with the (biased) target plane, rotate into it
    c, t = tl.coes()
    n_hat, d_node, _, _ = node_line_finder(tl.chaser, t.bias_raan(raan_bias).to_rv(mu))
    tl.coast(orbit_sweep(c, d_node, mu)[1])
    _, t = tl.coes()
    tl.burn(guid_plane_change_3d(tl.chaser, t.bias_raan(raan_bias).h(mu)), "Burn 1 (plane)")
    info["n_hat"] = n_hat

    # Burns 2-3: coast to the nearer apsis, Hohmann to the target radius
    peri = coast_to_nearer_apsis(tl)
    c, t = tl.coes()
    _, dt_trans, dv_a, dv_b = guid_coaxial_hohmann(
        c.r_p, c.r_a, t.a, t.a, depart_at_periapsis=peri, mu=mu
    )
    tl.burn(dv_a * unit(tl.chaser[3:]), "Burn 2 (Hohmann)")
    tl.coast(dt_trans)

    # Burn 3: circularize and enter the phasing orbit in one burn
    v_hat = unit(tl.chaser[3:])
    rv_circ = tl.chaser.copy()
    rv_circ[3:] += dv_b * v_hat
    c_circ = COES.from_rv(rv_circ, mu)
    _, t = tl.coes()
    dv_ph, dt_ph, a_ph = guid_phasing_same_orbit(
        t.a, wrap_to_pi(c_circ.aol - t.aol), mu, k_phase, k_phase, deg=False, r_cb=r_cb
    )
    info["a_phase"], info["t_phase"], info["a_target"], info["i_target"] = a_ph, dt_ph, t.a, t.i
    tl.burn((dv_b + dv_ph / 2) * v_hat, "Burn 3 (circularize + phase entry)")
    tl.coast(dt_ph / k_phase)

    # Burns 4.x: closed-loop same-orbit phasing (re-planned every rev from the actual state)
    phases_remaining = k_phase
    while phases_remaining:
        c, t = tl.coes()
        _, _, a_phase = guid_phasing_same_orbit(
            t.a,
            wrap_to_pi(c.aol - t.aol),
            mu,
            phases_remaining,
            phases_remaining,
            deg=False,
            r_cb=r_cb,
        )
        r_now = norm(tl.chaser[0:3])
        v_req = np.sqrt(mu * (2 / r_now - 1 / a_phase))
        tl.burn((v_req - norm(tl.chaser[3:6])) * unit(tl.chaser[3:]), f"Burn 4.{k_phase - phases_remaining} (phasing)")
        phases_remaining -= 1

        c, _ = tl.coes()
        tl.coast(c.period(mu))
        if norm(tl.to_rsw()[0:3]) < hcw_thresh:
            break

    # Handoff diagnostics (before the phase exit)
    rr = tl.to_rsw()
    c, t = tl.coes()
    info["handoff_rsw"] = rr[0:3]
    info["draan"] = wrap_to_pi(c.raan - t.raan)
    if verbose:
        print(
            f"Phasing exit: {phases_remaining=}, sep {norm(rr[0:3]) * M_TO_KM:.3f} km "
            f"(R {rr[0] * M_TO_KM:.3f}, S {rr[1] * M_TO_KM:.3f}, W {rr[2] * M_TO_KM:.3f} km), "
            f"dRAAN {info['draan'] * RAD_TO_DEG:.4f} deg"
        )

    # Burn 5: phase exit, match the target's semi-major axis
    r_now = norm(tl.chaser[0:3])
    v_req = np.sqrt(mu * (2 / r_now - 1 / t.a))
    tl.burn((v_req - norm(tl.chaser[3:6])) * unit(tl.chaser[3:]), "Burn 5 (phase exit)")

    # HCW two-impulse rendezvous, repeated to clean up linearization error
    for k in range(hcw_iter):
        rv_rsw = tl.to_rsw()
        dv1_rsw, dv2_rsw = hcw_two_impulse(
            rv_rsw[0:3], rv_rsw[3:6], t_rendezvous, tl.coes()[1].n(mu)
        )
        rv_rsw[3:6] += dv1_rsw
        tl.burn(tl.from_rsw(rv_rsw)[3:6] - tl.chaser[3:6], f"HCW {k}A")
        tl.coast(t_rendezvous)

        if norm(tl.to_rsw()[0:3]) < dock_thresh:
            break


    rv_rsw = tl.to_rsw()
    rv_rsw[3:6] += dv2_rsw
    tl.burn(tl.from_rsw(rv_rsw)[3:6] - tl.chaser[3:6], f"HCW {k}B")

    rr = tl.to_rsw()
    c, t = tl.coes()
    info["docking_rsw"] = rr[0:3]
    info["dock_draan"] = wrap_to_pi(c.raan - t.raan)
    if verbose:
        print(
            f"HCW exit: sep {norm(rr[0:3]) * M_TO_KM:.3f} km "
            f"(R {rr[0] * M_TO_KM:.3f}, S {rr[1] * M_TO_KM:.3f}, W {rr[2] * M_TO_KM:.3f} km), "
            f"dRAAN {info['draan'] * RAD_TO_DEG:.4f} deg"
        )

    tl.coast(1 * DAY_TO_SEC)
    if verbose:
        print(f"total dv = {tl.dv_total:.3f} m/s")
    return tl, info