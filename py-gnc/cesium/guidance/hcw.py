import numpy as np
from numpy.linalg import norm

from ..constants import DAY_TO_SEC, M_TO_KM, RAD_TO_DEG
from ..math.base import unit
from ..math.rotations import wrap_to_2pi, wrap_to_pi
from ..orbital.coes import COES, orbit_sweep
from ..orbital.hcw import calc_Phi
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


####################################################################################################
#               RPO phases
####################################################################################################

################################################################################
#               Plane-changing
################################################################################


def plane_change_at_node(tl: RpoTimeline, raan_bias: float = 0.0) -> np.ndarray:
    """Burn 1: coast to the node with the target plane (RAAN offset by `raan_bias` [rad]) and rotate into it.

    Returns node hat vector
    """
    # Node
    c, t = tl.coes()
    rv_target_biased = t.bias_raan(raan_bias).to_rv(tl.mu)
    n_hat, d_node, _, _ = node_line_finder(tl.chaser, rv_target_biased)

    # Sweep until node
    _, dt_coast = orbit_sweep(c, d_node, tl.mu)
    tl.coast(dt_coast)

    # Actual burn
    _, t = tl.coes()
    tl.burn(
        guid_plane_change_3d(tl.chaser, t.bias_raan(raan_bias).h(tl.mu)),
        "Burn 1 (plane)",
    )
    return n_hat


################################################################################
#               sma and eccentricity change
################################################################################


def coast_to_nearer_apsis(tl: RpoTimeline) -> bool:
    """Coast the chaser to its nearer apsis; returns True if that apsis is periapsis."""
    c, _ = tl.coes()
    periapsis_is_closer = c.ta > np.pi
    sweep = (2 * np.pi - c.ta) if periapsis_is_closer else wrap_to_2pi(np.pi - c.ta)
    tl.coast(orbit_sweep(c, sweep, tl.mu)[1])
    return periapsis_is_closer


def hohmann_to_target_radius(tl: RpoTimeline) -> float:
    """Burn 2: from the nearer closest apsis do Hohmann toward the target sma

    Applies the first burn and coasts the transfer. Returns the (unapplied) second-burn
    to-go to merge with the next maneuver.
    """
    periapsis_is_closer = coast_to_nearer_apsis(tl)
    c, t = tl.coes()
    _, dt_trans, dv_a, dv_b = guid_coaxial_hohmann(
        c.r_p, c.r_a, t.a, t.a, depart_at_periapsis=periapsis_is_closer, mu=tl.mu
    )
    tl.burn(dv_a * unit(tl.chaser[3:6]), "Burn 2 (Hohmann)")
    tl.coast(dt_trans)
    return dv_b


################################################################################
#               Phasing! !!!
################################################################################


def enter_phasing(tl: RpoTimeline, k_phase: int, dv_circ: float = 0.0) -> dict:
    """Burn 3: circularize (`dv_circ` [m/s], along-track) and enter a `k_phase`-rev phasing orbit in one burn.

    Returns the plan: phasing semi-major axis and duration, target a and i.
    """
    # Anomaly of chaser WITH the burn-to-go applied
    v_hat = unit(tl.chaser[3:6])
    rv_circ = tl.chaser.copy()
    rv_circ[3:6] += dv_circ * v_hat
    c_circ = COES.from_rv(rv_circ, tl.mu)

    # Phase velocity
    _, t = tl.coes()
    dv_ph, dt_ph, a_ph = guid_phasing_same_orbit(
        t.a,
        wrap_to_pi(c_circ.aol - t.aol),
        tl.mu,
        k_phase,
        k_phase,
        deg=False,
        r_cb=tl.r_cb,
    )

    # Apply, considering burn-to-go from before
    tl.burn((dv_circ + dv_ph / 2) * v_hat, "Burn 3 (circularize + phase entry)")
    tl.coast(dt_ph / k_phase)
    return {"a_phase": a_ph, "t_phase": dt_ph, "a_target": t.a, "i_target": t.i}


def closed_loop_phasing(tl: RpoTimeline, k_phase: int, stop_sep: float) -> int:
    """Burns 4.x: same-orbit phasing, re-planned every rev from the chaser's actual state.

    Each burn takes the chaser from its current speed to the vis-viva speed of the phasing
    orbit for the remaining revs. Stops early once separation < `stop_sep` [m].
    Returns the number of phasing revs left unused.
    """
    phases_remaining = k_phase  # Start at max

    while phases_remaining:
        # sma of phasing orbit (USING ARGUMENT OF LATITUDE)
        c, t = tl.coes()
        _, _, a_phase = guid_phasing_same_orbit(
            t.a,
            wrap_to_pi(c.aol - t.aol),
            tl.mu,
            phases_remaining,
            phases_remaining,
            deg=False,
            r_cb=tl.r_cb,
        )

        # Burn along chaser prograde
        r_now = norm(tl.chaser[0:3])
        v_req = np.sqrt(tl.mu * (2 / r_now - 1 / a_phase))
        tl.burn(
            (v_req - norm(tl.chaser[3:6])) * unit(tl.chaser[3:6]),
            f"Burn 4.{k_phase - phases_remaining} (phasing)",
        )
        phases_remaining -= 1

        # Coast and break condition
        c, _ = tl.coes()
        tl.coast(c.period(tl.mu))
        if norm(tl.to_rsw()[0:3]) < stop_sep:
            break

    return phases_remaining


def phase_exit(tl: RpoTimeline) -> None:
    """Burn 5: along-track burn to the target's semi-major axis."""
    # Duplicate of last part of closed-loop-phasing
    _, t = tl.coes()
    r_now = norm(tl.chaser[0:3])
    v_req = np.sqrt(tl.mu * (2 / r_now - 1 / t.a))
    tl.burn(
        (v_req - norm(tl.chaser[3:6])) * unit(tl.chaser[3:6]), "Burn 5 (phase exit)"
    )


def hcw_rendezvous(
    tl: RpoTimeline, t_transfer: float, dock_thresh: float, max_iter: int = 3
) -> int:
    """HCW two-impulse rendezvous, re-targeted after each transfer until the miss < `dock_thresh` [m].

    Every pass applies the first impulse from the current relative state and coasts
    `t_transfer` [s]; only the final pass's second impulse is applied (earlier ones are
    superseded by the next re-target). Returns the number of passes used.
    """
    for k in range(max_iter):
        # Guidance
        rv_rsw = tl.to_rsw()
        dv1_rsw, dv2_rsw = hcw_two_impulse(
            rv_rsw[0:3], rv_rsw[3:6], t_transfer, tl.coes()[1].n(tl.mu)
        )

        # Apply
        rv_rsw[3:6] += dv1_rsw
        tl.burn(tl.from_rsw(rv_rsw)[3:6] - tl.chaser[3:6], f"HCW {k}A")
        tl.coast(t_transfer)

        # Stop condition
        if norm(tl.to_rsw()[0:3]) < dock_thresh:
            break

    # applies burn-to-go
    rv_rsw = tl.to_rsw()
    rv_rsw[3:6] += dv2_rsw
    tl.burn(tl.from_rsw(rv_rsw)[3:6] - tl.chaser[3:6], f"HCW {k}B")
    return k + 1


################################################################################
#               Running
################################################################################


def run_rpo(
    chaser0: COES,
    target0: COES,
    mu: float,
    k_phase: int,
    hcw_thresh: float,
    dock_thresh: float,
    t_rendezvous: float,
    *,
    hcw_iter: int = 3,
    raan_bias: float = 0.0,
    epoch0_tdb_s: float = 0.0,
    j2: float | None = None,
    r_cb: float | None = None,
    frame: str = "MCI",
    t_final_coast: float = 1 * DAY_TO_SEC,
    verbose: bool = False,
) -> tuple[RpoTimeline, dict]:
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
    # Burn 1
    info = {"n_hat": plane_change_at_node(tl, raan_bias)}

    # Burn 2
    dv_circ = hohmann_to_target_radius(tl)

    # Burns 3 and 4.#
    info |= enter_phasing(tl, k_phase, dv_circ)
    info["phases_remaining"] = closed_loop_phasing(tl, k_phase, hcw_thresh)
    handoff = relative_summary(tl)

    # Burn 5
    phase_exit(tl)

    # Burn 6+
    info["hcw_passes"] = hcw_rendezvous(tl, t_rendezvous, dock_thresh, hcw_iter)
    dock = relative_summary(tl)

    tl.coast(t_final_coast)

    # Verbose
    if verbose:
        print_relative_summary(
            f"Phasing exit ({info['phases_remaining']} revs unused)", handoff
        )
        print_relative_summary(f"HCW exit ({info['hcw_passes']} passes)", dock)
        print(f"total dv = {tl.dv_total:.3f} m/s")

    # Info
    info["handoff_rsw"], info["draan"] = handoff["rsw"], handoff["draan"]
    info["docking_rsw"], info["dock_draan"] = dock["rsw"], dock["draan"]

    return tl, info


################################################################################
#               Helpers
################################################################################


def relative_summary(tl: RpoTimeline) -> dict:
    """Chaser position relative to the target (RSW) [m] and RAAN difference [rad], now."""
    c, t = tl.coes()
    return {"rsw": tl.to_rsw()[0:3], "draan": wrap_to_pi(c.raan - t.raan)}


def print_relative_summary(label: str, summary: dict) -> None:
    r = summary["rsw"]
    print(
        f"{label}: sep {norm(r) * M_TO_KM:.3f} km "
        f"(R {r[0] * M_TO_KM:.3f}, S {r[1] * M_TO_KM:.3f}, W {r[2] * M_TO_KM:.3f} km), "
        f"dRAAN {summary['draan'] * RAD_TO_DEG:.4f} deg"
    )
