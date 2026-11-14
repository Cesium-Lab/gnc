from dataclasses import dataclass

import numpy as np
from numpy.linalg import norm

from ..constants import DAY_TO_SEC, M_TO_KM, RAD_TO_DEG, SEC_TO_HOUR
from ..math.base import unit
from ..math.rotations import wrap_to_2pi, wrap_to_pi
from ..orbital.coes import COES, orbit_sweep, vis_viva
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
#               Tolerances
################################################################################

PLANE_TOL_RAD = 1e-6
"""Plane angle below which Burn 1 is skipped (the node line is undefined for coplanar orbits)."""

SHAPE_TOL_M = 1.0
"""Apsis-radius error below which Burn 2 is skipped."""


################################################################################
#               Pure planners: state / elements in, numbers out (no timeline)
################################################################################


def shape_matches(chaser: COES, a_target: float, tol: float = SHAPE_TOL_M) -> bool:
    """True if both chaser apses are within `tol` [m] of `a_target` (no Hohmann needed)."""
    return max(abs(chaser.r_p - a_target), abs(chaser.r_a - a_target)) < tol


def plan_plane_change(
    rv_chaser: np.ndarray,
    target: COES,
    mu: float,
    raan_bias: float = 0.0,
    tol: float = PLANE_TOL_RAD,
) -> tuple[np.ndarray, float] | None:
    """Node line with the target plane (RAAN offset by `raan_bias` [rad]) and the
    chaser's true-anomaly sweep to it.

    Returns (node-line unit vector, sweep in `[0, 2*pi)` [rad]), or None if the planes
    already agree to within `tol` [rad] (the node line is then undefined).
    """
    rv_target_biased = target.bias_raan(raan_bias).to_rv(mu)

    # Check before node_line_finder, which can't normalize a zero node vector
    h_c = np.cross(rv_chaser[0:3], rv_chaser[3:6])
    h_t = np.cross(rv_target_biased[0:3], rv_target_biased[3:6])
    if norm(np.cross(unit(h_c), unit(h_t))) < tol:
        return None
    n_hat, d_ta, _, _ = node_line_finder(rv_chaser, rv_target_biased)
    return n_hat, d_ta


def plan_coast_to_nearer_apsis(chaser: COES, mu: float) -> tuple[float, bool]:
    """(coast time [s], )."""
    closer_to_periapsis = chaser.ta > np.pi
    sweep = (
        (2 * np.pi - chaser.ta)
        if closer_to_periapsis
        else wrap_to_2pi(np.pi - chaser.ta)
    )
    _, dt = orbit_sweep(chaser, sweep, mu)
    return dt, closer_to_periapsis


def plan_hohmann_from_apsis(
    chaser: COES, a_target: float, at_periapsis: bool, mu: float
) -> tuple[float, float, float]:
    """Hohmann from the chaser's current apsis to a circle of radius `a_target`.

    Returns (dv_a [m/s], transfer time [s], dv_b [m/s]); both burns are along-track
    (positive prograde), since velocity is purely along-track at an apsis.
    """
    _, dt_trans, dv_a, dv_b = guid_coaxial_hohmann(
        chaser.r_p,
        chaser.r_a,
        a_target,
        a_target,
        depart_at_periapsis=at_periapsis,
        mu=mu,
    )
    return dv_a, dt_trans, dv_b


def plan_along_track_to_sma(rv: np.ndarray, a: float, mu: float) -> np.ndarray:
    """Along-track dv [m/s] that puts the orbit through `rv` onto semi-major axis `a` [m].

    Changes the speed to vis-viva at the current radius; direction is unchanged.
    """
    v_req = vis_viva(norm(rv[0:3]), a, mu)
    return (v_req - norm(rv[3:6])) * unit(rv[3:6])


def plan_enter_phasing(
    rv_chaser: np.ndarray,
    target: COES,
    *,
    mu: float,
    k_phase: int,
    dv_circ: float = 0.0,
    r_cb: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Combine a "would-be-applied" circularization burn (`dv_circ` [m/s], along-track) and `k_phase` phasing-entry in one burn

    Returns (dv vector [m/s], phasing semi-major axis [m], total phasing time [s]).
    """
    # Phase angle of the chaser WITH the circularization burn-to-go applied
    v_hat = unit(rv_chaser[3:6])
    rv_circ = rv_chaser.copy()
    rv_circ[3:6] += dv_circ * v_hat
    c_circ = COES.from_rv(rv_circ, mu)

    dv_ph, dt_ph, a_ph = guid_phasing_same_orbit(
        target.a,
        wrap_to_pi(c_circ.aol - target.aol),
        mu,
        k_phase,
        k_phase,
        deg=False,
        r_cb=r_cb,
    )
    # dv_ph is the round trip (enter + exit); only the entry half is applied here
    v_phase_enter = (dv_circ + dv_ph / 2) * v_hat
    return v_phase_enter, a_ph, dt_ph


################################################################################
#               Wrappers that act on the RpoTimeline
################################################################################


def plane_change_at_node(tl: RpoTimeline, raan_bias: float = 0.0) -> np.ndarray | None:
    """Burn 1: coast to the node with the target plane (RAAN offset by `raan_bias` [rad]) and rotate into it.

    Returns the node-line unit vector, or None (no burn) when the planes already
    agree to within `PLANE_TOL_RAD`.
    """
    c, t = tl.coes()
    plan = plan_plane_change(tl.chaser, t, tl.mu, raan_bias)
    if plan is None:
        return None
    n_hat, d_ta = plan

    _, dt = orbit_sweep(c, d_ta, tl.mu)
    tl.coast(dt)
    _, t = tl.coes()  # re-observe: the target plane drifts under J2 during the coast
    h_new_target = t.bias_raan(raan_bias).h(tl.mu)
    tl.burn(
        guid_plane_change_3d(tl.chaser, h_new_target),
        "Burn 1 (plane)",
    )
    return n_hat


def hohmann_to_target_radius(tl: RpoTimeline) -> float:
    """Burn 2: from the nearer apsis, Hohmann toward the target's semi-major axis.

    Applies the first burn and coasts the transfer. Returns the (unapplied) second-burn
    to-go [m/s] to merge with the next maneuver. Skipped (returns 0) when both apses are
    already within `SHAPE_TOL_M` of the target's semi-major axis.
    """
    c, t = tl.coes()
    if shape_matches(c, t.a):
        return 0.0

    dt_coast, closer_to_periapsis = plan_coast_to_nearer_apsis(c, tl.mu)
    tl.coast(dt_coast)

    c, t = tl.coes()  # re-observe at the apsis
    dv_a, dt_trans, dv_b = plan_hohmann_from_apsis(c, t.a, closer_to_periapsis, tl.mu)
    tl.burn(dv_a * unit(tl.chaser[3:6]), "Burn 2 (Hohmann)")
    tl.coast(dt_trans)
    return dv_b


def enter_phasing(tl: RpoTimeline, k_phase: int, dv_circ: float = 0.0) -> dict:
    """Burn 3: circularize (merging Burn 2's to-go `dv_circ` [m/s]) and enter phasing.

    Coasts the first phasing rev. Returns the plan: phasing semi-major axis and
    duration, target a and i.
    """
    _, t = tl.coes()
    dv_to_enter, a_full_ph, dt_full_ph = plan_enter_phasing(
        tl.chaser, t, mu=tl.mu, k_phase=k_phase, dv_circ=dv_circ, r_cb=tl.r_cb
    )
    tl.burn(dv_to_enter, "Burn 3 (circularize + phase entry)")
    tl.coast(dt_full_ph / k_phase)
    return {
        "a_phase": a_full_ph,
        "t_phase": dt_full_ph,
        "a_target": t.a,
        "i_target": t.i,
    }


def closed_loop_phasing(tl: RpoTimeline, k_phase: int, stop_sep: float) -> int:
    """Burns 4.x: same-orbit phasing, re-planned every rev from the chaser's actual state.

    Each burn takes the chaser from its current speed to the vis-viva speed of the phasing
    orbit for the remaining revs. Stops early once separation < `stop_sep` [m].
    Returns the number of phasing revs left unused.
    """
    phases_remaining = k_phase

    while phases_remaining:
        # Phasing orbit for the remaining revs (phase angle by argument of latitude)
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
        tl.burn(
            plan_along_track_to_sma(tl.chaser, a_phase, tl.mu),
            f"Burn 4.{k_phase - phases_remaining} (phasing)",
        )
        phases_remaining -= 1

        c, _ = tl.coes()
        tl.coast(c.period(tl.mu))
        if norm(tl.to_rsw()[0:3]) < stop_sep:
            break

    return phases_remaining


def phase_exit(tl: RpoTimeline) -> None:
    """Burn 5: along-track burn back onto the target's semi-major axis."""
    _, t = tl.coes()
    tl.burn(plan_along_track_to_sma(tl.chaser, t.a, tl.mu), "Burn 5 (phase exit)")


# TODO: test
def hcw_rendezvous(
    tl: RpoTimeline, t_transfer: float, dock_thresh: float, max_iter: int = 3
) -> int:
    """HCW two-impulse rendezvous, re-targeted after each transfer until the miss < `dock_thresh` [m].

    Every pass applies the first impulse from the current relative state and coasts
    `t_transfer` [s]; only the final pass's second impulse is applied (earlier ones are
    superseded by the next re-target). Returns the number of passes used; with
    `max_iter < 1` nothing is planned or applied and it returns 0.
    """
    if max_iter < 1:
        return 0

    for k in range(max_iter):
        # Guidance
        rv_rsw = tl.to_rsw()
        dv1_rsw, dv2_rsw = hcw_two_impulse(
            rv_rsw[0:3], rv_rsw[3:6], t_transfer, tl.coes()[1].n(tl.mu)
        )

        # Apply
        rv_rsw[3:6] += dv1_rsw
        name = "HCW start" if k == 0 else f"HCW retarget {k}"
        tl.burn(tl.from_rsw(rv_rsw)[3:6] - tl.chaser[3:6], name)
        tl.coast(t_transfer)

        # Stop condition
        if norm(tl.to_rsw()[0:3]) < dock_thresh:
            break

    # applies burn-to-go
    rv_rsw = tl.to_rsw()
    rv_rsw[3:6] += dv2_rsw
    tl.burn(tl.from_rsw(rv_rsw)[3:6] - tl.chaser[3:6], "HCW stop")
    return k + 1


################################################################################
#               Running
################################################################################


@dataclass(frozen=True, kw_only=True)
class RpoConfig:
    """Scenario settings for `run_rpo`: central body, timeline, and maneuver tuning."""

    # Central body
    mu: float
    """Gravitational parameter [m^3/s^2]."""
    j2: float | None = None
    """J2 coefficient; None for two-body truth."""
    r_cb: float | None = None
    """Equatorial radius [m]; required when `j2` is set."""
    frame: str = "MCI"
    """Inertial axes name, for burn labels."""

    # Timeline
    epoch0_tdb_s: float = 0.0
    """Absolute epoch of t = 0 [TDB s past J2000]."""
    t_final_coast: float = 1 * DAY_TO_SEC
    """Coast after docking, for plots [s]."""

    # Phasing
    k_phase: int
    """Phasing revolutions."""
    hcw_thresh: float
    """Separation that ends phasing and hands off to HCW [m]."""

    # HCW terminal
    t_rendezvous: float
    """HCW two-impulse transfer time [s]."""
    dock_thresh: float
    """Miss distance that ends HCW re-targeting [m]."""
    hcw_iter: int = 3
    """Maximum HCW passes."""


def run_rpo(
    chaser0: COES,
    target0: COES,
    cfg: RpoConfig,
    *,
    raan_bias: float = 0.0,
    verbose: bool = False,
) -> tuple[RpoTimeline, dict]:
    """Full RPO timeline. `raan_bias` [rad] pre-biases the Burn 1 plane against J2 nodal drift."""
    tl = RpoTimeline(
        chaser0.to_rv(cfg.mu),
        target0.to_rv(cfg.mu),
        cfg.mu,
        j2=cfg.j2,
        r_cb=cfg.r_cb,
        epoch0_tdb_s=cfg.epoch0_tdb_s,
        frame=cfg.frame,
    )
    # Burn 1
    info = {"n_hat": plane_change_at_node(tl, raan_bias)}

    # Burn 2
    dv_to_circularize = hohmann_to_target_radius(tl)

    # Burns 3 and 4.x
    info |= enter_phasing(tl, cfg.k_phase, dv_to_circularize)
    info["phases_remaining"] = closed_loop_phasing(tl, cfg.k_phase, cfg.hcw_thresh)
    handoff = relative_summary(tl)

    # Burn 5
    phase_exit(tl)

    # Burn 6+
    info["hcw_passes"] = hcw_rendezvous(
        tl, cfg.t_rendezvous, cfg.dock_thresh, cfg.hcw_iter
    )
    dock = relative_summary(tl)

    tl.coast(cfg.t_final_coast)

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


RPO_PHASES = (
    ("Plane change", ("Burn 1 ",)),
    ("Hohmann + phase entry", ("Burn 2 ", "Burn 3 ")),
    ("Phasing", ("Burn 4.",)),
    ("Phase exit", ("Burn 5 ",)),
    ("HCW terminal", ("HCW",)),
)
"""(phase label, burn-name prefixes) in mission order, matching `run_rpo`'s burn names."""


def dv_breakdown(tl: RpoTimeline, phases=RPO_PHASES) -> list[dict]:
    """Group a timeline's burns by phase.

    Returns one dict per phase that has burns: phase, n_burns, dv [m/s], frac of total,
    t_start and t_end [s since the timeline epoch], first_burn and last_burn names.
    Burns matching no phase go in "Other".
    """
    groups = {label: [] for label, _ in phases}
    groups["Other"] = []
    for b in tl.burns:
        label = next(
            (lab for lab, prefixes in phases if b.name.startswith(prefixes)), "Other"
        )
        groups[label].append(b)
    total = tl.dv_total
    return [
        {
            "phase": label,
            "n_burns": len(bs),
            "dv": sum(b.dv_mag for b in bs),
            "frac": sum(b.dv_mag for b in bs) / total if total else 0.0,
            "t_start": bs[0].t_s,
            "t_end": bs[-1].t_s,
            "first_burn": bs[0].name,
            "last_burn": bs[-1].name,
        }
        for label, bs in groups.items()
        if bs
    ]


def dv_breakdown_table(tl: RpoTimeline, phases=RPO_PHASES) -> str:
    """`dv_breakdown` as a fixed-width text table."""
    rows = dv_breakdown(tl, phases)
    lines = [
        f"{'phase':<24}{'burns':>6}{'t [hr]':>16}{'dv [m/s]':>11}{'share':>8}",
        "-" * 65,
    ]
    for r in rows:
        t = f"{r['t_start'] * SEC_TO_HOUR:.1f}-{r['t_end'] * SEC_TO_HOUR:.1f}"
        lines.append(
            f"{r['phase']:<24}{r['n_burns']:>6}{t:>16}{r['dv']:>11.2f}{r['frac']:>8.1%}"
        )
    lines.append("-" * 65)
    lines.append(f"{'total':<24}{len(tl.burns):>6}{'':>16}{tl.dv_total:>11.2f}{'':>8}")
    return "\n".join(lines)


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
