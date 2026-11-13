from dataclasses import dataclass

import numpy as np
from numpy.linalg import norm

from ..constants import DAY_TO_SEC, M_TO_KM, RAD_TO_DEG, SEC_TO_HOUR
from ..math.base import unit
from ..math.rotations import wrap_to_2pi, wrap_to_pi
from ..orbital.coes import COES, orbit_sweep, vis_viva
from ..orbital.hcw import calc_Phi, hcw_pci_to_rsw
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
#               Plane-changing
################################################################################


# TODO: test
PLANE_TOL_RAD = 1e-6
"""Plane angle below which Burn 1 is skipped (the node line is undefined for coplanar orbits)."""

SHAPE_TOL_M = 1.0
"""Apsis-radius error below which Burn 2 is skipped."""


def plane_change_at_node(tl: RpoTimeline, raan_bias: float = 0.0) -> np.ndarray | None:
    """Burn 1: coast to the node with the target plane (RAAN offset by `raan_bias` [rad]) and rotate into it.

    Returns the node-line unit vector, or None (no burn) when the planes already
    agree to within `PLANE_TOL_RAD`.
    """
    # Node
    c, t = tl.coes()
    rv_target_biased = t.bias_raan(raan_bias).to_rv(tl.mu)
    n_hat, d_node, h_c, h_t = node_line_finder(tl.chaser, rv_target_biased)
    if norm(np.cross(unit(h_c), unit(h_t))) < PLANE_TOL_RAD:
        return None

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

def plan_hohmann_from_apsis(
    chaser: COES, target_a: float, at_periapsis: bool, mu: float
) -> tuple[float, float, float] | None:
    """(dv_a [m/s] along-track, transfer time [s], dv_b [m/s] along-track),
    or None if both apses are already within SHAPE_TOL_M of target_a."""


# # TODO: test
# def coast_to_nearer_apsis(tl: RpoTimeline) -> bool:
#     """Coast the chaser to its nearer apsis; returns True if that apsis is periapsis."""
#     c, _ = tl.coes()
#     periapsis_is_closer = c.ta > np.pi
#     sweep = (2 * np.pi - c.ta) if periapsis_is_closer else wrap_to_2pi(np.pi - c.ta)
#     tl.coast(orbit_sweep(c, sweep, tl.mu)[1])
#     return periapsis_is_closer

# TODO: test
def plan_coast_to_nearer_apsis(chaser: COES, mu: float) -> tuple[float, bool]:
    """(coast time [s], True if the nearer apsis is periapsis)."""
    periapsis_is_closer = chaser.ta > np.pi
    sweep = (2 * np.pi - chaser.ta) if periapsis_is_closer else wrap_to_2pi(np.pi - chaser.ta)
    _, dt = orbit_sweep(chaser, sweep, mu)
    return dt, periapsis_is_closer


# TODO: test
def hohmann_to_target_radius(tl: RpoTimeline) -> float:
    """Burn 2: from the nearer closest apsis do Hohmann toward the target sma

    Applies the first burn and coasts the transfer. Returns the (unapplied) second-burn
    to-go to merge with the next maneuver. Skipped (returns 0) when both apses are
    already within `SHAPE_TOL_M` of the target's semi-major axis.
    """
    c, t = tl.coes()
    if max(abs(c.r_p - t.a), abs(c.r_a - t.a)) < SHAPE_TOL_M:
        return 0.0  # already on the target radius: no transfer needed
    
    dt, periapsis_is_closer = plan_coast_to_nearer_apsis(c, tl.mu)
    tl.coast(dt)

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

# TODO: test
def enter_phasing(
    rv_c: np.ndarray,
    target: COES,
    *,
    mu: float,
    r_cb: float,
    k_phase: int,
    dv_circ: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """Burn 3: circularize (`dv_circ` [m/s], along-track) and enter a `k_phase`-rev phasing orbit in one burn.

    Returns (Burn 3 dv vector [m/s], phasing semi-major axis [m], total phasing time [s]).
    """
    # Anomaly of chaser WITH the burn-to-go applied
    v_hat = unit(rv_c[3:6])
    rv_circ = rv_c.copy()
    rv_circ[3:6] += dv_circ * v_hat
    c_circ = COES.from_rv(rv_circ, mu)

    # Phase velocity
    dv_ph, dt_ph, a_ph = guid_phasing_same_orbit(
        target.a,
        wrap_to_pi(c_circ.aol - target.aol),
        mu,
        k_phase,
        k_phase,
        deg=False,
        r_cb=r_cb,
    )

    dv_enter = (dv_circ + dv_ph / 2) * v_hat
    return dv_enter, a_ph, dt_ph

# TODO: test
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
        v_req = vis_viva(r_now, a_phase, tl.mu)
        tl.burn(
            (v_req - norm(tl.chaser[3:6])) * unit(tl.chaser[3:6]),
            f"Burn 4.{k_phase - phases_remaining} (phasing)",
        )
        phases_remaining -= 1

        # Coast and break condition
        c, _ = tl.coes()
        tl.coast(c.period(tl.mu))
        # sep = c.to_r_and_v[0] - t.to_r_and_v[0]
        if norm(tl.to_rsw()[0:3]) < stop_sep:
            break

    return phases_remaining


# TODO: test
# def phase_exit(tl: RpoTimeline) -> None:
#     """Burn 5: along-track burn to the target's semi-major axis."""
#     # Duplicate of last part of closed-loop-phasing
#     _, t = tl.coes()
#     r_now = norm(tl.chaser[0:3])
#     v_req = vis_viva(r_now, t.a, tl.mu)
#     tl.burn(
#         (v_req - norm(tl.chaser[3:6])) * unit(tl.chaser[3:6]), "Burn 5 (phase exit)"
#     )

def phase_exit(chaser_rv: np.ndarray, a_target: float, mu: float) -> np.ndarray:
    """Burn 5: along-track burn to the target's semi-major axis."""
    # Duplicate of last part of closed-loop-phasing
    r_now = norm(chaser_rv[0:3])
    v_req = vis_viva(r_now, a_target, mu)
    v_burn = (v_req - norm(chaser_rv[3:6])) * unit(chaser_rv[3:6])
    return v_burn


# TODO: test
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


# TODO: test
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


# TODO: test
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
    n_hat = plane_change_at_node(tl, raan_bias)
    info = {"n_hat": n_hat}

    # Burn 2
    dv_circ = hohmann_to_target_radius(tl)

    # Burns 3 and 4.#
    
    
    # Apply, considering burn-to-go from before
    _, t = tl.coes()
    dv_enter, a_ph, dt_ph = enter_phasing(
        tl.chaser, t, mu=tl.mu, r_cb=tl.r_cb, k_phase=cfg.k_phase, dv_circ=dv_circ
    )
    tl.burn(dv_enter, "Burn 3 (circularize + phase entry)")
    tl.coast(dt_ph / cfg.k_phase)
    info |= {"a_phase": a_ph, "t_phase": dt_ph, "a_target": t.a, "i_target": t.i}


    info["phases_remaining"] = closed_loop_phasing(tl, cfg.k_phase, cfg.hcw_thresh)
    handoff = relative_summary(tl)

    # Burn 5
    _, t = tl.coes()
    v_burn = phase_exit(tl.chaser, t.a, tl.mu)
    tl.burn(v_burn, "Burn 5 (phase exit)")
    
    # phase_exit(tl)

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
