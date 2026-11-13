"""Chaser/target timeline: one propagated truth with impulsive chaser burns."""

import numpy as np

from ..constants import SEC_TO_HOUR
from ..math.time import jd_tdb_to_tt, jd_to_greg_str, jd_tt_to_utc, sec_j2000_to_jd
from ..orbital.coes import COES
from ..orbital.hcw import hcw_pci_to_rsw, hcw_rsw_to_pci
from ..physics.dynamics import propagate_two_body
from .maneuvers import Burn


class RpoTimeline:
    """Chaser and target that can be propagated

    `coast` propagates both spacecraft from the latest state and `burn` applies an
    impulsive delta-v to the chaser's latest state, so each burn can be planned
    from the current propagated COES. Times `t_s` are seconds since the
    timeline's start, whose absolute epoch is `epoch0_tdb_s` (TDB seconds past
    J2000). Keeping times relative preserves float precision; add
    `epoch0_tdb_s` (or use `epochs_tdb_s`) for absolute epochs.

    Parameters
    ----------
    rv_chaser0, rv_target0 : np.ndarray (6,)
        Initial chaser and target states in one inertial PCI frame [m, m/s].
    mu : float
        Gravitational parameter of the central body [m3/s2].
    j2 : float, optional
        J2 coefficient. If None, J2 is not applied.
    r_cb : float, optional
        Equatorial radius of the central body [m]. Required if `j2` is given.
    epoch0_tdb_s : float, optional
        Epoch of `t_s = 0`, in TDB seconds past J2000. Default 0.0.
    frame : str, optional
        Name of the PCI axes, stored on each `Burn` (e.g. "MCI"). Default "PCI".
    """

    def __init__(
        self,
        rv_chaser0: np.ndarray,
        rv_target0: np.ndarray,
        mu: float,
        *,
        j2: float = None,
        r_cb: float = None,
        epoch0_tdb_s: float = 0.0,
        frame: str = "PCI",
    ):
        if j2 is not None and r_cb is None:
            raise ValueError("r_cb is required when j2 is given")
        self.mu, self.j2, self.r_cb = mu, j2, r_cb
        self.epoch0_tdb_s = float(epoch0_tdb_s)
        self.frame = frame
        self._ts = [np.array([0.0])]
        self._rv_c = [np.asarray(rv_chaser0, dtype=float)[None].copy()]
        self._rv_t = [np.asarray(rv_target0, dtype=float)[None].copy()]
        self.burns: list[Burn] = []
        """Burns in the order applied; `t_s` is relative to `epoch0_tdb_s`, axes `frame`."""

    # ------------------------------------------------------------------
    # Current state
    # ------------------------------------------------------------------
    @property
    def t(self) -> float:
        """Current time since the timeline start [s]."""
        return float(self._ts[-1][-1])

    @property
    def epoch_tdb_s(self) -> float:
        """Current epoch [TDB s past J2000]."""
        return self.epoch0_tdb_s + self.t

    @property
    def epoch_jd_tdb(self) -> float:
        """Current epoch [Julian date, TDB]."""
        return sec_j2000_to_jd(self.epoch_tdb_s)

    @property
    def chaser(self) -> np.ndarray:
        """Current chaser state (copy) [m, m/s]."""
        return self._rv_c[-1][-1].copy()

    # @property
    # def chaser_coes(self) -> COES:
    #     """Current chaser state (copy) coes."""
    #     return COES.from_rv(self.chaser, self.mu)

    @property
    def target(self) -> np.ndarray:
        """Current target state (copy) [m, m/s]."""
        return self._rv_t[-1][-1].copy()

    # @property
    # def target_coes(self) -> COES:
    #     """Current target state (copy) coes."""
    #     return COES.from_rv(self.target, self.mu)

    def coes(self) -> tuple[COES, COES]:
        """(chaser, target) COES of the current propagated states."""
        return COES.from_rv(self.chaser, self.mu), COES.from_rv(self.target, self.mu)

    def to_rsw(self) -> np.ndarray:
        """Current chaser state relative to the target, in the target's RSW frame [m, m/s]."""
        return hcw_pci_to_rsw(self.target, self.chaser)

    def from_rsw(self, rv_chaser_rsw: np.ndarray) -> np.ndarray:
        """Absolute PCI chaser state for a relative RSW state about the current target [m, m/s]."""
        return hcw_rsw_to_pci(self.target, rv_chaser_rsw)

    # ------------------------------------------------------------------
    # Building the timeline
    # ------------------------------------------------------------------
    def coast(self, dt: float, n: int = 1000) -> None:
        """Propagate both spacecraft forward by `dt` [s], sampled at `n` points."""
        if dt < 0:
            raise ValueError("dt must be non-negative")
        if dt == 0:
            return
        ts = np.linspace(self.t, self.t + dt, n)

        def prop(rv):
            return propagate_two_body(rv, ts, mu=self.mu, j2=self.j2, r_cb=self.r_cb)[
                1:
            ]

        self._ts.append(ts[1:])
        self._rv_c.append(prop(self.chaser))
        self._rv_t.append(prop(self.target))

    def coast_until(self, t: float, n: int = 1000) -> None:
        """Coast to time `t` since the timeline start [s]."""
        self.coast(t - self.t, n)

    def burn(self, dv: np.ndarray, name: str = "") -> Burn:
        """Apply an impulsive chaser delta-v [m/s] (PCI) at the current time.

        The last stored chaser sample is replaced by the post-burn state.
        """
        dv = np.asarray(dv, dtype=float)
        self._rv_c[-1] = self._rv_c[-1].copy()
        self._rv_c[-1][-1, 3:] += dv
        burn = Burn(t_s=self.t, r=self.chaser[0:3], dv=dv, frame=self.frame, name=name)
        self.burns.append(burn)
        return burn

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(ts [s], rv_chaser (N, 6), rv_target (N, 6)) over the whole timeline."""
        return (
            np.concatenate(self._ts),
            np.concatenate(self._rv_c),
            np.concatenate(self._rv_t),
        )

    def epochs_tdb_s(self) -> np.ndarray:
        """Absolute epochs of every sample in `arrays()` [TDB s past J2000]."""
        return self.epoch0_tdb_s + np.concatenate(self._ts)

    def rsw_arrays(self, progress: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """(ts [s], rv_rel_rsw (N, 6)): chaser relative to the target in the target's RSW frame.

        `progress=True` shows a tqdm bar (requires tqdm).
        """
        ts, rv_c, rv_t = self.arrays()
        pairs = zip(rv_t, rv_c)
        if progress:
            from tqdm import tqdm

            pairs = tqdm(pairs, total=len(ts))
        return ts, np.array([hcw_pci_to_rsw(t_, c_) for t_, c_ in pairs])

    def rel_at(self, t):
        """Chaser RSW state relative to the target at the sample nearest t [m, m/s]."""
        ts, rv_c, rv_t = self.arrays()
        k = min(np.searchsorted(ts, t), len(ts) - 1)
        return hcw_pci_to_rsw(rv_t[k], rv_c[k])

    @property
    def dv_total(self) -> float:
        """Sum of burn magnitudes [m/s]."""
        return float(sum(b.dv_mag for b in self.burns))

    def burn_events(self) -> list[tuple[float, str]]:
        """[(t_s, name), ...] for marking burns, e.g. in `plot_rsw` or `plot_separation`."""
        return [(b.t_s, b.name) for b in self.burns]

    def burn_table(self) -> str:
        """Text table of burns: time since start, absolute epoch (TDB JD and UTC), |dv|, and name."""
        rows = [
            f"{'#':>3}  {'t [hr]':>9}  {'JD (TDB)':>17}  {'UTC':>19}  {'|dv| [m/s]':>10}  name"
        ]
        for i, b in enumerate(self.burns):
            jd = sec_j2000_to_jd(self.epoch0_tdb_s + b.t_s)
            utc = jd_to_greg_str(jd_tt_to_utc(jd_tdb_to_tt(jd)))
            rows.append(
                f"{i:>3}  {b.t_s * SEC_TO_HOUR:>9.3f}  {jd:>17.6f}  {utc:>19}  {b.dv_mag:>10.3f}  {b.name}"
            )
        rows.append(
            f"{'':>3}  {'':>9}  {'':>17}  {'total':>19}  {self.dv_total:>10.3f}"
        )
        return "\n".join(rows)
