"""End-to-end tests of `run_rpo` on the mars-courier scenario (from `rpo.ipynb`).

Two kinds of checks:
- Physical: docking, final plane, burn sequence, Δv bookkeeping. These should hold
  for any correct RPO implementation.
- Regression: total Δv pinned to the current implementation, so the refactor into
  pure guidance functions can't silently change the result.
"""

import numpy as np
import pytest
from numpy.linalg import norm

from cesium.constants import KM_TO_M, MIN_TO_SEC
from cesium.guidance.hcw import RpoConfig, dv_breakdown, run_rpo
from cesium.orbital.coes import COES
from cesium.world.constants import J2_MARS, MU_MARS, R_MARS


def raan_dot_j2(a, e, i):
    """Secular J2 nodal regression rate [rad/s]."""
    n = np.sqrt(MU_MARS / a**3)
    p = a * (1 - e**2)
    return -1.5 * n * J2_MARS * (R_MARS / p) ** 2 * np.cos(i)


def solve_case(chaser0, target0, cfg):
    """Pass 1 measures the J2 RAAN drift during phasing; pass 2 pre-biases Burn 1. Keeps the cheaper."""
    tl_0, info_0 = run_rpo(chaser0, target0, cfg)

    bias = -info_0["draan"]
    tl_b, info_b = run_rpo(chaser0, target0, cfg, raan_bias=bias)
    use_bias = tl_b.dv_total < tl_0.dv_total

    tl_, info_ = (tl_b, info_b) if use_bias else (tl_0, info_0)
    return {
        "tl": tl_,
        "info": info_,
        "raan_bias": bias if use_bias else 0.0,
        "tl_unbiased": tl_0,
        "info_unbiased": info_0,
    }


################################################################################
#               Scenario (from rpo.ipynb)
################################################################################

CFG = RpoConfig(
    mu=MU_MARS,
    j2=J2_MARS,
    r_cb=R_MARS,
    epoch0_tdb_s=1135425600.0,
    k_phase=20,
    hcw_thresh=20 * KM_TO_M,
    t_rendezvous=40 * MIN_TO_SEC,
    dock_thresh=50,  # m
)

TARGET = COES(
    a=3696000.0,
    e=0.0,
    i=0.7853981633974483,
    raan=2.181661564992912,
    aop=0.0,
    ta=0.22689280275926285,
)
CHASER_NOM = COES(
    a=3696000.0000000014,
    e=3.237139536196805e-16,
    i=0.7853981633974483,
    raan=2.181661564992912,
    aop=0.0,
    ta=2.1099010234134723,
)
CHASER_DISPERSED = COES(
    a=3768959.5174033185,
    e=0.008553025149320914,
    i=0.8179684575546289,
    raan=2.156680286377542,
    aop=-0.044275569252186266,
    ta=2.1099010234134723,
)

CASES = {"nominal": CHASER_NOM, "dispersed": CHASER_DISPERSED}

# Regression pins [m/s]: chosen (cheaper) plan, as of the mars-courier report
DV_TOTAL_PIN = {"nominal": 69.82710774931797, "dispersed": 185.980036961606}


@pytest.fixture(scope="module", params=list(CASES))
def case(request):
    """Solve each case once per module; every test below reuses it."""
    name = request.param
    return {"name": name, **solve_case(CASES[name], TARGET, CFG)}


################################################################################
#               Tests
################################################################################


class Test_run_rpo:
    """Regression for refactoring right now"""
    def test_docks_within_threshold(self, case):
        dock_rsw = case["info"]["docking_rsw"]
        assert norm(dock_rsw[0:3]) < CFG.dock_thresh
        # HCW stop burn nulls the relative velocity
        assert norm(dock_rsw[3:6]) == pytest.approx(0.0, abs=1e-6)

    def test_final_orbit_matches_target_plane_and_size(self, case):
        c, t = case["tl"].coes()
        assert c.i == pytest.approx(t.i, abs=1e-5)
        assert c.raan == pytest.approx(t.raan, abs=1e-5)
        assert c.a == pytest.approx(t.a, abs=10.0)  # m

    def test_burn_sequence(self, case):
        names = [b.name for b in case["tl"].burns]
        n_phasing = sum(n.startswith("Burn 4.") for n in names)

        # Nominal is already coplanar and on the target radius: no Burn 1 or 2
        has_shaping = case["name"] == "dispersed"
        assert ("Burn 1 (plane)" in names) == has_shaping
        assert ("Burn 2 (Hohmann)" in names) == has_shaping

        assert n_phasing == CFG.k_phase - case["info"]["phases_remaining"]
        assert names[-1] == "HCW stop"
        assert 1 <= case["info"]["hcw_passes"] <= CFG.hcw_iter

    def test_burns_are_chronological(self, case):
        t_burns = [b.t_s for b in case["tl"].burns]
        assert t_burns == sorted(t_burns)

    def test_dv_breakdown_sums_to_total(self, case):
        tl = case["tl"]
        rows = dv_breakdown(tl)
        assert sum(r["dv"] for r in rows) == pytest.approx(tl.dv_total, rel=1e-12)
        assert sum(r["n_burns"] for r in rows) == len(tl.burns)

    def test_j2_bias_never_costs_more(self, case):
        assert case["tl"].dv_total <= case["tl_unbiased"].dv_total

    def test_j2_drift_matches_analytic_estimate(self, case):
        """Phasing orbit sits higher, so its node regresses more slowly than the target's.

        Loose check: the analytic rate assumes one fixed phasing orbit, but closed-loop
        phasing re-plans every rev (measured is ~1.2-1.3x analytic).
        """
        info = case["info_unbiased"]
        drift_analytic = (
            raan_dot_j2(info["a_phase"], 0, info["i_target"])
            - raan_dot_j2(info["a_target"], 0, info["i_target"])
        ) * info["t_phase"]
        assert np.sign(info["draan"]) == np.sign(drift_analytic)
        assert 0.75 < info["draan"] / drift_analytic < 1.5

    def test_dv_total_regression(self, case):
        """Pinned to current behavior; update DV_TOTAL_PIN only for intended changes."""
        assert case["tl"].dv_total == pytest.approx(
            DV_TOTAL_PIN[case["name"]], rel=1e-4
        )

    def test_handoff_inside_hcw_thresh(self, case, request: pytest.MarkGenerator):
        if case["name"] == "nominal":
            # strict: once this is fixed the XPASS fails, reminding you to drop the marker
            request.applymarker(
                pytest.mark.xfail(
                    strict=True,
                    reason="Open issue: nominal phasing uses all revs and hands off "
                    "~37 km out, outside hcw_thresh (see mars-courier README)",
                )
            )
        assert norm(case["info"]["handoff_rsw"][0:3]) < CFG.hcw_thresh
