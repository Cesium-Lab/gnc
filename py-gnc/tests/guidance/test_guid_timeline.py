import numpy as np
import pytest

from cesium.constants import J2000_JD
from cesium.guidance import Burn, RpoTimeline
from cesium.orbital.coes import COES
from cesium.physics.dynamics import propagate_two_body
from cesium.world import MU_EARTH, R_EARTH

A = R_EARTH + 500e3
RV_T0 = COES(a=A, e=0.0, i=0.5, raan=0.2, aop=0.0, ta=0.0).to_rv(MU_EARTH)
RV_C0 = COES(a=A, e=0.0, i=0.5, raan=0.2, aop=0.0, ta=-1e-3).to_rv(MU_EARTH)


def make_timeline(**kwargs):
    return RpoTimeline(RV_C0, RV_T0, MU_EARTH, frame="ECI", **kwargs)


def test_coast_matches_direct_propagation_and_chains():
    tl = make_timeline()
    tl.coast(600.0, n=50)
    tl.coast(400.0, n=30)

    ts, rv_c, rv_t = tl.arrays()
    assert ts.shape == (1 + 49 + 29,) and rv_c.shape == rv_t.shape == (ts.size, 6)
    assert np.all(np.diff(ts) > 0) and tl.t == pytest.approx(1000.0)

    rv_t_direct = propagate_two_body(RV_T0, [0.0, 600.0, 1000.0], mu=MU_EARTH)
    assert tl.target == pytest.approx(rv_t_direct[-1], rel=1e-8, abs=1e-3)


def test_burn_changes_only_chaser_velocity_and_is_recorded():
    tl = make_timeline()
    tl.coast(300.0, n=20)
    before = tl.chaser
    dv = np.array([0.0, 1.0, 2.0])

    burn = tl.burn(dv, "test burn")

    assert isinstance(burn, Burn) and tl.burns == [burn]
    assert (burn.t_s, burn.name, burn.frame) == (300.0, "test burn", "ECI")
    assert burn.r == pytest.approx(before[0:3])
    assert tl.chaser == pytest.approx(before + np.r_[0, 0, 0, dv])
    assert tl.arrays()[0].size == 20  # no extra sample for an impulse
    assert tl.dv_total == pytest.approx(np.sqrt(5.0))
    assert tl.burn_events() == [(300.0, "test burn")]


def test_epochs_are_relative_times_plus_start_epoch():
    epoch0 = 8.0e8  # TDB s past J2000
    tl = make_timeline(epoch0_tdb_s=epoch0)
    tl.coast_until(120.0, n=5)

    assert tl.epochs_tdb_s() - epoch0 == pytest.approx(tl.arrays()[0])
    assert tl.epoch_tdb_s == epoch0 + 120.0
    assert tl.epoch_jd_tdb == pytest.approx(
        J2000_JD + (epoch0 + 120.0) / 86400.0, abs=1e-9
    )


def test_rsw_round_trip_and_arrays():
    tl = make_timeline()
    tl.coast(100.0, n=10)

    assert tl.from_rsw(tl.to_rsw()) == pytest.approx(tl.chaser, rel=1e-12, abs=1e-6)
    ts, rv_rsw = tl.rsw_arrays()
    assert rv_rsw.shape == (ts.size, 6)
    assert rv_rsw[-1] == pytest.approx(tl.to_rsw())


def test_burn_table_lists_burns_and_total():
    tl = make_timeline()
    tl.burn([3.0, 4.0, 0.0], "first")
    table = tl.burn_table()
    assert "first" in table and "5.000" in table and "total" in table


def test_invalid_inputs():
    with pytest.raises(ValueError, match="r_cb"):
        make_timeline(j2=1e-3)
    with pytest.raises(ValueError):
        make_timeline().coast(-1.0)


def test_burn_table_has_utc_column():
    tl = make_timeline()
    tl.burn([3.0, 4.0, 0.0], "first")
    header, row = tl.burn_table().splitlines()[:2]
    assert "UTC" in header
    # epoch0 = 0 TDB s past J2000 -> 2000-01-01 12:00:00 TDB = 11:58:50.8 UTC (TT - UTC = 69.184 s), rounded
    assert "2000-01-01 11:58:51" in row
