import json

import numpy as np
import pytest

from gdb.synth import injectors as inj
from gdb.synth import schedules as sch
from gdb.synth.dsbm import generate_stream
from gdb.synth.injectors import DriftEvent, DriftSchedule, Regime


def _reg():
    return Regime(B=np.array([[2.0, 0.5], [0.5, 2.0]]), beta_h=0.5, mark_mean=np.zeros(3))


def test_event_validation_and_ramp():
    with pytest.raises(ValueError):
        DriftEvent(1.0, "omega", 0.1)
    with pytest.raises(ValueError):
        DriftEvent(1.0, "alpha", 0.1, onset="gradual")  # needs ramp_width
    with pytest.raises(ValueError):
        DriftEvent(5.0, "alpha", 0.1, end=4.0)
    e = DriftEvent(10.0, "alpha", 0.2, onset="gradual", ramp_width=20.0, end=50.0)
    assert [e.ramp(t) for t in (9.9, 10.0, 20.0, 30.0, 49.9, 50.0)] == [
        0.0,
        0.0,
        0.5,
        1.0,
        1.0,
        0.0,
    ]
    assert DriftEvent(3.0, "beta", 0.1).ramp(3.0) == 1.0


def test_each_regime_injector_changes_only_its_own_factor():
    reg = _reg()
    a = inj.inject_alpha(reg, DriftEvent(0, "alpha", 0.2), 1.0, {"shift": np.ones(3)})
    assert np.array_equal(a.B, reg.B) and a.beta_h == reg.beta_h and a.churn == ()
    assert np.allclose(a.mark_mean, 0.2)

    b = inj.inject_beta(reg, DriftEvent(0, "beta", 0.2), 1.0, {"B_target": np.ones((2, 2))})
    assert np.array_equal(b.mark_mean, reg.mark_mean) and b.beta_h == reg.beta_h and b.churn == ()
    assert np.allclose(b.B, 0.8 * reg.B + 0.2)

    g = inj.inject_gamma(
        reg, DriftEvent(0, "gamma", 0.25), 1.0, {"sides": ["src"], "n_side": {"src": 40}}, 0
    )
    assert (
        np.array_equal(g.B, reg.B)
        and g.beta_h == reg.beta_h
        and np.array_equal(g.mark_mean, reg.mark_mean)
    )
    assert g.churn == ((0, "src", 10),)

    res = {"perm": np.array([1, 0]), "unipartite": False, "flip_homophily": False}
    d = inj.inject_delta(reg, DriftEvent(0, "delta", 1.0), 1.0, res)
    assert np.array_equal(d.mark_mean, reg.mark_mean) and d.beta_h == reg.beta_h and d.churn == ()
    assert np.array_equal(d.B, reg.B[:, [1, 0]])
    flip = inj.inject_delta(reg, DriftEvent(0, "delta", 1.0), 1.0, {**res, "flip_homophily": True})
    assert flip.beta_h == pytest.approx(-0.5)


def test_delta_full_permutation_keeps_entry_multiset_and_margins():
    B = np.array(
        [
            [1.0, 0.15, 0.15, 0.15],
            [0.15, 1.0, 0.15, 0.15],
            [0.15, 0.15, 1.0, 0.15],
            [0.15, 0.15, 0.15, 1.0],
        ]
    )
    n = np.array([10.0, 10.0, 10.0, 10.0])
    perm = inj.delta_permutation(B, n, n, False, np.random.default_rng(0), None)
    assert not np.array_equal(perm, np.arange(4))
    Bp = B[:, perm]
    assert not np.array_equal(Bp, B)
    assert sorted(Bp.ravel()) == sorted(B.ravel())  # multiset of entries unchanged
    assert np.allclose(Bp @ n, B @ n) and np.allclose(Bp.T @ n, B.T @ n)  # row and column margins


def test_delta_refuses_unequal_blocks_and_bad_perm():
    B = np.array([[1.0, 0.1], [0.1, 3.0]])
    with pytest.raises(ValueError):  # margins differ -> no pure delta exists
        inj.delta_permutation(
            B, np.array([5.0, 5.0]), np.array([5.0, 5.0]), False, np.random.default_rng(0), None
        )
    B2 = np.array([[1.0, 0.1, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 1.0]])
    n = np.array([4.0, 4.0, 7.0])
    with pytest.raises(ValueError):  # explicit perm exchanges a size-7 block with a size-4 block
        inj.delta_permutation(B2, n, n, False, np.random.default_rng(0), [2, 1, 0])


def test_beta_targets_preserve_mass():
    B = np.array([[1.0, 0.15, 0.15], [0.15, 1.0, 0.15], [0.15, 0.15, 1.0]])
    n = np.array([10.0, 10.0, 10.0])
    for kind in ("merge", "densify", "hub", "random"):
        t = inj.beta_target(B, n, n, kind, np.random.default_rng(0), True)
        assert inj.mass(t, n, n) == pytest.approx(inj.mass(B, n, n))
        assert not np.allclose(t, B)
    t = inj.beta_target(B, n, n, "hub", np.random.default_rng(0), False, boost=2.0)
    assert t[0, 0] == pytest.approx(4.0) and t[0, 1] == pytest.approx(0.3)


def test_schedule_csv_roundtrip_matches_configured_schedule(tmp_path, tiny_cfg):
    schedule = (
        sch.abrupt("alpha", 10.0, 0.2)
        + sch.gradual("beta", 20.0, 0.4, ramp_width=15.0, ramp_steps=5, target="hub", boost=10.0)
        + sch.mixed(
            ["gamma", "delta"], 30.0, {"gamma": 0.1, "delta": 0.3}, {"gamma": {"side": "both"}}
        )
        + sch.recurring("delta", 40.0, 60.0, 0.5, flip_homophily=True)
    )
    res = generate_stream(tiny_cfg, schedule)
    path = tmp_path / "drift_truth.csv"
    schedule.to_csv(path, res.resolved)
    back = DriftSchedule.from_csv(path)
    assert back.events == schedule.events  # exact: tau, type, eps, params, onset, ramp, end
    rows = path.read_text().splitlines()
    assert len(rows) == 1 + len(schedule)
    assert "perm" in json.loads(json.dumps(res.resolved[3]))  # event 3 is the delta at tau=30


def test_builders_produce_expected_shapes():
    assert len(sch.mixed(["alpha", "beta", "gamma"], 5.0, 0.2)) == 3
    r = sch.recurring("gamma", 10.0, 30.0, 0.2)
    assert r.events[0].end == 30.0
    g = sch.gradual("alpha", 10.0, 0.2, 40.0)
    assert g.events[0].onset == "gradual" and g.events[0].ramp(30.0) == 0.5
