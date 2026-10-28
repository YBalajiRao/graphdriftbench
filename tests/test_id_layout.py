"""Reserved id layout: disjoint namespaces for the whole stream, eps-independent tables, no future leakage."""

from dataclasses import replace

import numpy as np
import pytest

from gdb.data.stream import ROLE_DST, ROLE_SRC
from gdb.synth import schedules as sch
from gdb.synth.dsbm import generate_stream


@pytest.fixture
def cfg(tiny_cfg):
    return replace(tiny_cfg, id_layout="reserved", t_end=200.0, base_rate=200.0)


def test_namespaces_stay_disjoint_under_both_sided_churn(cfg):
    cs = cfg.capacity(cfg.n_src)
    res = generate_stream(cfg, sch.abrupt("gamma", 100.0, 0.4, side="both"))
    s = res.stream
    assert s.src.max() < cs <= s.dst.min() and s.dst.max() < cs + cfg.capacity(cfg.n_dst)
    assert (s.src[s.t >= 100] >= cfg.n_src).any() and (s.dst[s.t >= 100] >= cs + cfg.n_dst).any()
    assert (s.node_role[:cs] == ROLE_SRC).all() and (s.node_role[cs:] == ROLE_DST).all()


def test_table_size_and_dst_offset_do_not_depend_on_eps(cfg):
    shapes = set()
    for eps in (0.0, 0.1, 0.4):
        sched = sch.abrupt("gamma", 100.0, eps, side="both") if eps else None
        s = generate_stream(cfg, sched).stream
        shapes.add(
            (s.node_feat.shape, s.node_ids.shape, int(s.dst.min() >= cfg.capacity(cfg.n_src)))
        )
    assert len(shapes) == 1


def test_old_ids_are_never_renumbered(cfg):
    base = generate_stream(cfg).stream
    churn = generate_stream(cfg, sch.abrupt("gamma", 100.0, 0.3)).stream
    k = int(np.searchsorted(base.t, 100.0))
    assert np.array_equal(base.node_feat[: cfg.n_src], churn.node_feat[: cfg.n_src])
    cs = cfg.capacity(cfg.n_src)
    # before the onset only initial ids occur (the shared simulation RNG makes the prefix differ in
    # draws from a stationary run with other breakpoints, but ids are never remapped)
    assert churn.src[:k].max() < cfg.n_src and churn.dst[:k].min() >= cs
    assert churn.dst[:k].max() < cs + cfg.n_dst
    assert np.array_equal(base.node_feat[cs : cs + cfg.n_dst], churn.node_feat[cs : cs + cfg.n_dst])


def test_observed_table_hides_future_entities(cfg):
    res = generate_stream(cfg, sch.abrupt("gamma", 100.0, 0.3))
    s = res.stream
    k = int(np.searchsorted(s.t, 100.0))
    feat, known = s.observed_node_table(k)
    fresh = np.flatnonzero(~s.node_initial & (res.node_block >= 0))
    assert len(fresh) > 0 and not known[fresh].any() and not feat[fresh].any()
    feat2, known2 = s.observed_node_table(len(s))
    seen = np.unique(np.concatenate([s.src[k:], s.dst[k:]]))
    assert known2[seen].all() and feat2[seen[seen >= cfg.n_src]].any()
    assert known[: cfg.n_src].all()  # initial sources are known from the start


def test_headroom_overflow_is_refused(cfg):
    with pytest.raises(ValueError, match="id_headroom"):
        generate_stream(replace(cfg, id_headroom=0.05), sch.abrupt("gamma", 100.0, 0.3))


def test_reserved_layout_is_deterministic_and_legacy_unchanged(cfg, tiny_cfg):
    sched = sch.abrupt("gamma", 50.0, 0.2)
    assert generate_stream(cfg, sched).stream.hash() == generate_stream(cfg, sched).stream.hash()
    legacy = replace(tiny_cfg, id_layout="legacy")
    assert generate_stream(legacy).stream.node_initial is None


# ---- learner-facing representation (StreamReader), default layout, legacy pin -----------------

from gdb.data.stream import StreamReader  # noqa: E402


def test_default_layout_is_reserved_and_legacy_hash_is_pinned(small_cfg):
    assert small_cfg.id_layout == "reserved"
    legacy = replace(small_cfg, id_layout="legacy")
    sched = sch.abrupt("gamma", 25.0, 0.2) + sch.abrupt("alpha", 25.0, 0.2)
    # the value recorded in docs/progress/step-1.md for seed 0 (legacy streams are unchanged)
    assert generate_stream(legacy, sched).stream.hash().startswith("1e04b330")
    assert (
        generate_stream(small_cfg, sched).stream.hash()
        != generate_stream(legacy, sched).stream.hash()
    )


@pytest.mark.parametrize("side", ["src", "dst", "both"])
def test_reader_initial_table_is_independent_of_schedule_and_eps(cfg, side):
    tables = []
    for sched in (
        None,
        sch.abrupt("gamma", 100.0, 0.1, side=side),
        sch.abrupt("gamma", 100.0, 0.4, side=side),
    ):
        feat, role, init = StreamReader(generate_stream(cfg, sched).stream).initial_table()
        tables.append((feat, role, init))
    for f, r, i in tables[1:]:
        assert f.shape == tables[0][0].shape and np.array_equal(f, tables[0][0])
        assert np.array_equal(r, tables[0][1]) and np.array_equal(i, tables[0][2])
    assert not tables[0][0][~tables[0][2]].any()  # reserved rows carry no features at time 0


@pytest.mark.parametrize("side", ["src", "dst", "both"])
def test_reader_releases_features_only_with_first_event(cfg, side):
    res = generate_stream(cfg, sch.abrupt("gamma", 100.0, 0.3, side=side))
    s = res.stream
    assert not s.node_feat[s.late_ids].any()  # raw table holds no fresh features
    seen_events = np.zeros(len(res.node_block), dtype=bool)
    revealed: dict[int, np.ndarray] = {}
    pos = {int(i): k for k, i in enumerate(s.late_ids)}
    for b in StreamReader(s).batches(500):
        seen_events[b.src] = seen_events[b.dst] = True
        for i, f in zip(b.new_ids, b.new_feat):
            assert i not in revealed and seen_events[i] and not s.node_initial[i]
            assert np.array_equal(f, s.late_feat[pos[int(i)]])
            revealed[int(i)] = f
    appeared = np.unique(np.concatenate([s.src, s.dst]))
    expected = {int(i) for i in appeared if not s.node_initial[i]}
    assert set(revealed) == expected and len(expected) > 0
    assert (np.array(sorted(revealed)) >= cfg.n_src).all()


def test_batch_reveal_never_includes_future_ids(cfg):
    s = generate_stream(cfg, sch.abrupt("gamma", 100.0, 0.3, side="both")).stream
    k = int(np.searchsorted(s.t, 100.0))
    first_late = [b for b in StreamReader(s).batches(200) if len(b.new_ids)]
    assert all(b.t[0] <= b.t[-1] for b in first_late)
    pre = [b for b in StreamReader(s).batches(200) if b.t[-1] < s.t[k]]
    assert all(len(b.new_ids) == 0 for b in pre)  # nothing fresh is revealed before the onset


def test_mixed_and_recurring_drift_with_reserved_layout(cfg):
    mixed = generate_stream(
        cfg,
        sch.mixed(
            ["gamma", "alpha", "beta"],
            100.0,
            0.3,
            {"gamma": {"side": "both"}, "beta": {"target": "hub"}},
        ),
    ).stream
    assert mixed.src.max() < cfg.capacity(cfg.n_src) <= mixed.dst.min()
    rec = generate_stream(cfg, sch.recurring("gamma", 80.0, 140.0, 0.3, side="both")).stream
    late = rec.t >= 140.0
    assert (rec.src[late] < cfg.n_src).all() and (
        rec.dst[late] < cfg.capacity(cfg.n_src) + cfg.n_dst
    ).all()
    assert (rec.src[(rec.t >= 80) & (rec.t < 140)] >= cfg.n_src).any()
