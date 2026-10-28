"""delta: full permutation is the primary mechanism; mixing is a separately labelled sensitivity."""

from dataclasses import replace

import numpy as np
import pytest

from gdb.synth import schedules as sch
from gdb.synth.dsbm import block_pair_counts, generate_stream, make_block_matrix
from gdb.synth.injectors import DriftEvent, inject_delta
from gdb.utils import graph_stats as gs


def test_default_is_permute_and_requires_full_magnitude(tiny_cfg):
    res = generate_stream(tiny_cfg, sch.abrupt("delta", 10.0, 1.0))
    assert res.resolved[0]["mode"] == "permute"
    for bad in (sch.abrupt("delta", 10.0, 0.2), sch.gradual("delta", 10.0, 1.0, ramp_width=5.0)):
        with pytest.raises(ValueError, match="full permutation"):
            generate_stream(tiny_cfg, bad)
    with pytest.raises(ValueError, match="mode"):
        generate_stream(tiny_cfg, sch.abrupt("delta", 10.0, 1.0, mode="blend"))


def test_mix_mode_is_labelled_and_allows_magnitude(tiny_cfg):
    res = generate_stream(tiny_cfg, sch.abrupt("delta", 10.0, 0.2, mode="mix"))
    assert res.resolved[0]["mode"] == "mix"
    reg = type("R", (), {})  # inject_delta only needs the dataclass fields; use a real Regime
    from gdb.synth.injectors import Regime

    reg = Regime(np.array([[1.0, 0.0], [0.0, 1.0]]), 0.5, np.zeros(2))
    base = {"perm": np.array([1, 0]), "unipartite": False, "flip_homophily": False}
    mix = inject_delta(
        reg, DriftEvent(0, "delta", 0.2, {"mode": "mix"}), 1.0, {**base, "mode": "mix"}
    )
    full = inject_delta(reg, DriftEvent(0, "delta", 1.0), 1.0, {**base, "mode": "permute"})
    assert np.allclose(mix.B, 0.8 * reg.B + 0.2 * reg.B[:, [1, 0]])
    assert np.array_equal(full.B, reg.B[:, [1, 0]])


def test_permutation_changes_assignment_and_preserves_invariants(small_cfg):
    """Primary delta: relational assignment changes; margins, spectrum of B, node sets, marks do not."""
    cfg = replace(small_cfg, t_end=800.0)
    res = generate_stream(cfg, sch.abrupt("delta", 400.0, 1.0))
    s, perm = res.stream, np.array(res.resolved[0]["perm"])
    assert (perm != np.arange(len(perm))).any()  # the assignment really changes
    B = make_block_matrix(cfg.block_matrix, cfg.k_src, cfg.k_dst)
    assert not np.allclose(B[:, perm], B)
    assert np.allclose(
        np.linalg.svd(B, compute_uv=False), np.linalg.svd(B[:, perm], compute_uv=False)
    )
    assert np.allclose(B.sum(0), B[:, perm].sum(0)) and np.allclose(B.sum(1), B[:, perm].sum(1))
    pre, post = s.t < 400, s.t >= 400
    cnt = lambda m: block_pair_counts(s.src[m], s.dst[m], res.node_block, 4, 4).astype(float)  # noqa: E731
    tol = 6 * np.sqrt(cnt(pre).max())
    assert np.abs(cnt(post) - cnt(pre)[:, perm]).max() < tol
    # same node sets and marks law; same degree laws up to sampling noise
    assert set(np.unique(s.src[post])) <= set(np.unique(s.src))
    assert np.linalg.norm(s.edge_feat[post].mean(0) - s.edge_feat[pre].mean(0)) < 0.3
    for nodes in (s.src, s.dst):
        a, b = nodes[pre], nodes[post]
        assert gs.degree_ks(gs.node_degrees(a), gs.node_degrees(b)) < 0.25


def test_permute_with_heavy_tails_leaves_destination_popularity_unchanged(small_cfg):
    cfg = replace(
        small_cfg, t_end=1200.0, base_rate=200.0, pop_dst={"kind": "pareto", "alpha": 1.2}
    )
    res = generate_stream(cfg, sch.abrupt("delta", 600.0, 1.0))
    s = res.stream
    pre, post = s.t < 600, s.t >= 600
    d_pre = np.bincount(s.dst[pre] - cfg.n_src, minlength=cfg.n_dst)
    d_post = np.bincount(s.dst[post] - cfg.n_src, minlength=cfg.n_dst)
    # per-node popularity (not just its distribution) is preserved because swapped blocks have
    # equal size and equal column sum: node-level rank correlation stays high
    from scipy.stats import spearmanr

    assert spearmanr(d_pre, d_post).statistic > 0.8


def test_recurring_permute_restores_law(small_cfg):
    cfg = replace(small_cfg, t_end=600.0)
    res = generate_stream(cfg, sch.recurring("delta", 200.0, 400.0, 1.0))
    s = res.stream
    cnt = lambda a, b: block_pair_counts(  # noqa: E731
        s.src[(s.t >= a) & (s.t < b)], s.dst[(s.t >= a) & (s.t < b)], res.node_block, 4, 4
    ).astype(float)
    a1, b1, a2 = cnt(0, 200), cnt(200, 400), cnt(400, 600)
    tol = 6 * np.sqrt(a1.max())
    assert np.abs(a2 - a1).max() < tol and np.abs(b1 - a1).max() > tol
