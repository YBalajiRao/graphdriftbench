from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import chi2

from gdb.data.stream import ROLE_DST, ROLE_SRC
from gdb.synth import schedules as sch
from gdb.synth.dsbm import DSBMConfig, block_pair_counts, expected_block_counts, generate_stream


@pytest.mark.parametrize("seed", range(5))
def test_stationary_block_pair_counts_match_poisson_expectation(small_cfg, seed):
    cfg = replace(small_cfg, seed=seed)
    res = generate_stream(cfg)
    obs = block_pair_counts(res.stream.src, res.stream.dst, res.node_block, cfg.k_src, cfg.k_dst)
    exp = expected_block_counts(cfg)
    z = (obs - exp) / np.sqrt(exp)
    assert (np.abs(z) <= 3).mean() >= 0.95  # at least 95% of the 16 cells within 3 sigma
    assert chi2.sf((z**2).sum(), df=z.size) > 1e-3  # and no joint departure


def test_total_event_count_is_poisson_around_base_rate(small_cfg):
    n = len(generate_stream(small_cfg).stream)
    mean = small_cfg.base_rate * small_cfg.t_end
    assert abs(n - mean) < 4 * np.sqrt(mean)


def test_same_seed_identical_stream_hash_and_different_seed_differs(tiny_cfg):
    sched = sch.abrupt("gamma", 40.0, 0.2) + sch.abrupt("alpha", 60.0, 0.2)
    h = [generate_stream(replace(tiny_cfg, seed=s), sched).stream.hash() for s in (0, 0, 1)]
    assert h[0] == h[1] != h[2]
    assert generate_stream(tiny_cfg).stream.hash() == generate_stream(tiny_cfg).stream.hash()


def test_bipartite_conventions(tiny_cfg):
    res = generate_stream(tiny_cfg)
    s = res.stream
    assert s.src.max() < tiny_cfg.n_src <= s.dst.min()  # sources 0..N_src-1, destinations after
    assert s.dst.max() < tiny_cfg.n_src + tiny_cfg.n_dst
    assert np.all(np.diff(s.t) >= 0) and s.t.min() >= 0 and s.t.max() <= tiny_cfg.t_end
    assert s.edge_feat.shape == (len(s), tiny_cfg.d_edge) and s.edge_feat.dtype == np.float32
    assert (s.node_role[: tiny_cfg.n_src] == ROLE_SRC).all() and (
        s.node_role[tiny_cfg.n_src :] == ROLE_DST
    ).all()
    assert len(s.node_feat) == len(s.node_ids) == len(res.node_block)


def test_unipartite_mode_has_no_self_loops_and_one_id_space():
    paired = {
        "kind": "explicit",
        "matrix": [
            [0.15, 1, 0.15, 0.15],
            [1, 0.15, 0.15, 0.15],
            [0.15, 0.15, 0.15, 1],
            [0.15, 0.15, 1, 0.15],
        ],
    }
    cfg = DSBMConfig(
        mode="unipartite",
        n_src=120,
        k_src=4,
        block_matrix=paired,
        base_rate=100.0,
        t_end=60.0,
        d_edge=4,
    )
    res = generate_stream(cfg, sch.abrupt("delta", 30.0, 1.0) + sch.abrupt("gamma", 30.0, 0.2))
    s = res.stream
    assert (s.src == s.dst).sum() == 0
    assert s.src.min() >= 0 and max(s.src.max(), s.dst.max()) < len(s.node_feat)
    assert cfg.n_dst == cfg.n_src and cfg.k_dst == cfg.k_src


def test_heavy_tailed_destination_popularity():
    cfg = DSBMConfig(
        n_src=2000, n_dst=200, k_src=4, k_dst=4, base_rate=500.0, t_end=40.0, d_edge=4,
        pop_dst={"kind": "pareto", "alpha": 1.2},
    )  # fmt: skip
    deg = np.bincount(generate_stream(cfg).stream.dst - cfg.n_src, minlength=cfg.n_dst)
    assert deg.max() > 10 * np.median(deg[deg > 0])


def test_alpha_shifts_only_the_mark_mean(tiny_cfg):
    cfg = replace(tiny_cfg, t_end=200.0, base_rate=200.0)
    res = generate_stream(cfg, sch.abrupt("alpha", 100.0, 0.5))
    s = res.stream
    before, after = s.edge_feat[s.t < 100], s.edge_feat[s.t >= 100]
    shift = np.linalg.norm(after.mean(0) - before.mean(0))
    assert shift == pytest.approx(0.5 * cfg.edge_sigma * np.sqrt(cfg.d_edge), rel=0.1)


def test_gamma_old_ids_stop_and_fresh_ids_appear(tiny_cfg):
    cfg = replace(tiny_cfg, t_end=200.0, base_rate=200.0)
    res = generate_stream(cfg, sch.abrupt("gamma", 100.0, 0.25))  # default side: sources
    s = res.stream
    n0 = cfg.n_src + cfg.n_dst
    before, after = s.t < 100, s.t >= 100
    fresh = s.src[after][s.src[after] >= n0]
    assert len(np.unique(fresh)) == int(0.25 * cfg.n_src)  # every reserved fresh id is used
    assert (s.dst[after] < n0).all()  # destinations untouched
    gone = np.setdiff1d(np.unique(s.src[before]), np.unique(s.src[after]))
    assert len(gone) == int(0.25 * cfg.n_src)  # exactly the replaced nodes disappear
    assert (res.node_block[fresh] >= 0).all() and len(s.node_feat) == n0 + int(0.25 * cfg.n_src)


def test_gamma_both_sides_and_dst_option(tiny_cfg):
    cfg = replace(tiny_cfg, t_end=200.0, base_rate=200.0)
    n0 = cfg.n_src + cfg.n_dst
    s = generate_stream(cfg, sch.abrupt("gamma", 100.0, 0.2, side="both")).stream
    assert (s.src[s.t >= 100] >= n0).any() and (s.dst[s.t >= 100] >= n0).any()
    s = generate_stream(cfg, sch.abrupt("gamma", 100.0, 0.2, side="dst")).stream
    assert (s.src >= n0).sum() == 0 and (s.dst[s.t >= 100] >= n0).any()


def test_delta_full_permutation_permutes_block_pair_counts(small_cfg):
    cfg = replace(small_cfg, t_end=400.0)
    res = generate_stream(cfg, sch.abrupt("delta", 200.0, 1.0))
    s, perm = res.stream, np.array(res.resolved[0]["perm"])
    cnt = lambda m: block_pair_counts(s.src[m], s.dst[m], res.node_block, 4, 4).astype(float)  # noqa: E731
    c_b, c_a = cnt(s.t < 200), cnt(s.t >= 200)
    assert np.abs(c_a - c_b[:, perm]).max() < 6 * np.sqrt(
        c_b.max()
    )  # after = before with columns permuted
    assert not np.allclose(c_a, c_b, atol=6 * np.sqrt(c_b.max()))
    assert np.allclose(
        sorted(c_a.ravel()), sorted(c_b.ravel()), atol=6 * np.sqrt(c_b.max())
    )  # same multiset


def test_gradual_ramp_increases_effective_churn_monotonically(tiny_cfg):
    cfg = replace(tiny_cfg, t_end=300.0, base_rate=200.0)
    res = generate_stream(cfg, sch.gradual("gamma", 100.0, 0.4, ramp_width=100.0, ramp_steps=10))
    s, n0 = res.stream, cfg.n_src + cfg.n_dst
    frac = [
        (lambda m: (s.src[m] >= n0).mean())((s.t >= a) & (s.t < a + 20))
        for a in (80, 100, 120, 140, 160, 180, 200, 220)
    ]
    assert frac[0] == 0.0 and frac[-1] > 0.3 and all(b >= a - 0.05 for a, b in zip(frac, frac[1:]))


def test_recurring_gamma_brings_original_ids_back(tiny_cfg):
    cfg = replace(tiny_cfg, t_end=300.0, base_rate=200.0)
    res = generate_stream(cfg, sch.recurring("gamma", 100.0, 200.0, 0.3))
    s, n0 = res.stream, cfg.n_src + cfg.n_dst
    assert (s.src[s.t < 100] >= n0).sum() == 0
    assert (s.src[(s.t >= 100) & (s.t < 200)] >= n0).any()
    assert (s.src[s.t >= 200] >= n0).sum() == 0  # original law restored


def test_recurring_delta_restores_block_pair_counts(small_cfg):
    cfg = replace(small_cfg, t_end=600.0)
    res = generate_stream(cfg, sch.recurring("delta", 200.0, 400.0, 1.0))
    s = res.stream
    cnt = lambda a, b: block_pair_counts(  # noqa: E731
        s.src[(s.t >= a) & (s.t < b)], s.dst[(s.t >= a) & (s.t < b)], res.node_block, 4, 4
    ).astype(float)
    a1, b1, a2 = cnt(0, 200), cnt(200, 400), cnt(400, 600)
    tol = 6 * np.sqrt(a1.max())
    assert np.abs(a2 - a1).max() < tol and np.abs(b1 - a1).max() > tol


def test_mixed_drift_applies_all_components(tiny_cfg):
    cfg = replace(tiny_cfg, t_end=200.0, base_rate=200.0)
    res = generate_stream(cfg, sch.mixed(["alpha", "gamma"], 100.0, 0.3))
    s, n0 = res.stream, cfg.n_src + cfg.n_dst
    assert (s.src[s.t >= 100] >= n0).any()
    assert np.linalg.norm(s.edge_feat[s.t >= 100].mean(0) - s.edge_feat[s.t < 100].mean(0)) > 0.3


def test_temporal_data_export(tiny_cfg):
    td = generate_stream(tiny_cfg).stream.to_temporal_data()
    assert td.num_events > 0 and td.msg.shape[1] == tiny_cfg.d_edge
    ti = generate_stream(tiny_cfg).stream.to_temporal_data(time_scale=100.0)
    assert ti.t.dtype.is_floating_point is False


def test_stream_save_load_roundtrip(tmp_path, tiny_cfg):
    from gdb.data.stream import EventStream

    res = generate_stream(tiny_cfg, sch.abrupt("gamma", 50.0, 0.1))
    res.save(tmp_path / "x")
    assert EventStream.load(tmp_path / "x" / "stream.pt").hash() == res.stream.hash()
    assert (tmp_path / "x" / "drift_truth.csv").exists() and (
        tmp_path / "x" / "config.yaml"
    ).exists()


def test_wiki_like_preset_shape(small_cfg):
    from pathlib import Path

    cfg = DSBMConfig.from_yaml(
        Path(__file__).resolve().parents[1] / "configs" / "synth" / "wiki_like.yaml"
    )
    assert (cfg.n_src, cfg.n_dst, cfg.d_edge) == (8227, 1000, 172)
    assert cfg.base_rate * cfg.t_end == pytest.approx(157474, rel=1e-3)
    res = generate_stream(replace(cfg, t_end=20.0))  # ~3K events: shape check only
    s = res.stream
    assert s.edge_feat.shape[1] == 172 and s.src.max() < 8227 <= s.dst.min() and s.dst.max() < 9227


def test_delta_refuses_to_be_a_silent_noop(tiny_cfg):
    """An assortative unipartite B is invariant under block relabelling, so delta would change nothing."""
    cfg = DSBMConfig(mode="unipartite", n_src=120, k_src=4, base_rate=50.0, t_end=20.0, d_edge=4)
    with pytest.raises(ValueError, match="no-op"):
        generate_stream(cfg, sch.abrupt("delta", 10.0, 1.0))
    generate_stream(tiny_cfg, sch.abrupt("delta", 10.0, 1.0))  # bipartite assortative B is fine
