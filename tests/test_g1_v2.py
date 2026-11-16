"""Protocol-v2 candidate statistics: beta strata-TV and the bounded jackknife delta residual."""

import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gdb.detect import delta_residual as dr
from gdb.detect import standardise as sd
from gdb.detect import topology as tp
from gdb.detect import window_stats as ws
from gdb.eval import g1, g1_raw

_FRESH = itertools.count(10_000)
SRC = Path(__file__).resolve().parents[1] / "src" / "gdb"


def _stream_window(rng, n_src=120, n_dst=80, n=1500, hub_boost=1.0, churn=0.0):
    """Heavy-tailed bipartite window; ``hub_boost`` multiplies the 5 heaviest dst, ``churn`` relabels src slots."""
    ps = np.arange(1, n_src + 1, dtype=float) ** -0.8
    pd_ = np.arange(1, n_dst + 1, dtype=float) ** -0.8
    pd_[:5] *= hub_boost
    s = rng.choice(n_src, n, p=ps / ps.sum())
    d = rng.choice(n_dst, n, p=pd_ / pd_.sum()) + 1000
    if churn > 0:
        slots = rng.permutation(n_src)[: int(churn * n_src)]
        mapping = {int(x): next(_FRESH) for x in slots}
        s = np.array([mapping.get(int(x), int(x)) for x in s])
    return ws.Window(s, d, rng.normal(size=(n, 4)).astype(np.float32))


def _tv_series(ref):
    return np.array(
        [
            max(tp.strata_tv(ref[j], [ref[i] for i in range(len(ref)) if i != j]))
            for j in range(len(ref))
        ]
    )


def test_strata_are_equal_mass_with_heavy_nodes_first():
    rng = np.random.default_rng(0)
    pool = [_stream_window(rng) for _ in range(5)]
    nodes, strata = tp.pool_strata(pool, "dst", 4)
    cnt = {n: sum((w.dst == n).sum() for w in pool) for n in nodes}
    mass = np.array([sum(cnt[n] for n, s in zip(nodes, strata) if s == k) for k in range(4)])
    assert (mass / mass.sum()).min() > 0.12  # no empty / tiny stratum
    top = nodes[np.argmax([cnt[n] for n in nodes])]
    assert strata[list(nodes).index(top)] == 0


def test_beta_strata_stable_under_pure_turnover_and_sees_hub_drift():
    rng = np.random.default_rng(1)
    ref = [_stream_window(rng) for _ in range(10)]
    null = _tv_series(ref)
    mu, sd_ = null.mean(), null.std(ddof=1)
    pool = ref[1:]
    churned = [max(tp.strata_tv(_stream_window(rng, churn=0.2), pool)) for _ in range(8)]
    hub = [max(tp.strata_tv(_stream_window(rng, hub_boost=6.0), pool)) for _ in range(8)]
    assert np.mean(churned) < mu + 2.0 * sd_  # turnover is not read as topology drift
    assert np.min(hub) > mu + 5.0 * sd_  # genuine hub intervention is


def test_v1_dos_statistic_leaks_turnover_ablation():
    """Documents the v1 failure on a toy law: the persistent-subgraph DoS moves under pure turnover."""
    rng = np.random.default_rng(2)
    ref = [_stream_window(rng, n=1200) for _ in range(6)]
    kw = {"n_moments": 40, "n_probes": 60}

    def dos(w, pool, seed):
        return ws.window_dos_on(w, ws.persistent_nodes(w, pool), seed=seed, **kw)

    spectra = [dos(ref[i], [ref[k] for k in range(6) if k != i], i) for i in range(6)]
    from gdb.utils import graph_stats as gs

    null = [
        gs.dos_w1(spectra[i], ws.mean_spectrum([spectra[k] for k in range(6) if k != i]))
        for i in range(6)
    ]
    pool = ref[:5]
    mean = ws.mean_spectrum(spectra[:5])
    ch = [
        gs.dos_w1(dos(_stream_window(rng, n=1200, churn=0.3), pool, 50 + t), mean) for t in range(4)
    ]
    assert np.mean(ch) > np.mean(null) + 2 * np.std(null, ddof=1)


def test_delta_v1_extrapolates_but_bounded_residual_does_not():
    rng = np.random.default_rng(3)
    f_ref = rng.normal(size=(10, 3))
    l_ref = 1.2 + 0.02 * rng.normal(size=10)
    f_test = np.array([[0.0, 0.0, 105.0]])  # z_gamma far outside the reference range
    l_test = np.array([1.2])  # loss unchanged
    v1 = sd.residual(sd.huber_fit(f_ref, l_ref), f_test, l_test)[0]
    rr, rt = dr.jackknife_residuals(f_ref, l_ref, f_test, l_test)
    sig = rr.std(ddof=1)
    assert abs(v1) > 5 * sig  # v1 residual is dominated by the extrapolated slope
    assert abs(rt[0]) < 4 * sig  # bounded residual stays inside the calibration scale


def test_bounded_residual_responds_to_loss_shift_without_nuisance_movement():
    rng = np.random.default_rng(4)
    f_ref = rng.normal(size=(10, 3))
    l_ref = 1.2 + 0.02 * rng.normal(size=10)
    rr, rt = dr.jackknife_residuals(f_ref, l_ref, np.zeros((1, 3)), np.array([1.5]))
    assert rt[0] > 8 * rr.std(ddof=1)


def test_reference_and_test_residuals_have_comparable_scale_under_the_null():
    ratios = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        f = rng.normal(size=(20, 3))
        loss = 1.0 + 0.05 * rng.normal(size=20)
        rr, rt = dr.jackknife_residuals(f[:10], loss[:10], f[10:], loss[10:])
        ratios.append(np.sqrt(np.mean(rt**2)) / np.sqrt(np.mean(rr**2)))
    assert 0.7 < np.median(ratios) < 1.3


def test_attribution_gate():
    z = np.array([5.0, 5.0, -2.0])
    nuis = np.array([1.0, 50.0, 50.0])
    assert np.array_equal(dr.attribution_gate(z, nuis, 3.0), [5.0, 0.0, -2.0])


def test_none_nuisance_is_median_baseline():
    rr, rt = dr.jackknife_residuals(
        np.zeros((10, 3)), np.arange(10.0), np.zeros((1, 3)), np.array([4.5]), nuisance="none"
    )
    assert rt[0] == pytest.approx(4.5 - 4.5) and rr.shape == (10,)


def test_calibrate_v1_dos_reproduces_locked_compute_case():
    rng = np.random.default_rng(5)
    proto = replace(g1.PROTO, n_boot=40, n_moments=30, n_probes=20)
    wins = [_stream_window(rng, n=600) for _ in range(proto.r + proto.n_post)]
    losses = 1.2 + 0.02 * rng.normal(size=len(wins))
    old = g1.compute_case(wins, losses, seed=7, proto=proto)
    raw = g1_raw.raw_case(wins, losses, seed=7, proto=proto)
    new = g1_raw.calibrate(raw, seed=7, proto=proto)
    for m in range(proto.n_post):
        for t in g1.TYPES:
            assert new[f"z_{t}_w{m}"] == pytest.approx(old[f"z_{t}_w{m}"], rel=1e-6, abs=1e-8)


def test_v2_detector_code_reads_no_labels_or_schedules():
    """No identifier / attribute / argument in the v2 code refers to labels, onsets or generator internals."""
    import ast

    forbidden = (
        "drift_truth",
        "i_tau",
        "inject",
        "schedule",
        "late_feat",
        "late_ids",
        "node_initial",
        "tau",
    )
    for f in (
        SRC / "detect" / "topology.py",
        SRC / "detect" / "delta_residual.py",
        SRC / "eval" / "g1_raw.py",
    ):
        names = set()
        for n in ast.walk(ast.parse(f.read_text())):
            if isinstance(n, ast.Name):
                names.add(n.id)
            elif isinstance(n, ast.Attribute):
                names.add(n.attr)
            elif isinstance(n, ast.arg):
                names.add(n.arg)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) < 40:
                names.add(n.value)
        bad = [x for x in names if any(tok in x for tok in forbidden)]
        assert not bad, (f.name, bad)


def test_node_share_l2_stable_under_turnover_and_sees_hub_drift():
    rng = np.random.default_rng(11)
    ref = [_stream_window(rng) for _ in range(10)]
    null = np.array(
        [max(tp.node_share_l2(ref[j], [ref[i] for i in range(10) if i != j])) for j in range(10)]
    )
    mu, sd_ = null.mean(), null.std(ddof=1)
    pool = ref[1:]
    churned = [max(tp.node_share_l2(_stream_window(rng, churn=0.2), pool)) for _ in range(8)]
    hub = [max(tp.node_share_l2(_stream_window(rng, hub_boost=6.0), pool)) for _ in range(8)]
    assert np.mean(churned) < mu + 2.0 * sd_
    assert np.min(hub) > mu + 5.0 * sd_


def test_calibrate_reports_nonfinite_as_failed_stream():
    raw = {
        "ref": {k: [1.0] * 10 for k in g1_raw.SERIES},
        "test": {k: [1.0] * 3 for k in g1_raw.SERIES},
        "losses": [1.0] * 13,
        "n_act": 100.0,
        "n_pool": 100.0,
    }
    raw["test"]["l2_dst"][0] = float("nan")  # e.g. fewer than 2 persistent events
    assert g1_raw.calibrate(raw, seed=0, beta="l2", delta="bounded")["status"].startswith(
        "nonfinite"
    )


def test_cluster_bootstrap_is_wider_than_independent_when_seeds_are_paired():
    import pandas as pd

    rng = np.random.default_rng(0)
    rows = []
    for s in range(20):
        shared = rng.normal(0, 3)  # a seed-level effect shared by all four types
        for c, k in zip(g1.TYPES, range(4)):
            z = rng.normal(0, 1, 4) + shared * 0.0
            z[k] += 4 + shared
            rows.append(
                {
                    "drift": c,
                    "seed": s,
                    "status": "ok",
                    **{f"z_{t}_w{m}": z[i] for m in range(3) for i, t in enumerate(g1.TYPES)},
                }
            )
    df = pd.DataFrame(rows)
    cl = g1_raw.cluster_bootstrap(df, 0, n_boot=300)
    ind = g1.score_g1(df, 0)["macro_f1_argmax_ci95"]
    assert cl["n_seed_clusters"] == 20
    assert (cl["macro_f1_ci95"][1] - cl["macro_f1_ci95"][0]) >= 0.8 * (ind[1] - ind[0])
