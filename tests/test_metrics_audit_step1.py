"""Independent audit of Step 1 statistics (Blueprint 5.1): graph_stats and purity pass rules."""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from gdb.synth import purity
from gdb.utils import graph_stats as gs


def _graph(seed=0, n=60, p=0.12):
    g = nx.gnp_random_graph(n, p, seed=seed)
    g.remove_nodes_from([v for v in list(g) if g.degree(v) == 0])
    g = nx.convert_node_labels_to_integers(g)
    return g, sp.csr_matrix(nx.to_numpy_array(g))


def test_jackson_closed_form():
    # g_m = ((N-m+1) cos(pi m/(N+1)) + sin(pi m/(N+1)) cot(pi/(N+1))) / (N+1), N=5 by hand loop
    N = 5
    ref = np.array(
        [
            (
                (N - m + 1) * np.cos(np.pi * m / (N + 1))
                + np.sin(np.pi * m / (N + 1)) / np.tan(np.pi / (N + 1))
            )
            / (N + 1)
            for m in range(N)
        ]
    )
    np.testing.assert_allclose(gs._jackson(N), ref, atol=1e-12)
    assert gs._jackson(N)[0] == pytest.approx(1.0)  # g_0 = 1 (normalisation)
    g = gs._jackson(100)
    assert np.all(g > -1e-12) and np.all(np.diff(g) <= 1e-12)  # positive, decreasing


def test_kpm_moments_match_exact_chebyshev():
    _, a = _graph()
    n = a.shape[0]
    lam = np.linalg.eigvalsh(gs.normalized_adjacency(a).toarray())
    M = 12
    exact = np.array([np.mean(np.cos(m * np.arccos(np.clip(lam, -1, 1)))) for m in range(M)])
    # recover KPM moments by re-running the recursion with many probes
    ah = gs.normalized_adjacency(a)
    rng = np.random.default_rng(1)
    P = 4000
    z = rng.integers(0, 2, size=(n, P)) * 2.0 - 1.0
    v0, v1 = z, ah @ z
    mu = [np.sum(z * v0) / (n * P), np.sum(z * v1) / (n * P)]
    for _ in range(2, M):
        v0, v1 = v1, 2 * (ah @ v1) - v0
        mu.append(np.sum(z * v1) / (n * P))
    np.testing.assert_allclose(mu, exact, atol=0.03)


def test_kpm_dos_matches_exact_spectrum_in_w1():
    _, a = _graph(n=80, p=0.1)
    ex = gs.exact_spectrum(a)
    est = gs.kpm_dos(a, n_moments=100, n_probes=300, rng=np.random.default_rng(0))
    assert est.weights.sum() == pytest.approx(1.0)
    assert np.all(est.weights >= 0)
    assert est.values.min() >= -1e-9 and est.values.max() <= 2 + 1e-9
    assert gs.dos_w1(est, ex) < 0.05
    # mean eigenvalue of normalised Laplacian = 1 - tr(Ahat)/n = 1 (no self loops)
    assert np.sum(est.values * est.weights) == pytest.approx(1.0, abs=0.02)


def test_w1_hand_example():
    s1 = gs.Spectrum(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
    s2 = gs.Spectrum(np.array([0.5, 1.5]), np.array([0.5, 0.5]))
    assert gs.dos_w1(s1, s2) == pytest.approx(0.5)
    # integral of |F1-F2|: F1-F2 = 0.5 on [0,.5), 0 on [.5,1), 0.5 on [1,1.5) => 0.5
    s3 = gs.Spectrum(np.array([0.0]), np.array([1.0]))
    s4 = gs.Spectrum(np.array([2.0]), np.array([1.0]))
    assert gs.dos_w1(s3, s4) == pytest.approx(2.0)


def test_linear_mmd_brute_force():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(7, 2)), rng.normal(size=(9, 2)) + 0.5
    sig = 1.3
    k = lambda a, b: np.exp(-np.sum((a - b) ** 2) / (2 * sig**2))
    m = min(len(x), len(y)) // 2  # 3
    h = [
        k(x[2 * i], x[2 * i + 1])
        + k(y[2 * i], y[2 * i + 1])
        - k(x[2 * i], y[2 * i + 1])
        - k(x[2 * i + 1], y[2 * i])
        for i in range(m)
    ]
    assert gs.linear_mmd2_rbf(x, y, bandwidth=sig) == pytest.approx(np.mean(h), abs=1e-12)
    assert np.isnan(gs.linear_mmd2_rbf(x[:1], y))


def test_mmd_unbiased_zero_under_null_and_positive_under_shift():
    rng = np.random.default_rng(4)
    null = np.mean(
        [
            gs.linear_mmd2_rbf(rng.normal(size=(400, 2)), rng.normal(size=(400, 2)), bandwidth=1.0)
            for _ in range(200)
        ]
    )
    shift = np.mean(
        [
            gs.linear_mmd2_rbf(
                rng.normal(size=(400, 2)), rng.normal(size=(400, 2)) + 1, bandwidth=1.0
            )
            for _ in range(50)
        ]
    )
    assert abs(null) < 0.01
    assert shift > 0.05


def test_median_bandwidth_hand():
    z = np.array([[0.0], [1.0], [3.0]])  # distances 1,3,2 -> median 2
    assert gs.median_bandwidth(z) == pytest.approx(2.0)


def test_stratified_mmd_weights():
    rng = np.random.default_rng(5)
    nb0, nb1 = 60, 40  # reference shares 0.6 / 0.4 ; bucket 2 has too few events -> dropped
    x = rng.normal(size=(nb0 + nb1 + 5, 2))
    bx = np.r_[np.zeros(nb0), np.ones(nb1), 2 * np.ones(5)].astype(int)
    y = rng.normal(size=(150, 2)) + 0.3
    by = np.r_[np.zeros(70), np.ones(70), 2 * np.ones(10)].astype(int)
    sig = 1.0
    m0 = gs.linear_mmd2_rbf(x[bx == 0], y[by == 0], bandwidth=sig)
    m1 = gs.linear_mmd2_rbf(x[bx == 1], y[by == 1], bandwidth=sig)
    pi0, pi1 = nb0 / 105, nb1 / 105  # share of full x; renormalised over used buckets
    expected = (pi0 * m0 + pi1 * m1) / (pi0 + pi1)
    got = gs.stratified_linear_mmd2(x, y, bx, by, bandwidth=sig, min_count=20)
    assert got == pytest.approx(expected, abs=1e-12)
    assert np.isnan(gs.stratified_linear_mmd2(x, y, bx, by, bandwidth=sig, min_count=10_000))


def test_degree_bins_and_pair_buckets():
    np.testing.assert_array_equal(
        gs.log_degree_bins(np.array([1, 2, 3, 4, 7, 8, 100])), [0, 1, 1, 2, 2, 2, 2]
    )
    src = np.array([0, 0, 0, 0, 1])  # deg(0)=4 -> bin 2 ; deg(1)=1 -> bin 0
    dst = np.array([5, 5, 6, 7, 8])  # deg(5)=2 -> bin1; others 1 -> bin 0
    np.testing.assert_array_equal(gs.degree_pair_buckets(src, dst), [2 * 3 + 1, 2 * 3 + 1, 6, 6, 0])


def test_jaccard_new_node_rate():
    a, b = np.array([1, 2, 3, 3, 4]), np.array([3, 4, 5, 6])
    assert gs.jaccard(a, b) == pytest.approx(2 / 6)
    assert gs.new_node_rate(b, a) == pytest.approx(2 / 4)
    assert gs.new_node_rate(a, b) == pytest.approx(2 / 4)
    assert gs.new_node_rate(np.array([1, 2, 3]), np.array([1])) == pytest.approx(2 / 3)
    assert gs.jaccard(np.array([]), np.array([])) == 1.0


def test_degree_ks_hand():
    # F_a: 1->.5, 2->1 ; F_b: 2->.5, 3->1 ; sup diff = 0.5
    assert gs.degree_ks(np.array([1, 2]), np.array([2, 3])) == pytest.approx(0.5)
    assert gs.degree_ks(np.array([1, 2]), np.array([1, 2])) == 0.0


def test_core_numbers():
    # triangle 0-1-2 with a pendant 3 on node 2, self-loop ignored
    src, dst = np.array([0, 1, 2, 2, 3]), np.array([1, 2, 0, 3, 3])
    c = gs.core_numbers(src, dst)
    assert c == {0: 2, 1: 2, 2: 2, 3: 1}


def test_blockpair_chi2_hand_2x2():
    cb = np.array([[10, 20], [30, 40]])
    ca = np.array([[20, 20], [10, 50]])
    expected = 100 / 30 + 0 + 400 / 40 + 100 / 90  # 14.4444

    # layout: sources 0/1 = blocks 0/1, dests 2/3 = blocks 0/1
    def events(c):
        s, d = [], []
        for i in range(2):
            for j in range(2):
                s += [i] * c[i, j]
                d += [2 + j] * c[i, j]
        return np.array(s), np.array(d)

    sb, db = events(cb)
    sa, da = events(ca)
    src, dst = np.r_[sb, sa], np.r_[db, da]
    rng = np.random.default_rng(0)
    stream = SimpleNamespace(src=src, dst=dst, edge_feat=rng.normal(size=(len(src), 2)))
    res = SimpleNamespace(
        stream=stream, node_block=np.array([0, 1, 0, 1]), cfg=SimpleNamespace(k_src=2, k_dst=2)
    )
    out = purity.window_pair_stats(res, 100, 100, kpm_seed=0, n_moments=20, n_probes=5)
    assert out["blockpair_chi2"] == pytest.approx(expected, rel=1e-9)
    # null expectation: chi2 ~ df-ish, i.e. identical windows give 0
    res2 = SimpleNamespace(
        stream=SimpleNamespace(
            src=np.r_[sb, sb], dst=np.r_[db, db], edge_feat=rng.normal(size=(200, 2))
        ),
        node_block=res.node_block,
        cfg=res.cfg,
    )
    assert (
        purity.window_pair_stats(res2, 100, 100, kpm_seed=0, n_moments=20, n_probes=5)[
            "blockpair_chi2"
        ]
        == 0.0
    )


def _df(n_out_by_stat, n):
    """df where stat column has n_out values above band=1.0 and rest below."""
    return pd.DataFrame(
        {s: np.r_[np.full(n - k, 0.5), np.full(k, 2.0)] for s, k in n_out_by_stat.items()}
    )


def _band():
    return {"band": {s: 1.0 for s in purity.STATS}}


def test_pass_rates_n20_boundaries():
    # null cell: 17 inside passes, 16 inside fails
    for inside, ok in [(17, True), (16, False), (20, True)]:
        df = _df({s: 20 - inside for s in purity.STATS}, 20)
        r = purity.pass_rates(df, _band(), "none")
        assert bool(r["pass"].all()) is ok, inside
    # shift cell (alpha/mark_mmd): 18 outside passes, 17 fails
    for outside, ok in [(18, True), (17, False)]:
        k = {s: 0 for s in purity.STATS}
        k["mark_mmd"] = outside
        r = purity.pass_rates(_df(k, 20), _band(), purity.ALPHA).set_index("statistic")
        assert bool(r.loc["mark_mmd", "pass"]) is ok
        assert r.loc["mark_mmd", "n_counted"] == outside
    # info cell: pass is None
    r = purity.pass_rates(_df({s: 3 for s in purity.STATS}, 20), _band(), purity.BETA).set_index(
        "statistic"
    )
    assert r.loc["deg_ks_src", "pass"] is None
    # value equal to band counts as inside
    df = pd.DataFrame({s: [1.0] * 20 for s in purity.STATS})
    assert purity.pass_rates(df, _band(), "none")["n_inside_band"].eq(20).all()


@pytest.mark.parametrize("n", [5, 10, 40])
def test_pass_rates_scaling_is_proportional(n):
    """Threshold should be at least 17/20 (null) and 18/20 (shift) of n, i.e. ceil, not floor."""
    import math

    null_thr = math.ceil(17 * n / 20)
    shift_thr = math.ceil(18 * n / 20)
    # exactly one below the proportional ceil threshold must fail
    df = _df({s: n - (null_thr - 1) for s in purity.STATS}, n)
    assert not purity.pass_rates(df, _band(), "none")["pass"].any(), "null cell passes below 17/20"
    k = {s: 0 for s in purity.STATS}
    k["mark_mmd"] = shift_thr - 1
    r = purity.pass_rates(_df(k, n), _band(), purity.ALPHA).set_index("statistic")
    assert not r.loc["mark_mmd", "pass"], "shift cell passes below 18/20"
