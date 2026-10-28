import networkx as nx
import numpy as np
import pytest

from gdb.utils import graph_stats as gs


def _bip_sbm(ns: int, nd: int, k: int, p_in: float, p_out: float, seed: int):
    r = np.random.default_rng(seed)
    bs, bd = np.arange(ns) % k, np.arange(nd) % k
    p = np.where(bs[:, None] == bd[None, :], p_in, p_out)
    u, v = np.nonzero(r.random((ns, nd)) < p)
    return u, v + ns


@pytest.mark.parametrize(
    "ns,nd,k,p_in,p_out",
    [(120, 80, 4, 0.2, 0.02), (250, 200, 4, 0.1, 0.01), (300, 200, 5, 0.1, 0.02)],  # n = 200 .. 500
)
def test_kpm_matches_eigvalsh(ns, nd, k, p_in, p_out):
    """KPM DoS vs exact eigenvalues of the normalised Laplacian: W1 error < 0.02 (n <= 500)."""
    u, v = _bip_sbm(ns, nd, k, p_in, p_out, seed=1)
    a, nodes = gs.window_adjacency(u, v)
    assert 100 <= len(nodes) <= 500
    exact = gs.exact_spectrum(a)
    errs = [
        gs.dos_w1(gs.kpm_dos(a, n_moments=100, n_probes=30, rng=np.random.default_rng(s)), exact)
        for s in range(5)
    ]
    assert max(errs) < 0.02, errs


def test_kpm_is_a_probability_measure_in_range():
    u, v = _bip_sbm(100, 60, 3, 0.2, 0.03, seed=2)
    a, _ = gs.window_adjacency(u, v)
    s = gs.kpm_dos(a, rng=np.random.default_rng(0))
    assert s.weights.min() >= 0 and s.weights.sum() == pytest.approx(1.0)
    assert s.values.min() >= -1e-9 and s.values.max() <= 2 + 1e-9


def test_exact_spectrum_matches_numpy_laplacian():
    g = nx.path_graph(6)
    u, v = (np.array(x) for x in zip(*g.edges))
    a, _ = gs.window_adjacency(u, v)
    lam = np.sort(gs.exact_spectrum(a).values)
    ref = np.sort(np.linalg.eigvalsh(nx.normalized_laplacian_matrix(g).toarray()))
    assert np.allclose(lam, ref, atol=1e-10)


def test_dos_w1_zero_for_identical_and_positive_for_different():
    u, v = _bip_sbm(100, 60, 3, 0.2, 0.03, seed=3)
    u2, v2 = _bip_sbm(100, 60, 3, 0.5, 0.2, seed=3)
    s1 = gs.exact_spectrum(gs.window_adjacency(u, v)[0])
    s2 = gs.exact_spectrum(gs.window_adjacency(u2, v2)[0])
    assert gs.dos_w1(s1, s1) == 0.0
    assert gs.dos_w1(s1, s2) > 0.05


def test_window_adjacency_dedups_drops_selfloops_and_restricts():
    src = np.array([0, 0, 1, 2, 3])
    dst = np.array([5, 5, 5, 2, 6])  # duplicate (0,5), self-loop (2,2)
    a, nodes = gs.window_adjacency(src, dst)
    assert list(nodes) == [0, 1, 3, 5, 6]
    assert a.nnz == 6 and a.max() == 1.0  # 3 distinct undirected edges, symmetric
    a2, nodes2 = gs.window_adjacency(src, dst, node_subset=np.array([0, 1, 5]))
    assert list(nodes2) == [0, 1, 5] and a2.nnz == 4


def test_core_numbers_known_graph():
    # K4 on {0,1,2,3} (core 3), triangle {4,5,6} (core 2), pendant path 7-8 attached to 3 (core 1)
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (4, 5), (5, 6), (4, 6), (3, 7), (7, 8)]
    src, dst = (np.array(x) for x in zip(*edges))
    core = gs.core_numbers(src, dst)
    assert [core[i] for i in range(4)] == [3, 3, 3, 3]
    assert [core[i] for i in (4, 5, 6)] == [2, 2, 2]
    assert core[7] == 1 and core[8] == 1


def test_node_degrees_and_ks():
    deg = gs.node_degrees(np.array([1, 1, 1, 2, 3, 3]))
    assert sorted(deg) == [1, 2, 3]
    assert gs.degree_ks(deg, deg) == 0.0
    assert gs.degree_ks(np.array([1, 1, 1]), np.array([5, 5, 5])) == 1.0


def test_median_bandwidth_scales_with_data():
    x = np.random.default_rng(0).normal(size=(500, 4))
    assert gs.median_bandwidth(3.0 * x) == pytest.approx(3.0 * gs.median_bandwidth(x), rel=1e-6)


def test_linear_mmd_null_and_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 16))
    null = [
        gs.linear_mmd2_rbf(rng.normal(size=(4000, 16)), rng.normal(size=(4000, 16)))
        for _ in range(10)
    ]
    shifted = [
        gs.linear_mmd2_rbf(rng.normal(size=(4000, 16)), rng.normal(size=(4000, 16)) + 1.0)
        for _ in range(10)
    ]
    assert abs(np.mean(null)) < 0.01  # unbiased: centred at 0
    assert min(shifted) > max(null)
    assert np.isnan(gs.linear_mmd2_rbf(x[:1], x[:1]))


def test_linear_mmd_known_value_one_pair():
    # m = 1: h = k(x1,x2) + k(y1,y2) - k(x1,y2) - k(x2,y1) with sigma = 1
    x = np.array([[0.0], [1.0]])
    y = np.array([[2.0], [4.0]])
    k = lambda a, b: np.exp(-((a - b) ** 2) / 2)  # noqa: E731
    expected = k(0, 1) + k(2, 4) - k(0, 4) - k(1, 2)
    assert gs.linear_mmd2_rbf(x, y, bandwidth=1.0) == pytest.approx(expected)


def test_stratified_mmd_ignores_composition_shift_but_sees_conditional_shift():
    rng = np.random.default_rng(1)
    n = 6000
    # two buckets; reference 50/50. Shifted sample has 90/10 composition but identical per-bucket law.
    bx = rng.integers(0, 2, n)
    by = (rng.random(n) < 0.1).astype(int)
    mean = lambda b: np.where(b[:, None] == 0, 0.0, 3.0)  # noqa: E731
    x = mean(bx) + rng.normal(size=(n, 8))
    y_same = mean(by) + rng.normal(size=(n, 8))
    y_shift = mean(by) + 1.0 + rng.normal(size=(n, 8))
    bw = gs.median_bandwidth(x)
    plain = gs.linear_mmd2_rbf(x, y_same, bandwidth=bw)
    strat = gs.stratified_linear_mmd2(x, y_same, bx, by, bandwidth=bw)
    assert plain > 0.02  # unstratified MMD reacts to the composition change ...
    assert abs(strat) < 0.01  # ... the stratified one does not
    assert gs.stratified_linear_mmd2(x, y_shift, bx, by, bandwidth=bw) > 0.02


def test_degree_pair_buckets_shape_and_range():
    src = np.array([0, 0, 0, 0, 1, 2])
    dst = np.array([10, 10, 11, 12, 10, 11])
    b = gs.degree_pair_buckets(src, dst, n_bins=3)
    assert b.shape == src.shape and b.min() >= 0 and b.max() < 9


def test_turnover_stats_exact_values():
    assert gs.jaccard(np.array([1, 2, 3]), np.array([2, 3, 4])) == pytest.approx(2 / 4)
    assert gs.new_node_rate(np.array([1, 2, 3, 4]), np.array([3, 4, 5])) == pytest.approx(2 / 4)
    out = gs.turnover_stats(np.array([1, 2]), np.array([9]), np.array([1, 2]), np.array([9]))
    assert out["one_minus_jaccard_src"] == 0.0 and out["new_node_rate_dst"] == 0.0
