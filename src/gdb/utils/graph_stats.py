"""Window statistics for drift detection and drift purity tests.

Every function is pure and takes a *window of events* (arrays ``src``, ``dst``, optionally
edge marks), never a whole stream, so Gate G1 and the detector can call them directly.
Node ids are global integer ids (TGB convention: sources ``0..N_src-1``, destinations after).

Nothing here reads ground-truth drift information.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.stats import ks_2samp, wasserstein_distance

# ----------------------------------------------------------------------------------------------
# Degree distributions
# ----------------------------------------------------------------------------------------------


def node_degrees(nodes: np.ndarray) -> np.ndarray:
    r"""Event-degree of every active node in a window.

    $$d(v) = \bigl|\{e \in \mathcal{W} : v \in e\text{'s endpoint list}\}\bigr|$$

    ``nodes`` is the list of endpoints on one side (all sources, or all destinations).
    Returns one count per distinct node (inactive nodes are not included).
    """
    if len(nodes) == 0:
        return np.zeros(0, dtype=np.int64)
    return np.unique(nodes, return_counts=True)[1].astype(np.int64)


def degree_distributions(src: np.ndarray, dst: np.ndarray) -> dict[str, np.ndarray]:
    """Source-side, destination-side and total degree arrays of a window."""
    return {
        "src": node_degrees(src),
        "dst": node_degrees(dst),
        "all": node_degrees(np.concatenate([src, dst])),
    }


def degree_ks(deg_a: np.ndarray, deg_b: np.ndarray) -> float:
    r"""Two-sample Kolmogorov-Smirnov statistic between two degree samples.

    $$D = \sup_x \lvert F_a(x) - F_b(x) \rvert$$
    """
    if len(deg_a) == 0 or len(deg_b) == 0:
        return float("nan")
    return float(ks_2samp(deg_a, deg_b).statistic)


# ----------------------------------------------------------------------------------------------
# Window graph, normalised Laplacian and KPM density of states
# ----------------------------------------------------------------------------------------------


def window_adjacency(
    src: np.ndarray,
    dst: np.ndarray,
    node_subset: np.ndarray | None = None,
) -> tuple[sp.csr_matrix, np.ndarray]:
    r"""Symmetric binary adjacency matrix of the simple graph induced by a window.

    Multi-edges are collapsed, self-loops dropped. If ``node_subset`` is given, only events with
    both endpoints in the subset are kept (the persistent-node induced subgraph $G_k[P]$).
    Returns ``(A, nodes)`` where ``nodes[i]`` is the global id of row/column ``i``.
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    if node_subset is not None:
        keep = np.isin(src, node_subset) & np.isin(dst, node_subset)
        src, dst = src[keep], dst[keep]
    keep = src != dst
    src, dst = src[keep], dst[keep]
    nodes = np.unique(np.concatenate([src, dst]))
    n = len(nodes)
    if n == 0:
        return sp.csr_matrix((0, 0)), nodes
    u = np.searchsorted(nodes, src)
    v = np.searchsorted(nodes, dst)
    a = sp.coo_matrix((np.ones(len(u)), (u, v)), shape=(n, n)).tocsr()
    a = a + a.T
    a.data[:] = 1.0
    return a.tocsr(), nodes


def normalized_adjacency(a: sp.csr_matrix) -> sp.csr_matrix:
    r"""$\hat A = D^{-1/2} A D^{-1/2}$ (spectrum in $[-1, 1]$); the Laplacian is $L = I - \hat A$."""
    d = np.asarray(a.sum(axis=1)).ravel()
    inv = np.zeros_like(d, dtype=float)
    inv[d > 0] = d[d > 0] ** -0.5
    s = sp.diags(inv)
    return (s @ a @ s).tocsr()


@dataclass(frozen=True)
class Spectrum:
    r"""A discrete probability measure on $[0, 2]$ (eigenvalues of the normalised Laplacian)."""

    values: np.ndarray
    weights: np.ndarray


def exact_spectrum(a: sp.csr_matrix) -> Spectrum:
    r"""Exact empirical spectral measure $\frac1n\sum_i \delta_{\lambda_i}$ (dense; tests only)."""
    n = a.shape[0]
    lam = np.linalg.eigvalsh(np.eye(n) - normalized_adjacency(a).toarray())
    return Spectrum(lam, np.full(n, 1.0 / n))


def _jackson(n_moments: int) -> np.ndarray:
    r"""Jackson damping coefficients $g_m$, $m = 0..N-1$, with $N$ = ``n_moments``.

    $$g_m = \frac{(N-m+1)\cos\frac{\pi m}{N+1} + \sin\frac{\pi m}{N+1}\cot\frac{\pi}{N+1}}{N+1}$$
    """
    n = n_moments
    m = np.arange(n)
    return (
        (n - m + 1) * np.cos(np.pi * m / (n + 1))
        + np.sin(np.pi * m / (n + 1)) / np.tan(np.pi / (n + 1))
    ) / (n + 1)


def kpm_dos(
    a: sp.csr_matrix,
    *,
    n_moments: int = 100,
    n_probes: int = 30,
    n_grid: int = 1024,
    rng: np.random.Generator | None = None,
) -> Spectrum:
    r"""Kernel Polynomial Method estimate of the density of states of $L = I - \hat A$.

    With $\hat A = D^{-1/2} A D^{-1/2}$ (spectrum in $[-1,1]$, no rescaling needed) the Chebyshev
    moments are estimated with Rademacher probes $z$:

    $$\mu_m = \frac{1}{n\,n_p}\sum_{p=1}^{n_p} z_p^\top T_m(\hat A) z_p,\qquad
      T_{m+1}(x) = 2xT_m(x) - T_{m-1}(x).$$

    The Jackson-damped density at Chebyshev nodes $x_k = \cos\frac{\pi(k+1/2)}{K}$ is

    $$w_k = \frac1K\Bigl(g_0\mu_0 + 2\sum_{m\ge1} g_m \mu_m T_m(x_k)\Bigr),$$

    returned as a probability measure on eigenvalues $\lambda_k = 1 - x_k$ (tiny negative weights
    from probe noise are clipped, then renormalised). Cost: $O(M\, n_p\, |E|)$, no eigendecomposition.
    """
    rng = (
        np.random.default_rng(0) if rng is None else rng
    )  # fixed default: no hidden nondeterminism
    n = a.shape[0]
    if n == 0:
        return Spectrum(np.zeros(0), np.zeros(0))
    ah = normalized_adjacency(a)
    z = rng.integers(0, 2, size=(n, n_probes)).astype(np.float64) * 2.0 - 1.0
    mu = np.zeros(n_moments)
    v_prev, v_cur = z, ah @ z
    mu[0] = np.sum(z * z) / (n * n_probes)
    if n_moments > 1:
        mu[1] = np.sum(z * v_cur) / (n * n_probes)
    for m in range(2, n_moments):
        v_next = 2.0 * (ah @ v_cur) - v_prev
        mu[m] = np.sum(z * v_next) / (n * n_probes)
        v_prev, v_cur = v_cur, v_next
    g = _jackson(n_moments)
    k = np.arange(n_grid)
    theta = np.pi * (k + 0.5) / n_grid
    x = np.cos(theta)
    series = g[0] * mu[0] + 2.0 * (
        np.cos(np.outer(theta, np.arange(1, n_moments))) @ (g[1:] * mu[1:])
    )
    w = np.clip(series / n_grid, 0.0, None)
    w = w / w.sum()
    return Spectrum(1.0 - x, w)


def dos_w1(s1: Spectrum, s2: Spectrum) -> float:
    r"""Wasserstein-1 distance between two spectral measures on the real line.

    $$W_1(\rho_1,\rho_2) = \int_0^2 \lvert F_1(\lambda) - F_2(\lambda) \rvert\, d\lambda$$
    """
    if len(s1.values) == 0 or len(s2.values) == 0:
        return float("nan")
    return float(wasserstein_distance(s1.values, s2.values, s1.weights, s2.weights))


def window_dos(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    node_subset: np.ndarray | None = None,
    n_moments: int = 100,
    n_probes: int = 30,
    rng: np.random.Generator | None = None,
) -> Spectrum:
    """KPM density of states of the (optionally persistent-node induced) window graph."""
    a, _ = window_adjacency(src, dst, node_subset)
    return kpm_dos(a, n_moments=n_moments, n_probes=n_probes, rng=rng)


# ----------------------------------------------------------------------------------------------
# k-core numbers
# ----------------------------------------------------------------------------------------------


def core_numbers(src: np.ndarray, dst: np.ndarray) -> dict[int, int]:
    r"""Core number $\mathrm{core}(v)$: the largest $k$ such that $v$ lies in the $k$-core.

    Uses ``networkx.core_number`` on the simple undirected window graph (Batagelj-Zaversnik,
    $O(|E|)$). Self-loops are dropped.
    """
    g = nx.Graph()
    g.add_edges_from((int(u), int(v)) for u, v in zip(src, dst) if u != v)
    return dict(nx.core_number(g))


# ----------------------------------------------------------------------------------------------
# Linear-time MMD (RBF, median heuristic) and its degree-stratified version
# ----------------------------------------------------------------------------------------------


def median_bandwidth(
    x: np.ndarray,
    y: np.ndarray | None = None,
    *,
    max_samples: int = 1000,
    rng: np.random.Generator | None = None,
) -> float:
    r"""Median heuristic: $\sigma = \operatorname{median}\{\lVert z_i - z_j\rVert : i<j\}$.

    Computed on a random subsample (at most ``max_samples`` points) of ``x`` (pooled with ``y``).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    z = x if y is None else np.concatenate([x, y])
    if len(z) > max_samples:
        z = z[rng.choice(len(z), max_samples, replace=False)]
    z = np.asarray(z, dtype=np.float64)
    sq = (z**2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * z @ z.T, 0.0)
    iu = np.triu_indices(len(z), k=1)
    med = float(np.sqrt(np.median(d2[iu]))) if len(iu[0]) else 1.0
    return med if med > 0 else 1.0


def _rbf_pairs(a: np.ndarray, b: np.ndarray, bandwidth: float) -> np.ndarray:
    d2 = np.sum((a.astype(np.float64) - b.astype(np.float64)) ** 2, axis=1)
    return np.exp(-d2 / (2.0 * bandwidth**2))


def linear_mmd2_rbf(x: np.ndarray, y: np.ndarray, *, bandwidth: float | None = None) -> float:
    r"""Linear-time unbiased MMD$^2$ with an RBF kernel (Gretton et al., JMLR 2012).

    With $m=\lfloor\min(n_x,n_y)/2\rfloor$ and $k(a,b)=\exp(-\lVert a-b\rVert^2/2\sigma^2)$:

    $$\widehat{\mathrm{MMD}}^2_\ell = \frac1m\sum_{i=1}^{m}
      \bigl[k(x_{2i-1},x_{2i}) + k(y_{2i-1},y_{2i}) - k(x_{2i-1},y_{2i}) - k(x_{2i},y_{2i-1})\bigr]$$

    $O(m)$ time. $\sigma$ is the median heuristic on the pooled sample unless given. Can be negative
    (unbiased estimator). Returns ``nan`` when a sample has fewer than 2 points.
    """
    m = min(len(x), len(y)) // 2
    if m < 1:
        return float("nan")
    if bandwidth is None:
        bandwidth = median_bandwidth(x, y)
    x1, x2, y1, y2 = x[0 : 2 * m : 2], x[1 : 2 * m : 2], y[0 : 2 * m : 2], y[1 : 2 * m : 2]
    h = (
        _rbf_pairs(x1, x2, bandwidth)
        + _rbf_pairs(y1, y2, bandwidth)
        - _rbf_pairs(x1, y2, bandwidth)
        - _rbf_pairs(x2, y1, bandwidth)
    )
    return float(h.mean())


def log_degree_bins(deg: np.ndarray, n_bins: int = 3) -> np.ndarray:
    r"""Degree bucket $\min(\lfloor\log_2 d\rfloor, n_{bins}-1)$ for a degree array ($d\ge1$)."""
    return np.minimum(np.floor(np.log2(np.maximum(deg, 1))).astype(np.int64), n_bins - 1)


def degree_pair_buckets(src: np.ndarray, dst: np.ndarray, n_bins: int = 3) -> np.ndarray:
    r"""Per-event degree-pair bucket $b(e) = (\mathrm{bin}(d_u),\mathrm{bin}(d_v))$ as one integer.

    Degrees are the window's own event-degrees; buckets encode ``bin(d_u) * n_bins + bin(d_v)``.
    """
    su, inv_s = np.unique(src, return_inverse=True)
    dv, inv_d = np.unique(dst, return_inverse=True)
    deg_s = np.bincount(inv_s)
    deg_d = np.bincount(inv_d)
    return log_degree_bins(deg_s[inv_s], n_bins) * n_bins + log_degree_bins(deg_d[inv_d], n_bins)


def stratified_linear_mmd2(
    x: np.ndarray,
    y: np.ndarray,
    bucket_x: np.ndarray,
    bucket_y: np.ndarray,
    *,
    bandwidth: float | None = None,
    min_count: int = 20,
) -> float:
    r"""Degree-stratified linear-time MMD (Blueprint §5.1, statistic $s_\alpha$).

    $$s_\alpha = \sum_b \pi_b\, \widehat{\mathrm{MMD}}^2_\ell\bigl(X^{(b)}, Y^{(b)}\bigr)$$

    where $b$ ranges over buckets with at least ``min_count`` events on both sides and $\pi_b$ is
    the share of ``x`` (the reference) in bucket $b$, renormalised over the buckets used. The
    bandwidth is fixed across buckets (median heuristic on the reference ``x`` unless given), so
    the estimate targets $P(X\mid A)$ and not $P(X)$.
    """
    if bandwidth is None:
        bandwidth = median_bandwidth(x)
    total, wsum = 0.0, 0.0
    for b in np.unique(bucket_x):
        xb, yb = x[bucket_x == b], y[bucket_y == b]
        if len(xb) < min_count or len(yb) < min_count:
            continue
        pi = len(xb) / len(x)
        total += pi * linear_mmd2_rbf(xb, yb, bandwidth=bandwidth)
        wsum += pi
    return float(total / wsum) if wsum > 0 else float("nan")


# ----------------------------------------------------------------------------------------------
# Node-set turnover
# ----------------------------------------------------------------------------------------------


def jaccard(nodes_a: np.ndarray, nodes_b: np.ndarray) -> float:
    r"""Jaccard similarity of two active node sets, $J = |A\cap B| / |A\cup B|$."""
    a, b = np.unique(nodes_a), np.unique(nodes_b)
    union = len(np.union1d(a, b))
    return float(len(np.intersect1d(a, b, assume_unique=True)) / union) if union else 1.0


def new_node_rate(nodes_k: np.ndarray, nodes_ref: np.ndarray) -> float:
    r"""Share of the current window's active nodes absent from the reference window.

    $$r_{\text{new}} = \frac{|V_k \setminus V_R|}{|V_k|}$$
    """
    vk = np.unique(nodes_k)
    if len(vk) == 0:
        return 0.0
    return float(len(np.setdiff1d(vk, np.unique(nodes_ref), assume_unique=True)) / len(vk))


def turnover_stats(
    src_k: np.ndarray, dst_k: np.ndarray, src_ref: np.ndarray, dst_ref: np.ndarray
) -> dict[str, float]:
    r"""Turnover on each side: $1-J$ and new-node rate (the $s_\gamma$ ingredients, §5.1)."""
    return {
        "one_minus_jaccard_src": 1.0 - jaccard(src_k, src_ref),
        "one_minus_jaccard_dst": 1.0 - jaccard(dst_k, dst_ref),
        "new_node_rate_src": new_node_rate(src_k, src_ref),
        "new_node_rate_dst": new_node_rate(dst_k, dst_ref),
    }
