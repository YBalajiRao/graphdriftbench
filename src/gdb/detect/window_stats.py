r"""Reference-vs-window statistics $s_\gamma,s_\beta,s_\alpha$ of Blueprint §5.1 (pure, array inputs).

A *window* is a dict-like ``(src, dst, marks)`` triple of the events of one block of $w$ events.
Nothing here reads labels, schedules or generator internals. The reference is a list of windows;
the statistic of a target window $a$ is computed against a reference *pool* of $m$ windows, so the
same function serves a test window (pool = the $m$ most recent reference windows) and a
leave-one-out reference window (pool = the other $m$ reference windows): the two have the same
pool size, which keeps the standardisation unbiased.

* $s_\gamma$: $(1-J(V_a,V_R),\ |V_a\setminus V_R|/|V_a|)$ with $V_R$ the active nodes of the pool.
* $s_\beta=W_1(\hat\rho_a,\bar\rho_R)$: DoS of $G_a[P]$ against the mean DoS of the pool windows,
  $P=V_a\cap V_R$ the persistent nodes.
* $s_\alpha=\sum_b\pi_b\widehat{\mathrm{MMD}}^2_\ell(X_a^{(b)},X_R^{(b)})$ on events with both ends in $P$.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gdb.utils import graph_stats as gs


@dataclass(frozen=True)
class Window:
    """Events of one window: ids and edge marks."""

    src: np.ndarray
    dst: np.ndarray
    marks: np.ndarray

    def nodes(self) -> np.ndarray:
        return np.unique(np.concatenate([self.src, self.dst]))


def _pool_nodes(pool: list[Window]) -> np.ndarray:
    return np.unique(np.concatenate([w.nodes() for w in pool]))


def persistent_nodes(a: Window, pool: list[Window]) -> np.ndarray:
    r"""$P=V_a\cap V_R$."""
    return np.intersect1d(a.nodes(), _pool_nodes(pool), assume_unique=True)


def gamma_components(a: Window, pool: list[Window]) -> tuple[float, float]:
    r"""$(1-J(V_a,V_R),\ |V_a\setminus V_R|/|V_a|)$ over all active nodes (both sides)."""
    v_r = _pool_nodes(pool)
    return 1.0 - gs.jaccard(a.nodes(), v_r), gs.new_node_rate(a.nodes(), v_r)


def window_dos_on(a: Window, nodes: np.ndarray, *, n_moments: int, n_probes: int, seed: int):
    """KPM DoS of the window graph induced on ``nodes`` (fixed probe seed per window)."""
    return gs.window_dos(
        a.src,
        a.dst,
        node_subset=nodes,
        n_moments=n_moments,
        n_probes=n_probes,
        rng=np.random.default_rng(seed),
    )


def mean_spectrum(spectra: list[gs.Spectrum]) -> gs.Spectrum:
    r"""$\bar\rho=\frac1m\sum_i\hat\rho_i$ (all KPM spectra share the Chebyshev grid)."""
    v = spectra[0].values
    if any(not np.array_equal(s.values, v) for s in spectra):
        raise ValueError("spectra must share one support grid")
    w = np.mean([s.weights for s in spectra], axis=0)
    return gs.Spectrum(v, w / w.sum())


def alpha_statistic(a: Window, pool: list[Window], persistent: np.ndarray) -> float:
    r"""Degree-stratified linear MMD$^2$ between the window and the pooled reference marks.

    Buckets are computed per window on its own persistent events, so degree scales match.
    """

    def part(w: Window) -> tuple[np.ndarray, np.ndarray]:
        keep = np.isin(w.src, persistent) & np.isin(w.dst, persistent)
        if keep.sum() == 0:
            return np.zeros((0, w.marks.shape[1]), np.float32), np.zeros(0, np.int64)
        return w.marks[keep], gs.degree_pair_buckets(w.src[keep], w.dst[keep])

    xa, ba = part(a)
    parts = [part(w) for w in pool]
    xr = np.concatenate([p[0] for p in parts])
    br = np.concatenate([p[1] for p in parts])
    if len(xa) < 40 or len(xr) < 40:
        return float("nan")
    return gs.stratified_linear_mmd2(xr, xa, br, ba, bandwidth=gs.median_bandwidth(xr))
