r"""Revised topology statistic $s_\beta^{\rm strata}$ (G1 protocol v2 candidate; the v1 DoS statistic is kept).

Motivation (G1 v1 failure). The v1 statistic compares the spectrum of the persistent-node induced
subgraph $G_a[P]$ with the pool. Under pure entity turnover the persistent subgraph is a *thinner*
graph (80% of the reference events in the official runs), and thinning alone shifts the normalised
Laplacian spectrum, so $s_\beta$ reacted to $\gamma$. It was also nearly blind to hub drift on
``wiki_like`` (mean $z\approx0.24$).

Statistic. Fix the pool $R$ (the $m$ reference windows a window is compared with). On each side
$u\in\{{\rm src},{\rm dst}\}$ rank the pool's nodes by mean per-window event count and cut the ranking
into $K$ strata of equal pool *event mass* (stratum 0 = the heaviest nodes, i.e. the hubs). For a set
of events $E$ let $p^u_k(E)$ be the share of $E$ whose side-$u$ endpoint lies in stratum $k$. With
$P=V_a\cap V_R$ and $E_a[P]$ the events of $a$ with both endpoints in $P$,

$$ s^{u}_\beta(a)=\mathrm{TV}\big(p^u(E_a[P]),\ \bar p^u\big)=\tfrac12\sum_k\big|p^u_k(E_a[P])-\bar p^u_k\big|, $$

where $\bar p^u$ is the average over the pool windows of $p^u(E_w[P])$ (same $P$). Standardisation
and the pool-size matching are as for the other statistics; the two sides are combined after
standardisation by a maximum, exactly as the two $\gamma$ components are.

Why it is turnover-robust. Strata are fixed by the *pool*; $p^u(\cdot)$ is a share (a ratio), so it
has no dependence on the number of events. If entity turnover re-labels node slots independently of
the slot's activity level (the generator's churn keeps block and activity level), the persistent
events are a uniform thinning of the events of unchanged nodes, which leaves the expected stratum
shares unchanged. Assumption (stated, tested only on this generator): turnover is independent of
node activity. Sampling variance of $p$ grows by $1/(\text{kept fraction})$, which the standardisation
(done at full size) understates by a factor $\approx\sqrt{1/0.8}$ at 20% turnover.

Why it sees hub drift. A hub intervention moves event mass between heavy and light nodes, i.e.
between strata, and the heaviest stratum is a small set of nodes whose share moves with the
intervention. Nothing here reads labels, schedules or generator internals.
"""

from __future__ import annotations

import numpy as np

from gdb.detect.window_stats import Window, persistent_nodes

N_STRATA = 4  # equal-mass strata; sensitivity to K is a development-only question


def _side(w: Window, side: str) -> np.ndarray:
    return w.src if side == "src" else w.dst


def pool_strata(
    pool: list[Window], side: str, n_strata: int = N_STRATA
) -> tuple[np.ndarray, np.ndarray]:
    r"""Equal-mass strata of the pool's nodes on one side: returns ``(node_ids, stratum_of_node)``.

    Nodes are sorted by decreasing mean per-window event count; node $v$ gets stratum
    $\lfloor K\,c_{<v}/c\rfloor$ with $c_{<v}$ the mass of heavier nodes (so a node holding more than
    $1/K$ of the mass sits in stratum 0 alone with its successors up to the first cut).
    """
    ids = np.concatenate([_side(w, side) for w in pool])
    nodes, cnt = np.unique(ids, return_counts=True)
    order = np.argsort(-cnt, kind="stable")
    cum_before = np.cumsum(cnt[order]) - cnt[order]
    strata = np.empty(len(nodes), dtype=np.int64)
    strata[order] = np.minimum((n_strata * cum_before / cnt.sum()).astype(np.int64), n_strata - 1)
    return nodes, strata


def _shares(
    w: Window, keep_nodes: np.ndarray, side: str, nodes: np.ndarray, strata: np.ndarray, k: int
):
    m = np.isin(w.src, keep_nodes) & np.isin(w.dst, keep_nodes)
    if m.sum() == 0:
        return None
    ids = _side(w, side)[m]
    idx = np.searchsorted(nodes, ids)
    ok = (idx < len(nodes)) & (nodes[np.minimum(idx, len(nodes) - 1)] == ids)
    if not ok.all():  # cannot happen for persistent nodes (they are in the pool)
        raise ValueError("event endpoint outside the pool strata")
    return np.bincount(strata[idx], minlength=k) / m.sum()


def strata_tv(a: Window, pool: list[Window], *, n_strata: int = N_STRATA) -> tuple[float, float]:
    r"""$(s^{\rm src}_\beta,s^{\rm dst}_\beta)$ of window ``a`` against ``pool`` (see module docstring)."""
    p = persistent_nodes(a, pool)
    out = []
    for side in ("src", "dst"):
        nodes, strata = pool_strata(pool, side, n_strata)
        pa = _shares(a, p, side, nodes, strata, n_strata)
        qs = [_shares(w, p, side, nodes, strata, n_strata) for w in pool]
        qs = [q for q in qs if q is not None]
        if pa is None or not qs:
            out.append(float("nan"))
            continue
        out.append(float(0.5 * np.abs(pa - np.mean(qs, axis=0)).sum()))
    return out[0], out[1]


def node_share_l2(a: Window, pool: list[Window]) -> tuple[float, float]:
    r"""Bias-corrected node-share distance $(s^{\rm src}_\beta,s^{\rm dst}_\beta)$ (candidate v2b).

    Per side let $\hat p_v=c_v/n$ be the share of the window's persistent events ($n$ of them, both
    endpoints in $P$) at node $v$, and $\bar q_v$ the mean share over the pool windows restricted to
    the same $P$. The unbiased estimate of $\lVert p-\bar q\rVert_2^2$ is

    $$ s=\sum_v(\hat p_v-\bar q_v)^2-\frac1{n-1}\sum_v\hat p_v(1-\hat p_v), $$

    because $\mathbb E\sum_v(\hat p_v-p_v)^2=\sum_v p_v(1-p_v)/n$ for multinomial counts. The
    correction removes the dependence of the null mean on the number of events, which is exactly what
    turnover changes (persistent events fall to ~80%); its variance still grows as $n$ falls, a
    residual effect that the development study measures. Block-level interventions (a whole block's
    rate scaled) move every node of the block, so the L2 distance sees them even when the node-level
    mass strata of :func:`strata_tv` do not.
    """
    p = persistent_nodes(a, pool)
    out = []
    for side in ("src", "dst"):
        ids_a = _kept_ids(a, p, side)
        pools = [k for k in (_kept_ids(w, p, side) for w in pool) if len(k) > 0]
        if len(ids_a) < 2 or not pools:
            out.append(float("nan"))
            continue
        nodes = np.unique(np.concatenate([ids_a, *pools]))
        pa = _id_shares(ids_a, nodes)
        q = np.mean([_id_shares(k, nodes) for k in pools], axis=0)
        out.append(float(((pa - q) ** 2).sum() - (pa * (1 - pa)).sum() / (len(ids_a) - 1)))
    return out[0], out[1]


def _kept_ids(w: Window, persistent: np.ndarray, side: str) -> np.ndarray:
    m = np.isin(w.src, persistent) & np.isin(w.dst, persistent)
    return _side(w, side)[m]


def _id_shares(ids: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    return np.bincount(np.searchsorted(nodes, ids), minlength=len(nodes)) / len(ids)
