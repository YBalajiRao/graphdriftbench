r"""Gate G1, protocol-v2 development tooling: raw statistic series, then variant calibration.

``raw_case`` computes every expensive raw statistic once per stream (v1 DoS beta, v2 strata beta,
gamma, alpha, window losses) from learner-visible arrays only. ``calibrate`` turns the raw series into
the z-score row format of :mod:`gdb.eval.g1` for any (beta, delta, floor) variant, so variants are
compared on the same streams without re-running the TGN. Nothing here reads a label; the v1
``compute_case`` is untouched (``calibrate(..., delta='v1', beta='dos')`` reproduces its z-scores).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gdb.detect import delta_residual as dr
from gdb.detect import standardise as sd
from gdb.detect import topology as tp
from gdb.detect import window_stats as ws
from gdb.eval import g1
from gdb.eval.g1 import PROTO, TYPES, G1Protocol
from gdb.utils import graph_stats as gs

SERIES = ("g_jac", "g_new", "beta", "alpha", "tv_src", "tv_dst", "l2_src", "l2_dst")


def raw_case(
    wins: list[ws.Window], losses: np.ndarray, *, seed: int, proto: G1Protocol = PROTO
) -> dict:
    """Raw reference series (leave-one-out, pool = other r-1 windows) and test values (pool = ref[1:])."""
    r, npost = proto.r, proto.n_post
    if len(wins) != r + npost or len(losses) != r + npost:
        raise ValueError("expected r reference windows followed by n_post test windows")
    ref, tests = wins[:r], wins[r:]
    kw = {"n_moments": proto.n_moments, "n_probes": proto.n_probes}

    def others(j: int) -> list[ws.Window]:
        return [ref[i] for i in range(r) if i != j]

    dos_ref = [
        ws.window_dos_on(ref[i], ws.persistent_nodes(ref[i], others(i)), seed=seed * 100 + i, **kw)
        for i in range(r)
    ]
    rs = {k: np.empty(r) for k in SERIES}
    for j in range(r):
        pool = others(j)
        rs["g_jac"][j], rs["g_new"][j] = ws.gamma_components(ref[j], pool)
        rs["beta"][j] = gs.dos_w1(
            dos_ref[j], ws.mean_spectrum([dos_ref[i] for i in range(r) if i != j])
        )
        rs["alpha"][j] = ws.alpha_statistic(ref[j], pool, ws.persistent_nodes(ref[j], pool))
        rs["tv_src"][j], rs["tv_dst"][j] = tp.strata_tv(ref[j], pool)
        rs["l2_src"][j], rs["l2_dst"][j] = tp.node_share_l2(ref[j], pool)
    pool_t = ref[1:]
    mean_dos_t = ws.mean_spectrum(dos_ref[1:])
    ts = {k: np.empty(len(tests)) for k in SERIES}
    pers = []
    for m, a in enumerate(tests):
        p_a = ws.persistent_nodes(a, pool_t)
        pers.append(int((np.isin(a.src, p_a) & np.isin(a.dst, p_a)).sum()))
        dos_a = ws.window_dos_on(a, p_a, seed=seed * 100 + 50 + m, **kw)
        ts["g_jac"][m], ts["g_new"][m] = ws.gamma_components(a, pool_t)
        ts["beta"][m] = gs.dos_w1(dos_a, mean_dos_t)
        ts["alpha"][m] = ws.alpha_statistic(a, pool_t, p_a)
        ts["tv_src"][m], ts["tv_dst"][m] = tp.strata_tv(a, pool_t)
        ts["l2_src"][m], ts["l2_dst"][m] = tp.node_share_l2(a, pool_t)
    return {
        "ref": {k: v.tolist() for k, v in rs.items()},
        "test": {k: v.tolist() for k, v in ts.items()},
        "losses": np.asarray(losses, float).tolist(),
        "n_act": float(np.mean([len(w.nodes()) for w in ref])),
        "n_pool": float(np.mean([len(ws._pool_nodes(others(j))) for j in range(r)])),
        "pers_events_test": pers,
        "pers_events_ref_mean": float(
            np.mean(
                [
                    (
                        np.isin(ref[i].src, ws.persistent_nodes(ref[i], others(i)))
                        & np.isin(ref[i].dst, ws.persistent_nodes(ref[i], others(i)))
                    ).sum()
                    for i in range(r)
                ]
            )
        ),
    }


def calibrate(
    raw: dict,
    *,
    seed: int,
    beta: str = "dos",  # "dos" (v1) | "strata" | "l2"
    delta: str = "v1",  # "v1" | "bounded" | "none"
    gate: bool = False,
    floor_mult: float = 1.0,
    proto: G1Protocol = PROTO,
) -> dict:
    """z-score row (``z_<type>_w<m>``) for one variant; ``floor_mult`` scales the gamma resolution floor."""
    ref = {k: np.asarray(v) for k, v in raw["ref"].items()}
    test = {k: np.asarray(v) for k, v in raw["test"].items()}
    bad = [k for k in SERIES if not (np.all(np.isfinite(ref[k])) and np.all(np.isfinite(test[k])))]
    if bad:  # recorded as a failed stream, never dropped or imputed
        return {"status": f"nonfinite:{bad}"}
    loss = np.asarray(raw["losses"])
    r = proto.r
    floors = {"g_jac": floor_mult / raw["n_pool"], "g_new": floor_mult / raw["n_act"]}
    keys = SERIES
    mom = {
        k: sd.moving_block_moments(
            ref[k],
            block=proto.block,
            n_boot=proto.n_boot,
            seed=seed * 7 + n,
            sigma_floor=floors.get(k, 0.0),
        )
        for n, k in enumerate(keys)
    }

    def z(k: str, vals: np.ndarray) -> np.ndarray:
        return (vals - mom[k].mu) / mom[k].sigma

    def zb(src: dict) -> np.ndarray:
        if beta == "dos":
            return z("beta", src["beta"])
        pre = {"strata": "tv", "l2": "l2"}[beta]
        return np.maximum(z(f"{pre}_src", src[f"{pre}_src"]), z(f"{pre}_dst", src[f"{pre}_dst"]))

    zr = {
        "alpha": z("alpha", ref["alpha"]),
        "beta": zb(ref),
        "gamma": np.maximum(z("g_jac", ref["g_jac"]), z("g_new", ref["g_new"])),
    }
    zt = {
        "alpha": z("alpha", test["alpha"]),
        "beta": zb(test),
        "gamma": np.maximum(z("g_jac", test["g_jac"]), z("g_new", test["g_new"])),
    }
    f_ref = np.stack([zr["alpha"], zr["beta"], zr["gamma"]], axis=1)
    f_test = np.stack([zt["alpha"], zt["beta"], zt["gamma"]], axis=1)
    l_ref, l_test = loss[:r], loss[r:]
    if delta == "v1":
        w_h = sd.huber_fit(f_ref, l_ref)
        res_ref = sd.loo_residuals(f_ref, l_ref)
        res_test = sd.residual(w_h, f_test, l_test)
    else:
        res_ref, res_test = dr.jackknife_residuals(f_ref, l_ref, f_test, l_test, nuisance=delta)
    md = sd.moving_block_moments(res_ref, block=proto.block, n_boot=proto.n_boot, seed=seed * 7 + 9)
    zd = (res_test - md.mu) / md.sigma
    if gate:
        zd = dr.attribution_gate(zd, f_test.max(axis=1), proto.z_fire)
    row: dict = {"status": "ok"}
    for m in range(len(zd)):
        for t, v in zip(TYPES, [zt["alpha"][m], zt["beta"][m], zt["gamma"][m], zd[m]]):
            row[f"z_{t}_w{m}"] = float(v)
    return row


def cluster_bootstrap(
    df: pd.DataFrame, m: int | str = 0, *, n_boot: int = 2000, seed: int = 0
) -> dict:
    r"""Seed-clustered percentile bootstrap for paired streams (drifted streams only).

    The four drift types share the base stream of a seed, so streams of one seed are not independent.
    The resampling unit is the *seed*: draw $S$ seeds with replacement, keep all drifted streams of each
    drawn seed, and recompute (i) argmax macro-F1, (ii) the pooled $\delta$ fire rate on pure
    $\alpha/\beta/\gamma$ streams, (iii) the 4x4 mean-z audit matrix. Stationary streams are not paired
    and do not enter these three quantities. Returns 95% percentile intervals.
    """
    d = df[(df.status == "ok") & df.drift.isin(g1.TYPES)]
    seeds = np.array(sorted(d.seed.unique()))
    groups = {s: d[d.seed == s] for s in seeds}
    rng = np.random.default_rng(seed)
    f1s, fires, mats = [], [], []
    for _ in range(n_boot):
        b = pd.concat([groups[s] for s in rng.choice(seeds, len(seeds))])
        z = g1.zmat(b, m)
        true = b.drift.to_numpy(dtype=object)
        f1s.append(g1.macro_f1(true, g1.argmax_labels(z))[0])
        pure = ~np.isin(true, ["delta"])
        fires.append(float((z[pure, 3] > g1.PROTO.z_fire).mean()))
        mats.append(np.stack([z[true == c].mean(axis=0) for c in g1.TYPES]))
    q = lambda v: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))  # noqa: E731
    mats = np.stack(mats)
    return {
        "n_seed_clusters": len(seeds),
        "macro_f1_ci95": q(f1s),
        "delta_fire_pooled_ci95": q(fires),
        "audit_lo": np.quantile(mats, 0.025, axis=0),
        "audit_hi": np.quantile(mats, 0.975, axis=0),
    }
