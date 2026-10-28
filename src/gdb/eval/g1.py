r"""Gate G1 (Blueprint §13): per-stream statistics, z-scores and scoring.

``compute_case`` consumes only arrays a learner may see (window ids, marks, per-event prequential
losses); labels enter only in :func:`score_g1`. See ``docs/progress/g1-protocol.md`` for the locked
protocol and every constant below.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gdb.detect import standardise as sd
from gdb.detect import window_stats as ws
from gdb.utils import graph_stats as gs

TYPES = ["alpha", "beta", "gamma", "delta"]


@dataclass(frozen=True)
class G1Protocol:
    """Locked constants (changing any of them after the official run is a protocol change)."""

    r: int = 10  # reference windows
    block: int = 2
    n_boot: int = 500
    z_fire: float = 3.0  # a statistic "fires" when z > z_fire (also: detection threshold)
    n_moments: int = 100
    n_probes: int = 200
    n_post: int = 3  # post-onset windows evaluated (first = primary)
    delta_fire_max: float = 0.15
    macro_f1_min: float = 0.80


PROTO = G1Protocol()


def window_losses(event_loss: np.ndarray, start: int, w: int, n_windows: int) -> np.ndarray:
    """Mean prequential loss of ``n_windows`` consecutive windows of ``w`` events from ``start``."""
    return np.array(
        [event_loss[start + m * w : start + (m + 1) * w].mean() for m in range(n_windows)]
    )


def compute_case(
    wins: list[ws.Window], losses: np.ndarray, *, seed: int, proto: G1Protocol = PROTO
) -> dict:
    r"""z-scores of the post-onset windows given ``r`` reference windows then ``n_post`` test windows.

    ``wins`` = ``r`` reference windows followed by ``n_post`` windows; ``losses`` aligned to ``wins``.
    Reference statistics are leave-one-out against pools of ``r-1`` windows; test windows are
    compared with the ``r-1`` most recent reference windows (same pool size).
    """
    r, npost = proto.r, proto.n_post
    if len(wins) != r + npost or len(losses) != r + npost:
        raise ValueError("expected r reference windows followed by n_post test windows")
    ref, tests = wins[:r], wins[r:]
    t0 = time.perf_counter()
    kw = {"n_moments": proto.n_moments, "n_probes": proto.n_probes}

    def others(j: int) -> list[ws.Window]:
        return [ref[i] for i in range(r) if i != j]

    dos_ref = [
        ws.window_dos_on(ref[i], ws.persistent_nodes(ref[i], others(i)), seed=seed * 100 + i, **kw)
        for i in range(r)
    ]
    s = {k: np.empty(r) for k in ("g_jac", "g_new", "beta", "alpha")}
    for j in range(r):
        pool = others(j)
        s["g_jac"][j], s["g_new"][j] = ws.gamma_components(ref[j], pool)
        mean_dos = ws.mean_spectrum([dos_ref[i] for i in range(r) if i != j])
        s["beta"][j] = gs.dos_w1(dos_ref[j], mean_dos)
        s["alpha"][j] = ws.alpha_statistic(ref[j], pool, ws.persistent_nodes(ref[j], pool))
    if not all(np.all(np.isfinite(v)) for v in s.values()):
        return {"status": "nonfinite_reference"}
    # resolution floor (pre-registered, gamma components only): one node out of the active set
    n_act = float(np.mean([len(w.nodes()) for w in ref]))
    n_pool = float(np.mean([len(ws._pool_nodes(others(j))) for j in range(r)]))
    floors = {"g_jac": 1.0 / n_pool, "g_new": 1.0 / n_act}
    mom = {
        k: sd.moving_block_moments(
            v,
            block=proto.block,
            n_boot=proto.n_boot,
            seed=seed * 7 + n,
            sigma_floor=floors.get(k, 0.0),
        )
        for n, (k, v) in enumerate(s.items())
    }
    zref = {
        "alpha": np.array([sd.zscore(x, mom["alpha"]) for x in s["alpha"]]),
        "beta": np.array([sd.zscore(x, mom["beta"]) for x in s["beta"]]),
        "gamma": np.maximum(
            [sd.zscore(x, mom["g_jac"]) for x in s["g_jac"]],
            [sd.zscore(x, mom["g_new"]) for x in s["g_new"]],
        ),
    }
    feats_ref = np.stack([zref["alpha"], zref["beta"], zref["gamma"]], axis=1)
    l_ref = losses[:r]
    w_h = sd.huber_fit(feats_ref, l_ref)
    res_ref = sd.loo_residuals(feats_ref, l_ref)
    mom_delta = sd.moving_block_moments(
        res_ref, block=proto.block, n_boot=proto.n_boot, seed=seed * 7 + 9
    )

    pool_test = ref[1:]
    out_pe = [
        int(
            (
                np.isin(ref[i].src, ws.persistent_nodes(ref[i], others(i)))
                & np.isin(ref[i].dst, ws.persistent_nodes(ref[i], others(i)))
            ).sum()
        )
        for i in range(r)
    ]
    mean_dos_test = ws.mean_spectrum(dos_ref[1:])
    out: dict = {"status": "ok", "pers_events_ref_mean": float(np.mean(out_pe))}
    for m, a in enumerate(tests):
        p_a = ws.persistent_nodes(a, pool_test)
        out[f"pers_events_w{m}"] = int((np.isin(a.src, p_a) & np.isin(a.dst, p_a)).sum())
        dos_a = ws.window_dos_on(a, p_a, seed=seed * 100 + 50 + m, **kw)
        jac, new = ws.gamma_components(a, pool_test)
        sb = gs.dos_w1(dos_a, mean_dos_test)
        sa = ws.alpha_statistic(a, pool_test, p_a)
        vals = {"g_jac": jac, "g_new": new, "beta": sb, "alpha": sa}
        bad = [k for k, v in vals.items() if not np.isfinite(v)]
        if bad:
            out["status"] = f"nonfinite:{bad}"
            continue
        za, zb = sd.zscore(sa, mom["alpha"]), sd.zscore(sb, mom["beta"])
        zg = max(sd.zscore(jac, mom["g_jac"]), sd.zscore(new, mom["g_new"]))
        s_delta = sd.residual(w_h, np.array([[za, zb, zg]]), np.array([losses[r + m]]))[0]
        zd = sd.zscore(s_delta, mom_delta)
        for k, v in zip(TYPES, [za, zb, zg, zd]):
            out[f"z_{k}_w{m}"] = float(v)
        out[f"loss_w{m}"] = float(losses[r + m])
        out.update({f"s_{k}_w{m}": float(v) for k, v in vals.items()})
    for k, mm in mom.items():
        out.update(
            {
                f"mu_{k}": mm.mu,
                f"sigma_{k}": mm.sigma,
                f"sigma_{k}_lo": mm.sigma_ci[0],
                f"sigma_{k}_hi": mm.sigma_ci[1],
            }
        )
    out.update(
        {
            "mu_delta": mom_delta.mu,
            "sigma_delta": mom_delta.sigma,
            "sigma_delta_lo": mom_delta.sigma_ci[0],
            "sigma_delta_hi": mom_delta.sigma_ci[1],
            "loss_ref_mean": float(l_ref.mean()),
            "loss_ref_sd": float(l_ref.std(ddof=1)),
            "huber_w": ";".join(f"{x:.4f}" for x in w_h),
            "stat_seconds": time.perf_counter() - t0,
        }
    )
    return out


# ------------------------------------------------------------------------------------------------
# Scoring (labels are used only below)
# ------------------------------------------------------------------------------------------------


def zmat(df: pd.DataFrame, m: int | str) -> np.ndarray:
    """(n, 4) z-scores in TYPES order for post-onset window ``m`` or the mean of the first three."""
    if m == "mean3":
        return np.stack(
            [df[[f"z_{t}_w{i}" for i in range(PROTO.n_post)]].mean(axis=1) for t in TYPES], axis=1
        )
    return df[[f"z_{t}_w{m}" for t in TYPES]].to_numpy()


def argmax_labels(z: np.ndarray) -> np.ndarray:
    """argmax of the four z-scores; exact ties are labelled ``tie`` (counted as errors)."""
    mx = z.max(axis=1, keepdims=True)
    tied = (z == mx).sum(axis=1) > 1
    lab = np.array(TYPES, dtype=object)[z.argmax(axis=1)]
    lab[tied] = "tie"
    return lab


def macro_f1(true: np.ndarray, pred: np.ndarray) -> tuple[float, dict]:
    """Macro-F1 over the four drift types; 'none'/'tie' predictions are errors, never a class."""
    f1 = {}
    for c in TYPES:
        tp = int(((true == c) & (pred == c)).sum())
        fp = int(((true != c) & (pred == c)).sum())
        fn = int(((true == c) & (pred != c)).sum())
        f1[c] = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    return float(np.mean(list(f1.values()))), f1


def confusion(true: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    cols = [*TYPES, "none", "tie"]
    return pd.DataFrame(
        [[int(((true == t) & (pred == p)).sum()) for p in cols] for t in TYPES],
        index=TYPES,
        columns=cols,
    )


def audit_matrix(
    df: pd.DataFrame, m: int | str, n_boot: int = 1000, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r"""$\mathbb E[z_{c'}\mid\text{inject }c]$ with 95% percentile bootstrap CIs over seeds."""
    rng = np.random.default_rng(seed)
    mean = pd.DataFrame(index=TYPES, columns=TYPES, dtype=float)
    lo, hi = mean.copy(), mean.copy()
    for c in TYPES:
        z = zmat(df[df.drift == c], m)
        mean.loc[c] = z.mean(axis=0)
        bs = np.stack([z[rng.integers(0, len(z), len(z))].mean(axis=0) for _ in range(n_boot)])
        lo.loc[c], hi.loc[c] = np.quantile(bs, 0.025, axis=0), np.quantile(bs, 0.975, axis=0)
    return mean, lo, hi


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUROC (average ranks for ties)."""
    if not positive.any() or positive.all():
        return float("nan")
    ranks = pd.Series(score).rank().to_numpy()
    n1, n0 = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def score_g1(df: pd.DataFrame, m: int | str = 0, proto: G1Protocol = PROTO, w: int = 1) -> dict:
    """All G1 quantities for one configuration.

    ``df``: one row per stream with ``drift`` in TYPES + 'none' and ``status``; failed rows are
    counted in ``n_failed`` (a configuration with any failed stream cannot pass). ``w`` = window
    size in events (for false alarms per 1e5 events).
    """
    ok = df[df.status == "ok"]
    drifted = ok[ok.drift.isin(TYPES)]
    stat = ok[ok.drift == "none"]
    z = zmat(drifted, m)
    true = drifted.drift.to_numpy(dtype=object)
    pred = argmax_labels(z)
    f1_plain, f1_by = macro_f1(true, pred)
    mean, lo, hi = audit_matrix(drifted, m)
    diag_rows = {c: bool(mean.loc[c].idxmax() == c) for c in TYPES}
    diag_cols = {c: bool(mean[c].idxmax() == c) for c in TYPES}
    fired = z.max(axis=1) > proto.z_fire
    pred_thr = pred.copy()
    pred_thr[~fired] = "none"
    f1_thr, f1_thr_by = macro_f1(true, pred_thr)
    zs = zmat(stat, m) if len(stat) else np.zeros((0, 4))
    fa = zs.max(axis=1) > proto.z_fire
    tp, fn, fp = int(fired.sum()), int((~fired).sum()), int(fa.sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    zw = np.stack([zmat(drifted, i).max(axis=1) for i in range(proto.n_post)], axis=1)
    hit = zw > proto.z_fire
    first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    zsw = (
        np.stack([zmat(stat, i).max(axis=1) for i in range(proto.n_post)], axis=1)
        if len(stat)
        else np.zeros((0, proto.n_post))
    )
    pure = drifted[drifted.drift.isin(["alpha", "beta", "gamma"])]
    zp = zmat(pure, m)[:, 3]
    dfire = zp > proto.z_fire
    per_type_dfire = {
        c: float(np.mean(dfire[pure.drift.to_numpy() == c])) for c in ["alpha", "beta", "gamma"]
    }
    zdel = np.r_[zmat(drifted, m)[:, 3], zs[:, 3]]
    isd = np.r_[(true == "delta"), np.zeros(len(zs), bool)]
    rng = np.random.default_rng(0)
    boots = [
        macro_f1(true[ix], pred[ix])[0]
        for ix in (rng.integers(0, len(true), len(true)) for _ in range(1000))
    ]
    res = {
        "macro_f1_argmax_ci95": (
            float(np.quantile(boots, 0.025)),
            float(np.quantile(boots, 0.975)),
        ),
        "n_streams": len(df),
        "n_failed": int((df.status != "ok").sum()),
        "macro_f1_argmax": f1_plain,
        "f1_by_type": f1_by,
        "macro_f1_thresholded": f1_thr,
        "f1_thr_by_type": f1_thr_by,
        "confusion_argmax": confusion(true, pred),
        "confusion_thresholded": confusion(true, pred_thr),
        "audit_mean": mean,
        "audit_lo": lo,
        "audit_hi": hi,
        "diag_dominant_rows": diag_rows,
        "diag_dominant_cols": diag_cols,
        "delta_fire_pooled": float(dfire.mean()) if len(pure) else float("nan"),
        "delta_fire_by_type": per_type_dfire,
        "delta_auroc_vs_rest": auroc(zdel, isd),
        "detection_precision": prec,
        "detection_recall": rec,
        "missed": fn,
        "false_alarm_streams": fp,
        "detected_first_in_window": {int(k): int((first == k).sum()) for k in range(proto.n_post)},
        "missed_within_n_post": int((first == -1).sum()),
        "mean_delay_windows_detected": float(first[first >= 0].mean())
        if (first >= 0).any()
        else float("nan"),
        "false_alarms_per_1e5_events": float((zsw > proto.z_fire).mean() * 1e5 / w)
        if len(zsw)
        else float("nan"),
    }
    res["pass_a_diag_dominant"] = all(diag_rows.values())
    res["pass_b_macro_f1"] = bool(f1_plain >= proto.macro_f1_min)
    res["pass_c_delta_fire"] = bool(res["delta_fire_pooled"] <= proto.delta_fire_max)
    res["pass_all"] = bool(
        res["pass_a_diag_dominant"]
        and res["pass_b_macro_f1"]
        and res["pass_c_delta_fire"]
        and res["n_failed"] == 0
    )
    return res
