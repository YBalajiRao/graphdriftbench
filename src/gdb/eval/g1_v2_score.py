r"""Gate G1 protocol v2: read-only scoring of ``results/g1_v2/official/<preset>/``.

Scoring contract (frozen with the protocol; every rule below is implemented here and tested):

* **Source.** Only a directory whose parent is ``official`` and grandparent is ``g1_v2`` is read
  (``check_official_dir``). Quarantine, ``official_rerun``, ``dev_raw`` and v1 trees are refused.
* **Expected streams.** Per preset: ``alpha, beta, gamma, delta`` x seeds 3000-3019 and ``none`` x
  seeds 3200-3219 (``delta_mix`` x 3000-3019 is optional and secondary). A row that is missing is
  *added as a failed row* (``status="missing"``); a row with another preset, a filename that disagrees
  with its content, a duplicate (drift, seed), an unknown drift type or a seed outside the allocation
  raises ``ValueError`` (contamination is never scored around).
* **Failed rows.** ``status != "ok"`` or any non-finite z among the 12 first-three-window values.
  They stay in every count, are labelled ``fail`` in the confusion matrix and count as errors in
  macro-F1; a configuration with any failed/missing stream cannot pass.
* **Unit.** One stream = its first post-onset window z-vector ($z_\alpha,z_\beta,z_\gamma,z_\delta$ at
  window 0). The mean over windows 0-2 is reported as a labelled sensitivity only.
* **(a)** audit matrix = mean z over the ok streams of each injected type (4x4). Row $c$ passes iff its
  maximum is attained *uniquely* on the diagonal.
* **(b)** argmax label = the statistic with the largest z; an exact tie between maxima is the label
  ``tie`` (an error: no index-order tie-break exists). A failed/missing stream is the label ``fail``
  (an error). A stream whose maximum is $\le z_{\rm fire}$ still receives its argmax label in the
  gating macro-F1 (as in v1/Blueprint); the thresholded macro-F1 in which such streams are ``none``
  is reported as non-gating. Macro-F1 = mean of the four per-type F1 over the *expected* 80 streams.
* **(c)** $\delta$ fire rate = mean of $[z_\delta>3]$ over the ok pure $\alpha/\beta/\gamma$ streams.
* **Stationary streams** never enter (a)-(c); false alarm = $\max_c z_c>3$ at window 0; rate, per-1e5-event
  rate and per-statistic z mean/sd are reported.
* **Stationary stream-alarm count per 1e5 tested events** (not an online false-alarm rate or ARL: one known
  window is tested per stationary stream). Each *valid* stationary stream contributes exactly one
  tested window (its first post-onset window, $w$ events). With $A$ = number of valid stationary streams
  whose $\max_c z_c>3$ and $E=\sum_{
m valid}w_i$: rate $=A/E\cdot10^5$. Failed/missing stationary streams
  are not in $E$ (they have no tested window); they are counted, and the stream-level rate with failures
  treated as alarms is reported next to it.
* **Intervals.** 2000 percentile replicates (seed 0, deterministic; distinct from the 500 moving-block replicates that
  standardise each stream's statistics inside ``calibrate``). Drift metrics resample the 20 *seeds*
  with replacement and keep each drawn seed's expected alpha/beta/gamma/delta streams together
  (``_bootstrap``). The macro-F1 interval is failure-aware: a failed/missing stream stays an error in
  every replicate. The $\delta$ fire-rate, audit-matrix and stationary-rate intervals are *conditional on
  valid streams* (a failed stream has no z) and are labelled so; the number of failed/missing streams is
  reported beside them, ``delta_fire_upper_if_failed_pure_streams_fired`` bounds the fire rate, and no
  configuration with a failure passes whatever its interval. Stationary rate: resample the independent
  stationary streams. Reruns are never clusters; they are compared separately (``repeatability``).
* The primary argmax macro-F1 assesses *type classification* given known windows; it is not an
  operational online-detection score (the thresholded macro-F1 and the miss counts are non-gating).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from gdb.eval import g1

DRIFT_SEEDS = range(3000, 3020)
STATIONARY_SEEDS = range(3200, 3220)
EXPECTED = {
    **dict.fromkeys(g1.TYPES, DRIFT_SEEDS),
    "none": STATIONARY_SEEDS,
    "delta_mix": DRIFT_SEEDS,
}
OPTIONAL = {"delta_mix"}
GATING = ("small_bipartite", "wiki_like")
N_BOOT = 2000
ZCOLS = [f"z_{t}_w{m}" for m in range(g1.PROTO.n_post) for t in g1.TYPES]


def check_official_dir(d: Path, preset: str) -> Path:
    """Accept only ``.../g1_v2/official/<preset>``."""
    d = Path(d).resolve()
    if d.name != preset or d.parent.name != "official" or d.parent.parent.name != "g1_v2":
        raise ValueError(f"not an official v2 results directory for {preset}: {d}")
    return d


def load_rows(d: Path, preset: str) -> pd.DataFrame:
    """Validated rows (one per expected stream; missing ones added as ``status='missing'``)."""
    d = check_official_dir(d, preset)
    seen: dict[tuple[str, int], dict] = {}
    for f in sorted(d.glob("*.json")):
        m = re.fullmatch(r"([a-z_]+)_(\d+)", f.stem)
        if not m or m.group(1) not in EXPECTED:
            raise ValueError(f"unexpected file name {f.name}")
        key = (m.group(1), int(m.group(2)))
        if key[1] not in EXPECTED[key[0]]:
            raise ValueError(f"seed outside the allocation: {f.name}")
        row = json.loads(f.read_text())
        if (row.get("preset"), row.get("drift"), row.get("seed")) != (preset, key[0], key[1]):
            raise ValueError(f"{f.name}: content disagrees with preset/name (wrong preset or copy)")
        if key in seen:
            raise ValueError(f"duplicate row for {key}")
        seen[key] = row
    rows = []
    for drift, seeds in EXPECTED.items():
        for s in seeds:
            if (drift, s) in seen:
                rows.append(seen[(drift, s)])
            elif drift not in OPTIONAL:
                rows.append({"preset": preset, "drift": drift, "seed": s, "status": "missing"})
    df = pd.DataFrame(rows)
    for c in ZCOLS:
        if c not in df:
            df[c] = np.nan
    ok = (df.status == "ok") & np.isfinite(df[ZCOLS].to_numpy(dtype=float)).all(axis=1)
    df["valid"] = ok
    df.loc[~ok & (df.status == "ok"), "status"] = "nonfinite"
    return df


def _confusion(true: np.ndarray, pred: np.ndarray) -> dict:
    """Rows = injected type; columns = label, ``none`` (thresholded miss), ``tie``, ``fail`` (failed/missing)."""
    cols = [*g1.TYPES, "none", "tie", "fail"]
    return {p: {c: int(((true == c) & (pred == p)).sum()) for c in g1.TYPES} for p in cols}


def _bootstrap(drifted, lab, true, z0, valid, *, n_boot, seed):
    """Failure-aware seed-clustered percentile bootstrap (see module docstring, 'Intervals').

    All expected streams of a drawn seed are resampled together. A failed/missing stream keeps its
    ``fail`` label, so it remains an error in the macro-F1 replicate. The delta fire rate and the audit
    matrix are *conditional on valid streams* (failed streams have no z to average).
    """
    zf = g1.PROTO.z_fire
    seeds = np.array(sorted(drifted.seed.unique()))
    idx = {s: np.flatnonzero((drifted.seed == s).to_numpy()) for s in seeds}
    pure = np.isin(true, ["alpha", "beta", "gamma"])
    rng = np.random.default_rng(seed)
    f1s, fires, mats = [], [], []
    for _ in range(n_boot):
        ii = np.concatenate([idx[s] for s in rng.choice(seeds, len(seeds))])
        f1s.append(g1.macro_f1(true[ii], lab[ii])[0])
        pm = valid[ii] & pure[ii]
        fires.append(float((z0[ii][pm, 3] > zf).mean()) if pm.any() else np.nan)
        rows = []
        for c in g1.TYPES:
            m = valid[ii] & (true[ii] == c)
            rows.append(z0[ii][m].mean(axis=0) if m.any() else np.full(4, np.nan))
        mats.append(np.stack(rows))
    mats = np.stack(mats)
    q = lambda v: (float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975)))  # noqa: E731
    return {
        "macro_f1_ci95": q(f1s),
        "delta_fire_ci95": q(fires),
        "audit_lo": np.nanquantile(mats, 0.025, axis=0),
        "audit_hi": np.nanquantile(mats, 0.975, axis=0),
        "n_seed_clusters": len(seeds),
    }


def _ci(v) -> tuple[float, float]:
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def score_preset(df: pd.DataFrame, *, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """All gating and reporting quantities for one configuration (see module docstring)."""
    zf = g1.PROTO.z_fire
    drifted = df[df.drift.isin(g1.TYPES)].reset_index(drop=True)
    good = drifted[drifted.valid]
    stat = df[(df.drift == "none")]
    stat_ok = stat[stat.valid]
    n_failed = int((~drifted.valid).sum() + (~stat.valid).sum())

    z0 = drifted[[f"z_{t}_w0" for t in g1.TYPES]].to_numpy(dtype=float)
    lab = np.array(["fail"] * len(drifted), dtype=object)
    v = drifted.valid.to_numpy()
    lab[v] = g1.argmax_labels(z0[v])
    true = drifted.drift.to_numpy(dtype=object)
    f1, f1_by = g1.macro_f1(true, lab)
    thr = lab.copy()
    thr[v & (z0.max(axis=1) <= zf)] = "none"
    f1_thr, _ = g1.macro_f1(true, thr)

    mean = pd.DataFrame(index=g1.TYPES, columns=g1.TYPES, dtype=float)
    unique_diag, row_available = {}, {}
    n_valid_by_type = {c: int((good.drift == c).sum()) for c in g1.TYPES}
    n_failed_by_type = {c: int(((drifted.drift == c) & ~drifted.valid).sum()) for c in g1.TYPES}
    for c in g1.TYPES:
        zc = good[good.drift == c][[f"z_{t}_w0" for t in g1.TYPES]].to_numpy(dtype=float)
        row_available[c] = bool(len(zc))  # zero valid streams: row unavailable (NaN), never a mean
        mean.loc[c] = zc.mean(axis=0) if len(zc) else np.nan
        r = mean.loc[c].to_numpy(dtype=float)
        unique_diag[c] = bool(
            row_available[c]
            and np.isfinite(r).all()
            and (r == r.max()).sum() == 1
            and g1.TYPES[int(r.argmax())] == c
        )

    pure = good[good.drift.isin(["alpha", "beta", "gamma"])]
    fire = (pure.z_delta_w0 > zf).to_numpy()
    d_by = {
        c: float((pure[pure.drift == c].z_delta_w0 > zf).mean()) for c in ["alpha", "beta", "gamma"]
    }
    dfire = float(fire.mean()) if len(fire) else float("nan")

    cb = _bootstrap(drifted, lab, true, z0, v, n_boot=n_boot, seed=seed)
    z3 = good.copy()
    sens = {}
    if len(z3):  # conditional on valid streams
        zm = g1.zmat(z3, "mean3")
        lm = g1.argmax_labels(zm)
        sens = {
            "macro_f1_mean3": g1.macro_f1(z3.drift.to_numpy(dtype=object), lm)[0],
            "delta_fire_mean3": float((zm[~z3.drift.isin(["delta"]).to_numpy(), 3] > zf).mean()),
        }

    zs = stat_ok[[f"z_{t}_w0" for t in g1.TYPES]].to_numpy(dtype=float)
    fa = zs.max(axis=1) > zf if len(zs) else np.zeros(0, bool)
    rng = np.random.default_rng(seed)
    fa_ci = (
        _ci([fa[rng.integers(0, len(fa), len(fa))].mean() for _ in range(n_boot)])
        if len(fa)
        else (np.nan,) * 2
    )
    # events in the tested windows: each valid stationary stream contributes its first post-onset
    # window of w events; rate = sum(alarms) / sum(w) x 1e5 (failed streams are not in the denominator)
    ev_tested = float(stat_ok.w.sum()) if len(stat_ok) and "w" in stat_ok else 0.0
    n_pure_failed = int((~drifted.valid & drifted.drift.isin(["alpha", "beta", "gamma"])).sum())

    mix = df[(df.drift == "delta_mix") & df.valid]
    res = {
        "n_expected_drift": len(drifted),
        "n_expected_stationary": len(stat),
        "n_failed_or_missing": n_failed,
        "failed_streams": df[~df.valid & (df.drift != "delta_mix")][
            ["drift", "seed", "status"]
        ].to_dict("records"),
        "macro_f1_argmax": f1,
        "macro_f1_argmax_ci95_seed_clustered_failure_aware": cb["macro_f1_ci95"],
        "f1_by_type": f1_by,
        "macro_f1_thresholded_nongating": f1_thr,
        "audit_mean": mean.to_dict(),
        "audit_lo_conditional_on_valid": pd.DataFrame(
            cb["audit_lo"], index=g1.TYPES, columns=g1.TYPES
        ).to_dict(),
        "audit_hi_conditional_on_valid": pd.DataFrame(
            cb["audit_hi"], index=g1.TYPES, columns=g1.TYPES
        ).to_dict(),
        "audit_row_available": row_available,
        "valid_streams_by_type": n_valid_by_type,
        "failed_or_missing_streams_by_type": n_failed_by_type,
        "diagonal_unique_max_rows": unique_diag,
        "delta_fire_pooled": dfire,
        "delta_fire_n_pure_streams": int(len(pure)),
        "delta_fire_by_type": d_by,
        "delta_fire_ci95_seed_clustered_conditional_on_valid": cb["delta_fire_ci95"],
        "delta_fire_upper_if_failed_pure_streams_fired": float(
            (fire.sum() + n_pure_failed) / (len(fire) + n_pure_failed)
        )
        if len(fire) + n_pure_failed
        else float("nan"),
        "n_seed_clusters": cb["n_seed_clusters"],
        "bootstrap_note": "macro-F1 CI keeps failed streams as errors; delta-fire, audit and stationary intervals are conditional on valid streams; a pass is never granted when n_failed_or_missing > 0",
        "confusion_argmax": _confusion(true, lab),
        "confusion_thresholded_nongating": _confusion(true, thr),
        "missed_detections_nongating": int((thr == "none").sum()),
        "stationary_false_alarm_streams": int(fa.sum()),
        "stationary_false_alarm_rate": float(fa.mean()) if len(fa) else float("nan"),
        "stationary_false_alarm_rate_ci95_stream_bootstrap_conditional_on_valid": fa_ci,
        "stationary_alarm_definition": (
            "stationary stream-alarm count normalised by tested-window event volume: numerator A = number "
            "of valid stationary streams whose max_c z_c(first post-onset window) > 3; denominator E = "
            "sum of w over those valid streams (one tested window of w events each); value = A/E*1e5. "
            "One known window per stream is tested, so this is NOT an online detector false-alarm rate "
            "or ARL. Failed/missing stationary streams are excluded from E, counted, and block the pass."
        ),
        "stationary_valid_streams": int(len(fa)),
        "stationary_tested_events": ev_tested,
        "stationary_stream_alarms_per_1e5_tested_events": float(fa.sum() * 1e5 / ev_tested)
        if ev_tested
        else float("nan"),
        "stationary_false_alarm_rate_if_failed_streams_alarmed": float(
            (fa.sum() + (len(stat) - len(fa))) / len(stat)
        )
        if len(stat)
        else float("nan"),
        "stationary_z_mean_by_statistic": dict(zip(g1.TYPES, zs.mean(axis=0).tolist()))
        if len(zs)
        else {},
        "stationary_z_sd_by_statistic": dict(zip(g1.TYPES, zs.std(axis=0, ddof=1).tolist()))
        if len(zs) > 1
        else {},
        "sensitivity_first3_mean": sens,
        "delta_mix_nongating": {
            "n": int(len(mix)),
            "mean_z_delta_w0": float(mix.z_delta_w0.mean()) if len(mix) else None,
            "fire_rate": float((mix.z_delta_w0 > zf).mean()) if len(mix) else None,
        },
    }
    res["pass_a_diagonal_unique_max"] = bool(all(unique_diag.values()))
    res["pass_b_macro_f1"] = bool(f1 >= g1.PROTO.macro_f1_min)
    res["pass_c_delta_fire"] = bool(np.isfinite(dfire) and dfire <= g1.PROTO.delta_fire_max)
    res["pass_all"] = bool(
        res["pass_a_diagonal_unique_max"]
        and res["pass_b_macro_f1"]
        and res["pass_c_delta_fire"]
        and n_failed == 0
    )
    return res


def overall(summaries: dict[str, dict]) -> dict:
    """G1 v2 verdict: passes only if both gating configurations pass all three criteria."""
    if set(summaries) != set(GATING):
        raise ValueError("both gating configurations are required")
    for k, v in summaries.items():
        if v.get("n_expected_drift") != 80 or v.get("n_expected_stationary") != 20:
            raise ValueError(f"{k}: summary does not cover the expected 80 + 20 streams")
    return {
        "per_config_pass": {k: v["pass_all"] for k, v in summaries.items()},
        "g1_v2_passed": bool(all(v["pass_all"] for v in summaries.values())),
    }


def repeatability(first_dir: Path, rerun_dir: Path, preset: str) -> dict:
    """Diagnostic only: z differences and label flips between first-run rows and the rerun subset.

    Reads ``.../g1_v2/official/<preset>`` and ``.../g1_v2/official_rerun/<preset>``; the rerun rows are
    never merged into any score. Failed rows on either side are listed, not dropped.
    """
    first = check_official_dir(first_dir, preset)
    rerun = Path(rerun_dir).resolve()
    if rerun.name != preset or rerun.parent.name != "official_rerun":
        raise ValueError("rerun directory must be .../official_rerun/<preset>")
    out, flips, failed = [], 0, []
    for f in sorted(rerun.glob("*.json")):
        b = json.loads(f.read_text())
        g = first / f.name
        a = json.loads(g.read_text()) if g.exists() else None
        if a is None or a.get("status") != "ok" or b.get("status") != "ok":
            failed.append(f.name)
            continue
        za = np.array([a[f"z_{t}_w0"] for t in g1.TYPES])
        zb = np.array([b[f"z_{t}_w0"] for t in g1.TYPES])
        flip = bool(g1.argmax_labels(za[None])[0] != g1.argmax_labels(zb[None])[0])
        flips += flip
        out.append(
            {
                "stream": f.stem,
                "max_abs_dz": float(np.abs(za - zb).max()),
                "dz_delta": float(zb[3] - za[3]),
                "label_flip": flip,
            }
        )
    return {
        "n_pairs": len(out),
        "label_flips": flips,
        "not_comparable_or_failed": failed,
        "pairs": out,
    }
