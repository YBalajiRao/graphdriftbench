r"""Generator/injector purity checks against a stationary null band.

A *purity* test asks whether an injection changes only its own factor. For one seed we generate a
stream with one injection at $\tau$, take the $w$ events just before and just after $\tau$, and
compute a panel of window-pair statistics (all from :mod:`gdb.utils.graph_stats`, plus the
ground-truth block-pair counts). Each statistic is compared with a **null band**: its 95th
percentile over many *stationary* streams of the same preset and window size.

Because each seed is a random test (about 1 in 20 seeds leaves the band by chance):

* a "should stay null" cell passes when ``>= NULL_MIN_INSIDE`` of ``N_SEEDS`` seeds stay inside the band;
* a "should shift" cell passes when ``>= SHIFT_MIN_DETECT`` of ``N_SEEDS`` seeds exceed the band.

Thresholds, $\varepsilon$ and the window size are fixed inputs; they must not be tuned to pass.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from gdb.synth.dsbm import DSBMConfig, SynthResult, block_pair_counts, generate_stream
from gdb.synth.injectors import ALPHA, BETA, DELTA, GAMMA, DriftSchedule
from gdb.synth.schedules import abrupt
from gdb.utils import graph_stats as gs

N_SEEDS = 20
NULL_MIN_INSIDE = 17
SHIFT_MIN_DETECT = 18
BAND_QUANTILE = 0.95
NULL_SEED0 = 1000  # null-band streams use seeds 1000.. ; test seeds are 0..19

STATS = [
    "deg_ks_src",
    "deg_ks_dst",
    "dos_w1",
    "mark_mmd",
    "blockpair_chi2",
    "new_node_rate_src",
    "new_node_rate_dst",
]

_ALL_NULL = {s: "null" for s in STATS}
EXPECTED: dict[str, dict[str, str]] = {
    "none": dict(_ALL_NULL),
    ALPHA: {**_ALL_NULL, "mark_mmd": "shift"},
    BETA: {
        **_ALL_NULL,
        "dos_w1": "shift",
        "deg_ks_src": "info",
        "deg_ks_dst": "info",
        "blockpair_chi2": "info",
    },
    GAMMA: {**_ALL_NULL, "new_node_rate_src": "shift"},
    DELTA: {**_ALL_NULL, "blockpair_chi2": "shift"},
}


# label, drift type, params, primary (counts toward the verdict), expectation overrides
VARIANTS = [
    ("none", "none", {}, True, None),
    ("alpha", "alpha", {}, True, None),
    ("beta[hub]", "beta", {"target": "hub", "boost": 10.0}, True, None),
    ("beta[merge]", "beta", {"target": "merge"}, False, None),
    ("beta[densify]", "beta", {"target": "densify", "power": 4.0}, False, None),
    ("gamma[src]", "gamma", {}, True, None),
    (
        "gamma[both]",
        "gamma",
        {"side": "both"},
        False,
        {**EXPECTED["gamma"], "new_node_rate_dst": "shift"},
    ),
    ("delta", "delta", {}, True, None),
    (
        "delta[+flip]",
        "delta",
        {"flip_homophily": True},
        False,
        {**EXPECTED["delta"], "deg_ks_src": "info", "deg_ks_dst": "info", "dos_w1": "info"},
    ),
]


def window_pair_stats(
    res: SynthResult,
    i_split: int,
    w: int,
    kpm_seed: int,
    *,
    n_moments: int = 100,
    n_probes: int = 200,
) -> dict[str, float]:
    """Statistics between the windows ``[i_split-w, i_split)`` (reference) and ``[i_split, i_split+w)``."""
    s = res.stream
    b, a = slice(i_split - w, i_split), slice(i_split, i_split + w)
    src_b, dst_b, src_a, dst_a = s.src[b], s.dst[b], s.src[a], s.dst[a]
    cfg = res.cfg

    def cnt(sl_s: np.ndarray, sl_d: np.ndarray) -> np.ndarray:
        return block_pair_counts(sl_s, sl_d, res.node_block, cfg.k_src, cfg.k_dst).astype(float)

    c_b, c_a = cnt(src_b, dst_b), cnt(src_a, dst_a)
    mask = (c_b + c_a) > 0
    chi2 = float((((c_b - c_a) ** 2)[mask] / (c_b + c_a)[mask]).sum())
    dos_b = gs.window_dos(
        src_b,
        dst_b,
        n_moments=n_moments,
        n_probes=n_probes,
        rng=np.random.default_rng(2 * kpm_seed),
    )
    dos_a = gs.window_dos(
        src_a,
        dst_a,
        n_moments=n_moments,
        n_probes=n_probes,
        rng=np.random.default_rng(2 * kpm_seed + 1),
    )
    marks_b, marks_a = s.edge_feat[b], s.edge_feat[a]
    return {
        "deg_ks_src": gs.degree_ks(gs.node_degrees(src_b), gs.node_degrees(src_a)),
        "deg_ks_dst": gs.degree_ks(gs.node_degrees(dst_b), gs.node_degrees(dst_a)),
        "dos_w1": gs.dos_w1(dos_b, dos_a),
        "mark_mmd": gs.linear_mmd2_rbf(marks_b, marks_a, bandwidth=gs.median_bandwidth(marks_b)),
        "blockpair_chi2": chi2,
        "new_node_rate_src": gs.new_node_rate(src_a, src_b),
        "new_node_rate_dst": gs.new_node_rate(dst_a, dst_b),
    }


def _horizon(cfg: DSBMConfig, w: int) -> float:
    """Horizon giving about ``2.5 w`` events on each side of the midpoint."""
    return 5.0 * w / cfg.base_rate


def _run_one(
    cfg: DSBMConfig, w: int, seed: int, schedule: DriftSchedule | None, **kw
) -> dict[str, float]:
    c = replace(cfg, seed=seed, t_end=_horizon(cfg, w))
    res = generate_stream(c, schedule)
    tau = c.t_end / 2
    i_split = int(np.searchsorted(res.stream.t, tau))
    if i_split < w or len(res.stream) - i_split < w:
        raise RuntimeError("stream too short for the requested window size")
    stats = window_pair_stats(res, i_split, w, kpm_seed=seed, **kw)
    bad = [k for k, v in stats.items() if not np.isfinite(v)]
    if (
        bad
    ):  # a NaN would count as "outside the band" and make shift cells pass for the wrong reason
        raise RuntimeError(
            f"non-finite statistics {bad} (seed {seed}); window too small or degenerate"
        )
    return stats


def build_null_band(cfg: DSBMConfig, w: int, n_null: int = 200, **kw) -> dict:
    """Stationary null distribution of every statistic and its 95th percentile (the band)."""
    rows = [_run_one(cfg, w, NULL_SEED0 + i, None, **kw) for i in range(n_null)]
    df = pd.DataFrame(rows)
    return {
        "window_events": w,
        "n_null": n_null,
        "quantile": BAND_QUANTILE,
        "config": cfg.to_dict(),
        "band": {s: float(df[s].quantile(BAND_QUANTILE)) for s in STATS},
        "null_mean": {s: float(df[s].mean()) for s in STATS},
        "null_sd": {s: float(df[s].std()) for s in STATS},
    }


def save_band(band: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(band, indent=2))


def load_band(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def run_injection(
    cfg: DSBMConfig,
    w: int,
    drift_type: str,
    eps: float,
    seeds: range | list[int],
    stat_kw: dict | None = None,
    **params,
) -> pd.DataFrame:
    """Statistics for one drift type over ``seeds`` (``drift_type='none'`` = stationary sanity row)."""
    rows = []
    for s in seeds:
        sched = None
        if drift_type != "none":
            sched = abrupt(drift_type, _horizon(cfg, w) / 2, eps, **params)
        rows.append(
            {"type": drift_type, "seed": s, **_run_one(cfg, w, s, sched, **(stat_kw or {}))}
        )
    return pd.DataFrame(rows)


def pass_rates(
    df: pd.DataFrame,
    band: dict,
    drift_type: str,
    label: str | None = None,
    expected: dict | None = None,
) -> pd.DataFrame:
    """Per-statistic pass-rate rows for one drift type (see module docstring for the rules)."""
    n = len(df)
    out = []
    for stat in STATS:
        inside = int((df[stat] <= band["band"][stat]).sum())
        expect = (expected or EXPECTED[drift_type])[stat]
        if expect == "null":
            ok = inside >= -(-NULL_MIN_INSIDE * n // N_SEEDS)
            count = inside
        elif expect == "shift":
            ok = (n - inside) >= -(-SHIFT_MIN_DETECT * n // N_SEEDS)
            count = n - inside
        else:
            ok, count = None, n - inside
        out.append(
            {
                "type": label or drift_type,
                "statistic": stat,
                "expectation": expect,
                "n_seeds": n,
                "n_inside_band": inside,
                "n_outside_band": n - inside,
                "n_counted": count,
                "pass": ok,
                "band_q95": band["band"][stat],
                "median_value": float(df[stat].median()),
            }
        )
    return pd.DataFrame(out)
