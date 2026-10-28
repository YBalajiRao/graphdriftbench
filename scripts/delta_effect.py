"""EXPLORATORY: effect of the delta variants on every available statistic (not Gate G1).

    python scripts/delta_effect.py --preset small_bipartite

Variants: stationary, delta[permute] (primary, full permutation), delta[permute+flip], delta[mix]
(secondary, eps = 0.2) and beta[hub] (to check whether delta is distinguishable from beta).
Statistics: the purity panel (degree KS src/dst, DoS W1, linear MMD, block-pair chi2, new-node
rates) plus sv_w1 and the destination/source concentration change. Null band = q95 of stationary
streams (seeds 8000+); injection seeds 7000..7019: exploratory, disjoint from purity and G1 seeds.
Output: results/exploratory/delta_effect_<preset>.csv (+ _power.csv)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gdb.synth import purity as P  # noqa: E402
from gdb.synth.dsbm import DSBMConfig, generate_stream  # noqa: E402
from gdb.synth.schedules import abrupt  # noqa: E402
from gdb.utils import graph_stats as gs  # noqa: E402

TEST0, NULL0 = 7000, 8000
VARS = [
    ("none", "none", 0.0, {}),
    ("delta[permute]", "delta", 1.0, {"mode": "permute"}),
    ("delta[permute+flip]", "delta", 1.0, {"mode": "permute", "flip_homophily": True}),
    ("delta[mix,0.2]", "delta", 0.2, {"mode": "mix"}),
    ("beta[hub,0.2]", "beta", 0.2, {"target": "hub", "boost": 10.0}),
]
COLS = [*P.STATS, "sv_w1", "hhi_dst_abs", "hhi_src_abs"]


def one(cfg: DSBMConfig, w: int, seed: int, sched) -> dict[str, float]:
    c = replace(cfg, seed=seed, t_end=P._horizon(cfg, w))
    res = generate_stream(c, sched)
    i = int(np.searchsorted(res.stream.t, c.t_end / 2))
    out = P.window_pair_stats(res, i, w, kpm_seed=seed)
    s = res.stream
    b, a = slice(i - w, i), slice(i, i + w)
    out["sv_w1"] = gs.sv_w1(s.src[b], s.dst[b], s.src[a], s.dst[a])
    out["hhi_dst_abs"] = abs(gs.concentration_index(s.dst[a]) - gs.concentration_index(s.dst[b]))
    out["hhi_src_abs"] = abs(gs.concentration_index(s.src[a]) - gs.concentration_index(s.src[b]))
    bad = [k for k, v in out.items() if not np.isfinite(v)]
    if bad:
        raise RuntimeError(f"non-finite statistics {bad} at seed {seed}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--n-seeds", type=int, default=20)
    args = ap.parse_args()
    cfg = DSBMConfig.from_yaml(ROOT / "configs" / "synth" / f"{args.preset}.yaml")
    w = cfg.window_events
    null = pd.DataFrame([one(cfg, w, NULL0 + k, None) for k in range(args.n_null)])
    band = null.quantile(0.95)
    rows = []
    for label, typ, eps, params in VARS:
        for s in range(TEST0, TEST0 + args.n_seeds):
            sched = None if typ == "none" else abrupt(typ, P._horizon(cfg, w) / 2, eps, **params)
            rows.append({"variant": label, "seed": s, **one(cfg, w, s, sched)})
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "exploratory"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"delta_effect_{args.preset}.csv", index=False)
    null.to_csv(out / f"delta_effect_null_{args.preset}.csv", index=False)

    def frac(x: pd.Series, thr: float) -> str:
        return f"{int((x > thr).sum())}/{len(x)}"

    tab = pd.DataFrame(
        {c: df.groupby("variant", sort=False)[c].apply(frac, thr=band[c]) for c in COLS}
    )
    tab.to_csv(out / f"delta_effect_power_{args.preset}.csv")
    print(
        f"preset={args.preset} w={w} n_null={args.n_null} n_seeds={args.n_seeds} (seeds outside q95 band)"
    )
    print(tab.to_string())


if __name__ == "__main__":
    main()
