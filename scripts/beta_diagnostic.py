"""EXPLORATORY beta-statistic comparison (not Gate G1, not the locked detector).

    python scripts/beta_diagnostic.py --preset small_bipartite

Prespecified analysis: candidate statistics are compared on the same window pairs as the purity
panel, with a null band (q95 of N stationary streams) per candidate. Exploratory seeds 7000..7019
for injections and 8000..8000+n_null for the null band; both are disjoint from the purity seeds
(0-19), the purity null seeds (1000+), and the earlier exploration seeds (5000+/6000+, 100-199).
eps = 0.2 and the preset window size are unchanged. Output: results/exploratory/beta_diag_<preset>.csv
"""

from __future__ import annotations

import argparse
import sys
import time
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
CAND = ["dos_w1", "sv_w1", "hhi_dst_abs", "hhi_src_abs"]
VARS = [
    ("none", "none", {}),
    ("beta[hub]", "beta", {"target": "hub", "boost": 10.0}),
    ("beta[merge]", "beta", {"target": "merge"}),
    ("beta[densify]", "beta", {"target": "densify", "power": 4.0}),
    ("alpha", "alpha", {}),
    ("gamma[src]", "gamma", {}),
    ("delta", "delta", {}),
]


def one(cfg, w, seed, sched):
    c = replace(cfg, seed=seed, t_end=P._horizon(cfg, w))
    res = generate_stream(c, sched)
    i = int(np.searchsorted(res.stream.t, c.t_end / 2))
    s = res.stream
    b, a = slice(i - w, i), slice(i, i + w)
    out, cost = {}, {}
    t0 = time.perf_counter()
    db = gs.window_dos(
        s.src[b], s.dst[b], n_moments=100, n_probes=200, rng=np.random.default_rng(2 * seed)
    )
    da = gs.window_dos(
        s.src[a], s.dst[a], n_moments=100, n_probes=200, rng=np.random.default_rng(2 * seed + 1)
    )
    out["dos_w1"] = gs.dos_w1(db, da)
    cost["dos_w1"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    out["sv_w1"] = gs.sv_w1(s.src[b], s.dst[b], s.src[a], s.dst[a])
    cost["sv_w1"] = time.perf_counter() - t0
    out["hhi_dst_abs"] = abs(gs.concentration_index(s.dst[a]) - gs.concentration_index(s.dst[b]))
    out["hhi_src_abs"] = abs(gs.concentration_index(s.src[a]) - gs.concentration_index(s.src[b]))
    if not all(np.isfinite(list(out.values()))):
        raise RuntimeError(f"non-finite candidate statistic, seed {seed}")
    return out, cost


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--eps", type=float, default=0.2)
    args = ap.parse_args()
    cfg = DSBMConfig.from_yaml(ROOT / "configs" / "synth" / f"{args.preset}.yaml")
    w = cfg.window_events
    null, costs = [], []
    for k in range(args.n_null):
        r, c = one(cfg, w, NULL0 + k, None)
        null.append(r), costs.append(c)
    nd = pd.DataFrame(null)
    band = nd.quantile(0.95)
    rows = []
    for label, typ, params in VARS:
        for s in range(TEST0, TEST0 + args.n_seeds):
            sched = (
                None if typ == "none" else abrupt(typ, P._horizon(cfg, w) / 2, args.eps, **params)
            )
            r, _ = one(cfg, w, s, sched)
            rows.append({"variant": label, "seed": s, **r})
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "exploratory"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"beta_diag_{args.preset}.csv", index=False)
    nd.to_csv(out / f"beta_diag_null_{args.preset}.csv", index=False)
    cdf = pd.DataFrame(costs).mean()
    print(f"preset={args.preset} w={w} eps={args.eps} n_null={args.n_null} n_seeds={args.n_seeds}")
    print("mean seconds per window pair:", {k: round(v, 3) for k, v in cdf.items()})

    def frac(x: pd.Series, thr: float) -> str:
        return f"{int((x > thr).sum())}/{len(x)}"

    tab = pd.DataFrame(
        {c: df.groupby("variant", sort=False)[c].apply(frac, thr=band[c]) for c in CAND}
    )
    print(tab.to_string())
    tab.to_csv(out / f"beta_diag_power_{args.preset}.csv")


if __name__ == "__main__":
    main()
