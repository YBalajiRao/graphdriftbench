"""Build the stationary null band and the per-type purity pass-rate table for one preset.

    python scripts/purity_report.py --preset small_bipartite
    python scripts/purity_report.py --preset wiki_like --n-null 200

Writes results/null_band_<preset>.json, results/purity_raw_<preset>.csv (per-seed statistics),
results/purity_<preset>.csv (pass-rate table) and results/purity_<preset>.md. Test seeds are
0..19, null-band seeds are 1000.., both fixed in gdb.synth.purity. Thresholds, eps and the window
size are inputs; do not tune them to make a check pass.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from gdb.synth import purity as P  # noqa: E402
from gdb.synth.dsbm import DSBMConfig  # noqa: E402

warnings.filterwarnings("ignore", message="ks_2samp")


def to_md(df: pd.DataFrame) -> str:
    """Minimal GitHub-markdown table (avoids the optional `tabulate` dependency)."""
    head = "| type | " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    rows = [
        "| " + str(i) + " | " + " | ".join(str(v) for v in r) + " |"
        for i, r in zip(df.index, df.values)
    ]
    return "\n".join([head, sep, *rows])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--n-seeds", type=int, default=P.N_SEEDS)
    args = ap.parse_args()

    cfg = DSBMConfig.from_yaml(ROOT / "configs" / "synth" / f"{args.preset}.yaml")
    w = cfg.window_events
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    t0 = time.time()
    band = P.build_null_band(cfg, w, n_null=args.n_null)
    P.save_band(band, out / f"null_band_{args.preset}.json")
    print(f"null band: {args.n_null} streams, w={w}, {time.time() - t0:.0f}s")

    raw, tables = [], []
    for label, ty, params, primary, expected in P.VARIANTS:
        if cfg.unipartite and label == "gamma[both]":
            continue
        if cfg.unipartite and ty == "gamma":  # one id space: fresh ids also appear as destinations
            expected = {**P.EXPECTED["gamma"], "new_node_rate_dst": "shift"}
        df = P.run_injection(cfg, w, ty, args.eps, range(args.n_seeds), **params)
        df["label"] = label
        raw.append(df)
        t = P.pass_rates(df, band, ty, label=label, expected=expected)
        t["primary"] = primary
        tables.append(t)
        print(f"  {label}: done ({time.time() - t0:.0f}s)")
    pd.concat(raw).to_csv(out / f"purity_raw_{args.preset}.csv", index=False)
    table = pd.concat(tables)
    table.to_csv(out / f"purity_{args.preset}.csv", index=False)

    cell = table.assign(
        cell=lambda d: d.apply(
            lambda r: (
                f"{r.n_counted}/{r.n_seeds}"
                + ("" if r.expectation == "info" else ("✓" if r["pass"] else "✗"))
            ),
            axis=1,
        )
    ).pivot(index="type", columns="statistic", values="cell")
    order = [lbl for lbl, *_ in P.VARIANTS if lbl in cell.index]
    md = (
        f"Preset `{args.preset}`, w={w}, eps={args.eps}, {args.n_seeds} seeds, null band q95 of {args.n_null} streams.\n"
        "Cells: seeds counted / total. Null cells count seeds INSIDE the band (pass >= 17/20); "
        "shift cells count seeds OUTSIDE (pass >= 18/20); info cells have no requirement.\n\n"
        + to_md(cell.loc[order, P.STATS])
        + "\n"
    )
    (out / f"purity_{args.preset}.md").write_text(md, encoding="utf-8")
    print(md)
    prim = table[table.primary & table["pass"].notna()]
    sec = table[~table.primary & table["pass"].notna() & ~table["pass"].astype(bool)]
    for r in sec.itertuples():
        print(f"  non-primary check failed: {r.type} / {r.statistic} ({r.n_counted}/{r.n_seeds})")
    print(f"primary checks failed: {int((~prim['pass'].astype(bool)).sum())} of {len(prim)}")


if __name__ == "__main__":
    main()
