"""Score a directory of G1 stream rows (results/g1/<mode>/<preset>/*.json) -> summary files.

    python scripts/score_g1.py --mode official --preset small_bipartite

Writes results/g1/<mode>/<preset>_summary.json (+ .md). Never overwrites an existing summary
unless --force-dev is given in dev mode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from gdb.eval import g1  # noqa: E402


def load_rows(d: Path) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
    return pd.DataFrame(rows)


def md_frame(df: pd.DataFrame, nd: int = 2) -> str:
    cols = list(df.columns)
    lines = ["| | " + " | ".join(map(str, cols)) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        cells = [f"{v:.{nd}f}" if isinstance(v, float) else str(v) for v in r]
        lines.append(f"| {i} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--force-dev", action="store_true")
    args = ap.parse_args()
    d = ROOT / "results" / "g1" / args.mode / args.preset
    df = load_rows(d)
    if "status" not in df:
        raise SystemExit("no rows")
    w = int(df["w"].dropna().iloc[0])
    out_json = ROOT / "results" / "g1" / args.mode / f"{args.preset}_summary.json"
    if out_json.exists() and not (args.mode == "dev" and args.force_dev):
        raise SystemExit(f"{out_json} exists; refusing to overwrite")
    parts, summary = [], {}
    for label, m in (
        ("primary_first_post_window", 0),
        ("sensitivity_mean_first3_windows", "mean3"),
    ):
        r = g1.score_g1(df, m, w=w)
        summary[label] = {
            k: (v.to_dict() if isinstance(v, pd.DataFrame) else v) for k, v in r.items()
        }
        parts.append(f"## {args.preset} — {label} (mode={args.mode})\n")
        parts.append(
            f"streams {r['n_streams']} (failed {r['n_failed']}); macro-F1 argmax {r['macro_f1_argmax']:.3f} (95% bootstrap CI {r['macro_f1_argmax_ci95'][0]:.2f}-{r['macro_f1_argmax_ci95'][1]:.2f}) "
            f"(thresholded z>{g1.PROTO.z_fire}: {r['macro_f1_thresholded']:.3f}); per-type F1 "
            f"{ {k: round(v, 2) for k, v in r['f1_by_type'].items()} }\n"
        )
        parts.append("Audit matrix: mean z of statistic (columns) given injected type (rows)\n")
        parts.append(md_frame(r["audit_mean"]) + "\n")
        parts.append("95% bootstrap CI low / high\n")
        parts.append(md_frame(r["audit_lo"]) + "\n\n" + md_frame(r["audit_hi"]) + "\n")
        parts.append("Confusion (argmax; rows = true)\n")
        parts.append(md_frame(r["confusion_argmax"], 0) + "\n")
        parts.append("Confusion (thresholded; 'none' = missed detection)\n")
        parts.append(md_frame(r["confusion_thresholded"], 0) + "\n")
        parts.append(
            f"Criteria — (a) diagonal-dominant rows {r['diag_dominant_rows']} -> {r['pass_a_diag_dominant']}; "
            f"(b) macro-F1 >= {g1.PROTO.macro_f1_min}: {r['pass_b_macro_f1']}; "
            f"(c) delta fire on pure a/b/g {r['delta_fire_pooled']:.3f} (by type {r['delta_fire_by_type']}) "
            f"<= {g1.PROTO.delta_fire_max}: {r['pass_c_delta_fire']}; **pass_all = {r['pass_all']}**\n"
        )
        parts.append(
            f"Detection (z>{g1.PROTO.z_fire}): precision {r['detection_precision']:.3f}, recall "
            f"{r['detection_recall']:.3f}, missed {r['missed']}, stationary false-alarm streams "
            f"{r['false_alarm_streams']}, first-detection window counts {r['detected_first_in_window']}, "
            f"missed within 3 windows {r['missed_within_n_post']}, mean delay (windows) "
            f"{r['mean_delay_windows_detected']:.2f}, false alarms per 1e5 events "
            f"{r['false_alarms_per_1e5_events']:.2f}; delta-statistic AUROC (delta vs all others incl. "
            f"stationary) {r['delta_auroc_vs_rest']:.3f}\n"
        )
    out_json.write_text(json.dumps(summary, indent=1, default=float))
    out_json.with_suffix(".md").write_text("\n".join(parts), encoding="utf-8")
    print("\n".join(parts))


if __name__ == "__main__":
    main()
