"""Offline comparison of protocol-v2 variants on DEVELOPMENT raw series (results/g1_v2/dev_raw).

    python scripts/analyze_g1_dev.py --preset small_bipartite

Prints one row per variant and writes results/g1_v2/dev_variants_<preset>.csv (refuses to overwrite).
Development output only: nothing here is an official G1 result and no threshold is tuned here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from gdb.eval import g1, g1_raw  # noqa: E402
from run_g1 import PRESET_GEOMETRY  # noqa: E402

VARIANTS = {
    "V0 v1 (dos, huber, floor1)": {"beta": "dos", "delta": "v1"},
    "V1b l2 + v1 delta": {"beta": "l2", "delta": "v1"},
    "V1 strata + v1 delta": {"beta": "strata", "delta": "v1"},
    "V2 dos + bounded delta": {"beta": "dos", "delta": "bounded"},
    "V3 strata + bounded": {"beta": "strata", "delta": "bounded"},
    "V4 strata + median-loss": {"beta": "strata", "delta": "none"},
    "V5 strata + bounded + gate": {"beta": "strata", "delta": "bounded", "gate": True},
    "V6 strata + median-loss + gate": {"beta": "strata", "delta": "none", "gate": True},
    "V3b l2 + bounded": {"beta": "l2", "delta": "bounded"},
    "V4b l2 + median-loss": {"beta": "l2", "delta": "none"},
    "V5b l2 + bounded + gate": {"beta": "l2", "delta": "bounded", "gate": True},
    "V6b l2 + median-loss + gate": {"beta": "l2", "delta": "none", "gate": True},
    "V0 floor x0.5": {"beta": "dos", "delta": "v1", "floor_mult": 0.5},
    "V0 floor x2": {"beta": "dos", "delta": "v1", "floor_mult": 2.0},
    "V3b floor x0.5": {"beta": "l2", "delta": "bounded", "floor_mult": 0.5},
    "V3b floor x2": {"beta": "l2", "delta": "bounded", "floor_mult": 2.0},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    args = ap.parse_args()
    d = ROOT / "results" / "g1_v2" / "dev_raw" / args.preset
    rows = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"{args.preset}: {len(ok)} ok / {len(rows)} streams")
    w = PRESET_GEOMETRY[args.preset][0]
    out = []
    for name, kw in VARIANTS.items():
        recs = [
            {
                "drift": r["drift"],
                "seed": r["seed"],
                **g1_raw.calibrate(r["raw"], seed=r["seed"], **kw),
            }
            for r in ok
        ]
        df = pd.DataFrame(recs)
        s = g1.score_g1(df, 0, w=w)
        am = s["audit_mean"]
        g = df[df.drift == "gamma"]
        out.append(
            {
                "variant": name,
                "macro_f1": round(s["macro_f1_argmax"], 3),
                "diag_rows": sum(s["diag_dominant_rows"].values()),
                "delta_fire_pooled": round(s["delta_fire_pooled"], 3),
                **{f"dfire_{k}": round(v, 2) for k, v in s["delta_fire_by_type"].items()},
                "z_b|beta": round(am.loc["beta", "beta"], 2),
                "z_b|gamma": round(am.loc["gamma", "beta"], 2),
                "z_d|delta": round(am.loc["delta", "delta"], 2),
                "z_d|gamma_min": round(g["z_delta_w0"].min(), 1),
                "z_d|gamma_max": round(g["z_delta_w0"].max(), 1),
                "delta_auroc": round(s["delta_auroc_vs_rest"], 3),
                "recall_z>3": round(s["detection_recall"], 3),
                "false_alarm_streams": s["false_alarm_streams"],
            }
        )
    res = pd.DataFrame(out)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(res.to_string(index=False))
    f = ROOT / "results" / "g1_v2" / f"dev_variants_{args.preset}.csv"
    if f.exists():
        print(f"{f} exists; not overwriting")
    else:
        res.to_csv(f, index=False)


if __name__ == "__main__":
    main()
