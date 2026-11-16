"""Gate G1 protocol v2 OFFICIAL runner (frozen seeds, new output tree, never overwrites).

    python scripts/run_g1_v2.py --preset small_bipartite --drifts none,alpha,beta,gamma,delta,delta_mix
    python scripts/run_g1_v2.py --preset small_bipartite --tag rerun   # reproducibility subset only

Official rows: results/g1_v2/official/<preset>/<drift>_<seed>.json (raw series + the frozen variant's
z-scores: beta=l2, delta=bounded, no gate, floor x1). The v1 tree results/g1/ is never touched.
Seeds: drifts 3000-3019, stationary 3200-3219 (3100-3119 were run before approval and are quarantined); ``--tag rerun`` writes seeds 3000-3004 to
results/g1_v2/official_rerun/ for run-to-run (CUDA) variability only; it never replaces a row.
A stream that fails is written as a failed row. No threshold or seed is configurable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import dev_g1_raw  # noqa: E402
from gdb.eval import g1_raw  # noqa: E402

SEEDS = {"drift": range(3000, 3020), "none": range(3200, 3220)}
# executed before protocol approval (quarantined, never official): see results/g1_v2/quarantine_*
QUARANTINED_SEEDS = set(range(3100, 3120))
FROZEN = {"beta": "l2", "delta": "bounded", "gate": False, "floor_mult": 1.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--drifts", default="none,alpha,beta,gamma,delta")
    ap.add_argument("--tag", choices=["official", "rerun"], default="official")
    ap.add_argument(
        "--out-root", help="DRY CHECK ONLY: scratch directory outside results/g1*; needs --seeds"
    )
    ap.add_argument(
        "--seeds", help="DRY CHECK ONLY (with --out-root): non-official dev seeds, e.g. 4200"
    )
    args = ap.parse_args()
    if bool(args.out_root) != bool(args.seeds):
        raise SystemExit("--out-root and --seeds are only valid together (dry check)")
    if args.out_root:
        root = Path(args.out_root).resolve()
        if root == ROOT / "results" or (ROOT / "results") in root.parents:
            raise SystemExit("dry-check output must be outside results/")
        dry_seeds = [int(x) for x in args.seeds.split(",")]
        frozen = {s for r in SEEDS.values() for s in r} | dev_g1_raw.CONSUMED | QUARANTINED_SEEDS
        if frozen & set(dry_seeds):
            raise SystemExit("dry-check seeds must not be official or consumed seeds")
    else:
        root = (
            ROOT
            / "results"
            / "g1_v2"
            / ("official" if args.tag == "official" else "official_rerun")
        )
        dry_seeds = None
    if QUARANTINED_SEEDS & {s for r in SEEDS.values() for s in r}:
        raise SystemExit("official seeds overlap quarantined seeds")
    out = root / args.preset
    out.mkdir(parents=True, exist_ok=True)
    dev_g1_raw.IDS_DIR = root / "ids"
    (dev_g1_raw.IDS_DIR / args.preset).mkdir(parents=True, exist_ok=True)
    for drift in args.drifts.split(","):
        seeds = list(SEEDS["none" if drift == "none" else "drift"])
        if dry_seeds is not None:
            seeds = dry_seeds
        elif args.tag == "rerun":
            seeds = [
                s
                for s in seeds
                if s in (3000, 3001, 3002, 3003, 3004, 3200, 3201, 3202, 3203, 3204)
            ]
        for seed in seeds:
            f = out / f"{drift}_{seed}.json"
            if f.exists():
                continue
            t0 = time.time()
            try:
                row = dev_g1_raw.run_one(args.preset, drift, seed)
                if row.get("status") == "ok":
                    row.update(g1_raw.calibrate(row["raw"], seed=seed, **FROZEN))
            except Exception as e:  # recorded, never dropped
                row = {
                    "preset": args.preset,
                    "drift": drift,
                    "seed": seed,
                    "status": f"error:{type(e).__name__}:{e}",
                    "trace": traceback.format_exc()[-1500:],
                }
            row["wall_seconds"] = time.time() - t0
            f.write_text(json.dumps(row, default=float))
            print(
                args.preset,
                drift,
                seed,
                row.get("status"),
                f"{row['wall_seconds']:.0f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
