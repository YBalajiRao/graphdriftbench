"""Protocol-v2 DEVELOPMENT runs: raw statistic series per stream (never official seeds).

    python scripts/dev_g1_raw.py --preset small_bipartite --seeds 4100-4109

Writes results/g1_v2/dev_raw/<preset>/<drift>_<seed>.json (resumable, never overwrites). Official seeds
(2000-2019, 2100-2119) are refused. Variants are calibrated offline by scripts/analyze_g1_dev.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from gdb.detect.window_stats import Window  # noqa: E402
from gdb.eval import g1, g1_raw  # noqa: E402
from gdb.models.tgn_prequential import prequential_event_losses  # noqa: E402
from gdb.synth.dsbm import DSBMConfig, generate_stream  # noqa: E402
from run_g1 import PRESET_GEOMETRY, parse_seeds, schedule  # noqa: E402

IDS_DIR = ROOT / "results" / "g1_v2" / "dev_ids"
CONSUMED = set(range(2000, 2020)) | set(range(2100, 2120)) | set(range(3100, 3120))


def run_one(preset: str, drift: str, seed: int) -> dict:
    w, warm = PRESET_GEOMETRY[preset]
    cfg0 = DSBMConfig.from_yaml(ROOT / "configs" / "synth" / f"{preset}.yaml")
    tau = 1.03 * (warm + g1.PROTO.r) * w / cfg0.base_rate
    cfg = replace(cfg0, seed=seed, t_end=tau + (g1.PROTO.n_post + 0.4) * w / cfg0.base_rate)
    s = generate_stream(cfg, schedule(drift, tau)).stream
    i_tau = int(np.searchsorted(s.t, tau))
    r, npost = g1.PROTO.r, g1.PROTO.n_post
    row = {"preset": preset, "drift": drift, "seed": seed, "w": w, "i_tau": i_tau}
    if i_tau < r * w or len(s) - i_tau < npost * w:
        return {**row, "status": "stream_too_short"}
    tg = prequential_event_losses(s, seed=seed)
    start = i_tau - r * w
    wins = [
        Window(
            s.src[start + m * w : start + (m + 1) * w],
            s.dst[start + m * w : start + (m + 1) * w],
            s.edge_feat[start + m * w : start + (m + 1) * w],
        )
        for m in range(r + npost)
    ]
    losses = g1.window_losses(tg["loss"], start, w, r + npost)
    ids = {f"src{m}": x.src for m, x in enumerate(wins)} | {
        f"dst{m}": x.dst for m, x in enumerate(wins)
    }
    np.savez_compressed(IDS_DIR / preset / f"{drift}_{seed}.npz", losses=losses, **ids)
    return {
        **row,
        "status": "ok",
        "raw": g1_raw.raw_case(wins, losses, seed=seed),
        "device": tg["device"],
        "peak_gb": tg["peak_gb"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--drifts", default="none,alpha,beta,gamma,delta")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    if CONSUMED & set(seeds):
        raise SystemExit("seeds overlap the consumed official allocations")
    out = ROOT / "results" / "g1_v2" / "dev_raw" / args.preset
    out.mkdir(parents=True, exist_ok=True)
    (IDS_DIR / args.preset).mkdir(parents=True, exist_ok=True)
    for drift in args.drifts.split(","):
        for seed in seeds:
            f = out / f"{drift}_{seed}.json"
            if f.exists():
                continue
            t0 = time.time()
            try:
                row = run_one(args.preset, drift, seed)
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
