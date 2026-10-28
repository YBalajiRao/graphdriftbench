"""Run Gate G1 streams (dev or official) and write one JSON row per stream. Never overwrites.

    python scripts/run_g1.py --mode dev      --preset small_bipartite --seeds 4000-4004
    python scripts/run_g1.py --mode official --preset small_bipartite

Protocol: docs/progress/g1-protocol.md (locked before any official run). A row already on disk is
skipped (resume); a row is never rewritten. Failures are written as rows with status != "ok".
The learner side (TGN prequential loss, window statistics) only sees the stream through
StreamReader / window arrays; the drift type and onset are used by this harness to build the
stream, to choose which windows are the reference and test windows, and (later) to score.
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

import numpy as np  # noqa: E402

from gdb.detect.window_stats import Window  # noqa: E402
from gdb.eval import g1  # noqa: E402
from gdb.models.tgn_prequential import prequential_event_losses  # noqa: E402
from gdb.synth import schedules as sch  # noqa: E402
from gdb.synth.dsbm import DSBMConfig, generate_stream  # noqa: E402

# preset -> (window events w, warm-up windows before the 10 reference windows)
PRESET_GEOMETRY = {
    "small_bipartite": (2000, 30),
    "g2_learnable": (2000, 30),
    "wiki_like": (5000, 30),
}
EPS = 0.2
OFFICIAL_SEEDS = {"drift": range(2000, 2020), "none": range(2100, 2120)}


def schedule(drift: str, tau: float):
    if drift == "none":
        return None
    if drift == "alpha":
        return sch.abrupt("alpha", tau, EPS)
    if drift == "beta":
        return sch.abrupt("beta", tau, EPS, target="hub", boost=10.0)
    if drift == "gamma":
        return sch.abrupt("gamma", tau, EPS, side="src")
    if drift == "delta":
        return sch.abrupt("delta", tau, 1.0, mode="permute")  # full permutation, no epsilon
    if drift == "delta_mix":
        return sch.abrupt("delta", tau, EPS, mode="mix")  # secondary sensitivity (not gating)
    raise ValueError(drift)


def run_one(preset: str, drift: str, seed: int) -> dict:
    w, warm = PRESET_GEOMETRY[preset]
    cfg0 = DSBMConfig.from_yaml(ROOT / "configs" / "synth" / f"{preset}.yaml")
    rate = cfg0.base_rate
    tau = 1.03 * (warm + g1.PROTO.r) * w / rate
    cfg = replace(cfg0, seed=seed, t_end=tau + (g1.PROTO.n_post + 0.4) * w / rate)
    res = generate_stream(cfg, schedule(drift, tau))
    s = res.stream
    i_tau = int(np.searchsorted(s.t, tau))
    r, npost = g1.PROTO.r, g1.PROTO.n_post
    row = {
        "preset": preset,
        "drift": drift,
        "seed": seed,
        "w": w,
        "n_events": len(s),
        "i_tau": i_tau,
    }
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
    case = g1.compute_case(wins, losses, seed=seed)
    return {
        **row,
        **case,
        "device": tg["device"],
        "free_gb_before": tg["free_gb_before"],
        "peak_gb": tg["peak_gb"],
        "tgn_seconds": tg["seconds"],
        "loss_warm_last5": float(tg["loss"][start - 5 * w : start].mean())
        if start >= 5 * w
        else None,
    }


def parse_seeds(txt: str) -> list[int]:
    a, _, b = txt.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev", "official"], required=True)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--seeds", help="dev only, e.g. 4000-4004 (stationary uses the same seeds)")
    ap.add_argument("--drifts", default="none,alpha,beta,gamma,delta")
    args = ap.parse_args()
    out = ROOT / "results" / "g1" / args.mode / args.preset
    out.mkdir(parents=True, exist_ok=True)
    for drift in args.drifts.split(","):
        seeds = (
            parse_seeds(args.seeds)
            if args.mode == "dev"
            else list(OFFICIAL_SEEDS["none" if drift == "none" else "drift"])
        )
        if args.mode == "official" and args.seeds:
            raise SystemExit("official mode uses the frozen seed lists only")
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
