"""Generate a synthetic drift stream and write stream.pt, drift_truth.csv, config.yaml.

    python scripts/gen_synth.py --preset small_bipartite --drift gamma:0.2:25 --seeds 0 1 2 3 4
    python scripts/gen_synth.py --preset wiki_like --drift delta:0.2:500:ramp=100 --seeds 0

``--drift type:eps:tau[:ramp=W][:end=T][:key=value ...]`` (repeatable; several = mixed drift).
Output goes to ``data/synth/<name>/`` (git-ignored); ``<name>`` defaults to ``<preset>_<drifts>_s<seed>``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gdb.synth.dsbm import DSBMConfig, generate_stream  # noqa: E402
from gdb.synth.injectors import DriftEvent, DriftSchedule  # noqa: E402


def parse_drift(spec: str) -> DriftEvent:
    ty, eps, tau, *rest = spec.split(":")
    ramp, end, params = 0.0, None, {}
    for item in rest:
        k, v = item.split("=", 1)
        if k == "ramp":
            ramp = float(v)
        elif k == "end":
            end = float(v)
        else:
            try:
                params[k] = json.loads(v)
            except json.JSONDecodeError:
                params[k] = v
    return DriftEvent(
        float(tau),
        ty,
        float(eps),
        params,
        onset="gradual" if ramp > 0 else "abrupt",
        ramp_width=ramp,
        end=end,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--drift", action="append", default=[])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=str(ROOT / "data" / "synth"))
    args = ap.parse_args()

    cfg0 = DSBMConfig.from_yaml(ROOT / "configs" / "synth" / f"{args.preset}.yaml")
    schedule = DriftSchedule([parse_drift(s) for s in args.drift])
    tag = "-".join(f"{e.type}{e.eps:g}" for e in schedule.events) or "stationary"
    for seed in args.seeds:
        name = args.name or f"{args.preset}_{tag}_s{seed}"
        t0 = time.time()
        res = generate_stream(replace(cfg0, seed=seed), schedule)
        res.save(Path(args.out) / name)
        print(
            f"{name}: {len(res.stream)} events, {res.stream.n_nodes} nodes, "
            f"{time.time() - t0:.1f}s, hash {res.stream.hash()[:16]}"
        )


if __name__ == "__main__":
    main()
