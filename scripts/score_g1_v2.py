"""Read-only G1 v2 scorer. Reads ONLY results/g1_v2/official/<preset>/ (see gdb.eval.g1_v2_score).

    python scripts/score_g1_v2.py --preset small_bipartite
    python scripts/score_g1_v2.py --overall                 # needs both gating summaries
    python scripts/score_g1_v2.py --preset wiki_like --repeatability   # diagnostic, not a score

Writes results/g1_v2/official/<preset>_summary.json and never overwrites an existing file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gdb.eval import g1_v2_score as sc  # noqa: E402

OFFICIAL = ROOT / "results" / "g1_v2" / "official"
RERUN = ROOT / "results" / "g1_v2" / "official_rerun"


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=[*sc.GATING, "g2_learnable"])
    ap.add_argument("--overall", action="store_true")
    ap.add_argument("--repeatability", action="store_true")
    a = ap.parse_args()
    if a.overall:
        s = {p: json.loads((OFFICIAL / f"{p}_summary.json").read_text()) for p in sc.GATING}
        v = sc.overall(s)
        write_new(OFFICIAL / "g1_v2_verdict.json", json.dumps(v, indent=1))
        print(json.dumps(v, indent=1))
        return
    if not a.preset:
        raise SystemExit("--preset required")
    if a.repeatability:
        rep = sc.repeatability(OFFICIAL / a.preset, RERUN / a.preset, a.preset)
        write_new(RERUN / f"{a.preset}_repeatability.json", json.dumps(rep, indent=1))
        print({k: v for k, v in rep.items() if k != "pairs"})
        return
    res = sc.score_preset(sc.load_rows(OFFICIAL / a.preset, a.preset))
    write_new(OFFICIAL / f"{a.preset}_summary.json", json.dumps(res, indent=1, default=float))
    keys = ("pass", "macro", "delta_fire", "n_", "stationary_false")
    print(json.dumps({k: v for k, v in res.items() if k.startswith(keys)}, indent=1, default=float))


if __name__ == "__main__":
    main()
