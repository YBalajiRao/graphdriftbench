"""EXPLORATORY: pre-drift predictive signal of the original vs the G2 learnable preset.

    python scripts/g2_learnability.py

Development seeds 0-4 (not G1 seeds). For each preset: block-oracle AP, TGN AP without and with
node features (20 epochs, lr 3e-3, held-out last 15%), mean and sd over seeds. Also reports the
factor-purity checks relevant to G2 streams: delta[permute] destination popularity rank
correlation and gamma active-id counts before/after. Output: results/exploratory/g2_learnability.csv
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gdb.models.tgn_smoke import run_tgn_smoke  # noqa: E402
from gdb.synth import schedules as sch  # noqa: E402
from gdb.synth.dsbm import DSBMConfig, generate_stream  # noqa: E402
from scripts_learn_helpers import oracle_ap  # noqa: E402

PRESETS = ["small_bipartite", "g2_learnable"]


def main() -> None:
    rows = []
    for name in PRESETS:
        cfg0 = DSBMConfig.from_yaml(ROOT / "configs" / "synth" / f"{name}.yaml")
        for seed in range(5):
            cfg = replace(cfg0, seed=seed, t_end=100.0 if name == "g2_learnable" else 150.0)
            res = generate_stream(cfg)
            split = int(0.85 * len(res.stream))
            row = {
                "preset": name,
                "seed": seed,
                "n_events": len(res.stream),
                "oracle_ap": oracle_ap(res, split, seed),
            }
            for feat in (False, True):
                out = run_tgn_smoke(res.stream, epochs=20, lr=3e-3, seed=seed, use_node_feat=feat)
                k = "feat" if feat else "nofeat"
                row[f"tgn_ap_{k}"], row[f"loss_end_{k}"] = out["ap"], out["losses"][-1]
                row["device"] = out["device"]
            # purity checks relevant for G2 streams (delta popularity, gamma turnover)
            half = cfg.t_end / 2
            d = generate_stream(replace(cfg, t_end=2 * half), sch.abrupt("delta", half, 1.0))
            s = d.stream
            pre, post = s.t < half, s.t >= half
            a = np.bincount(s.dst[pre] - cap(cfg), minlength=cfg.n_dst)
            b = np.bincount(s.dst[post] - cap(cfg), minlength=cfg.n_dst)
            from scipy.stats import spearmanr

            row["delta_dst_popularity_spearman"] = float(spearmanr(a, b).statistic)
            g = generate_stream(
                replace(cfg, t_end=2 * half), sch.abrupt("gamma", half, 0.2, side="both")
            ).stream
            pre, post = g.t < half, g.t >= half
            row["gamma_active_src_pre_post"] = (
                f"{len(np.unique(g.src[pre]))}/{len(np.unique(g.src[post]))}"
            )
            row["gamma_active_dst_pre_post"] = (
                f"{len(np.unique(g.dst[pre]))}/{len(np.unique(g.dst[post]))}"
            )
            rows.append(row)
            print(row, flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "exploratory"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "g2_learnability.csv", index=False)
    print(
        df.groupby("preset")[["oracle_ap", "tgn_ap_nofeat", "tgn_ap_feat"]]
        .agg(["mean", "std"])
        .round(3)
        .to_string()
    )


def cap(cfg: DSBMConfig) -> int:
    return cfg.capacity(cfg.n_src) if cfg.id_layout == "reserved" else cfg.n_src


if __name__ == "__main__":
    main()
