"""EXPLORATORY learnability study (not a result of the paper): which stationary link rules can a
small TGN learn, and what is the information ceiling?

    python scripts/learnability_study.py

For each variant it reports a block-oracle AP ceiling (true block-pair rates, negatives drawn from
the training destinations like the smoke test) and the AP of a TGN trained 20 epochs without and with
static node features. Seeds 7000.. (exploratory). Output: results/exploratory/learnability.csv
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from gdb.models.tgn_smoke import run_tgn_smoke  # noqa: E402
from gdb.synth.dsbm import DSBMConfig, generate_stream, make_block_matrix  # noqa: E402

VARIANTS = {
    "uniform_block_beta_h0.5": {},
    "uniform_block_beta_h2": {"beta_h": 2.0},
    "sharp_block (p_out=0.02)": {
        "block_matrix": {"kind": "assortative", "p_in": 1.0, "p_out": 0.02}
    },
    "sharp_block+beta_h2": {
        "block_matrix": {"kind": "assortative", "p_in": 1.0, "p_out": 0.02},
        "beta_h": 2.0,
    },
    "pareto_popularity": {
        "pop_dst": {"kind": "pareto", "alpha": 1.2},
        "pop_src": {"kind": "pareto", "alpha": 1.5},
    },
}


def oracle_ap(res, split: int, seed: int) -> float:
    """AP of the true block-pair rate B[a,b] as score; negatives = random training destinations."""
    cfg, s = res.cfg, res.stream
    B = make_block_matrix(cfg.block_matrix, cfg.k_src, cfg.k_dst)
    rng = np.random.default_rng(seed)
    pool = np.unique(s.dst[:split])
    src, dst = s.src[split:], s.dst[split:]
    neg = pool[rng.integers(0, len(pool), len(src))]
    blk = res.node_block
    pos_sc = B[blk[src], blk[dst] - 0]
    neg_sc = B[blk[src], blk[neg]]
    y = np.r_[np.ones(len(src)), np.zeros(len(src))]
    return float(average_precision_score(y, np.r_[pos_sc, neg_sc]))


def main() -> None:
    base = DSBMConfig.from_yaml(ROOT / "configs" / "synth" / "tiny_fast.yaml")
    base = replace(base, d_node=8, node_sigma=0.3, seed=7000, t_end=100.0, base_rate=150.0)
    rows = []
    for name, over in VARIANTS.items():
        cfg = replace(base, **over)
        res = generate_stream(cfg)
        split = int(0.85 * len(res.stream))
        row = {
            "variant": name,
            "n_events": len(res.stream),
            "oracle_ap": oracle_ap(res, split, 7000),
        }
        for feat in (False, True):
            out = run_tgn_smoke(res.stream, epochs=20, lr=3e-3, use_node_feat=feat)
            row[f"tgn_ap_{'feat' if feat else 'nofeat'}"] = out["ap"]
            row[f"loss_end_{'feat' if feat else 'nofeat'}"] = out["losses"][-1]
        rows.append(row)
        print(row, flush=True)
    out = ROOT / "results" / "exploratory"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "learnability.csv", index=False)


if __name__ == "__main__":
    main()
