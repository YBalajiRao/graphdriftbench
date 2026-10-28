"""EXPLORATORY: effective magnitude of eps for each injector (block-pair law, total variation).

For beta/delta the block-pair event law is p_ab = B_ab n_a n_b / Z; the table reports TV(p, p') at the
given eps, and for delta also the spectral change of B (singular values), which is zero only at eps=1.
Output: results/exploratory/eps_effective.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gdb.synth.dsbm import DSBMConfig, _block_labels, make_block_matrix  # noqa: E402
from gdb.synth.injectors import beta_target, delta_permutation  # noqa: E402

rows = []
for preset in ("small_bipartite", "wiki_like"):
    cfg = DSBMConfig.from_yaml(ROOT / "configs" / "synth" / f"{preset}.yaml")
    B = make_block_matrix(cfg.block_matrix, cfg.k_src, cfg.k_dst)
    ns = np.diff(_block_labels(cfg.n_src, cfg.k_src)[1]).astype(float)
    nd = np.diff(_block_labels(cfg.n_dst, cfg.k_dst)[1]).astype(float)
    w_ab = np.outer(ns, nd)

    def p(M, w_ab=w_ab):
        return (M * w_ab) / (M * w_ab).sum()

    def sv(M, w_ab=w_ab):
        return np.linalg.svd(np.sqrt(w_ab) * M, compute_uv=False)

    rng = np.random.default_rng(0)
    perm = delta_permutation(B, ns, nd, False, rng, None)
    tg = {
        "beta[hub,boost10]": beta_target(B, ns, nd, "hub", rng, True, boost=10.0),
        "beta[merge]": beta_target(B, ns, nd, "merge", rng, True),
        "beta[densify,p4]": beta_target(B, ns, nd, "densify", rng, True, power=4.0),
        "delta": B[:, perm],
    }
    for eps in (0.05, 0.1, 0.2, 0.4, 1.0):
        for name, T in tg.items():
            M = (1 - eps) * B + eps * T
            s0, s1 = sv(B), sv(M)
            rows.append(
                {
                    "preset": preset,
                    "injector": name,
                    "eps": eps,
                    "tv_blockpair_law": 0.5 * np.abs(p(M) - p(B)).sum(),
                    "spectral_change_B": np.abs(s1 / s1.sum() - s0 / s0.sum()).sum() * 0.5,
                }
            )
    d = cfg.d_edge
    for eps in (0.05, 0.1, 0.2, 0.4):
        rows.append(
            {
                "preset": preset,
                "injector": "alpha(shift/(sigma*sqrt(d)))",
                "eps": eps,
                "tv_blockpair_law": 0.0,
                "spectral_change_B": 0.0,
                "alpha_shift_norm": eps * cfg.edge_sigma * np.sqrt(d),
                "alpha_unit_vector_norm": eps,
            }
        )
df = pd.DataFrame(rows)
out = ROOT / "results" / "exploratory"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "eps_effective.csv", index=False)
print(df[df.eps.isin([0.2, 1.0])].round(4).to_string(index=False))
