"""Shared helper for the exploratory learnability scripts (block-oracle AP)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from gdb.synth.dsbm import make_block_matrix


def oracle_ap(res, split: int, seed: int) -> float:
    """AP of the true block rate $B_{ab}$; negatives = random training destinations."""
    cfg, s = res.cfg, res.stream
    B = make_block_matrix(cfg.block_matrix, cfg.k_src, cfg.k_dst)
    rng = np.random.default_rng(seed)
    pool = np.unique(s.dst[:split])
    src, dst = s.src[split:], s.dst[split:]
    neg = pool[rng.integers(0, len(pool), len(src))]
    blk = res.node_block
    y = np.r_[np.ones(len(src)), np.zeros(len(src))]
    return float(average_precision_score(y, np.r_[B[blk[src], blk[dst]], B[blk[src], blk[neg]]]))
