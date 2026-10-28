from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gdb.synth.dsbm import DSBMConfig, generate_stream
from gdb.utils import graph_stats as gs

ROOT = Path(__file__).resolve().parents[1]


def test_sv_spectrum_matches_dense_svd_and_is_scaled():
    rng = np.random.default_rng(0)
    src, dst = rng.integers(0, 30, 400), rng.integers(100, 140, 400)
    pairs = np.unique(np.stack([src, dst], 1), axis=0)
    a = np.zeros((30, 40))
    a[pairs[:, 0], pairs[:, 1] - 100] = 1.0
    ref = np.sort(np.linalg.svd(a, compute_uv=False))[::-1] / np.sqrt(len(pairs))
    got = np.sort(gs.bipartite_sv_spectrum(src, dst))[::-1]
    assert np.allclose(got[: len(ref)], ref[: len(got)], atol=1e-8)
    assert (got**2).sum() == pytest.approx(1.0)  # sum sigma^2 = ||A||_F^2 = m
    assert gs.sv_w1(src, dst, src, dst) == pytest.approx(0.0, abs=1e-12)


def test_concentration_index_extremes():
    assert gs.concentration_index(np.arange(10)) == pytest.approx(0.1)
    assert gs.concentration_index(np.zeros(10, dtype=int)) == pytest.approx(1.0)


@pytest.mark.slow
def test_node_features_make_block_sbm_learnable_for_tgn():
    """With static node features a TGN reliably learns the sharp block SBM (and beats featureless)."""
    from gdb.models.tgn_smoke import run_tgn_smoke

    cfg = DSBMConfig.from_yaml(ROOT / "configs" / "synth" / "tiny_fast.yaml")
    cfg = replace(
        cfg,
        node_sigma=0.3,
        base_rate=150.0,
        t_end=60.0,
        block_matrix={"kind": "assortative", "p_in": 1.0, "p_out": 0.02},
    )
    s = generate_stream(cfg).stream
    plain = run_tgn_smoke(s, epochs=15, lr=3e-3)
    feat = run_tgn_smoke(s, epochs=15, lr=3e-3, use_node_feat=True)
    # a featureless TGN is seed/layout dependent here (0.50-0.68 observed): do not assert chance level
    assert feat["ap"] > 0.68 and feat["ap"] > plain["ap"]
