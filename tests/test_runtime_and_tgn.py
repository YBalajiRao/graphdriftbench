import time
from dataclasses import replace
from pathlib import Path

import pytest

from gdb.synth import schedules as sch
from gdb.synth.dsbm import DSBMConfig, generate_stream

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_one_million_events_in_under_two_minutes_cpu():
    """wiki_like shape (172-d marks, heavy tails) with a mixed drift; horizon stretched to ~1M events."""
    cfg = DSBMConfig.from_yaml(ROOT / "configs" / "synth" / "wiki_like.yaml")
    cfg = replace(cfg, t_end=cfg.t_end * 1_000_000 / (cfg.base_rate * cfg.t_end))
    sched = sch.abrupt("gamma", cfg.t_end / 2, 0.2) + sch.abrupt("alpha", cfg.t_end / 2, 0.2)
    t0 = time.time()
    res = generate_stream(cfg, sched)
    elapsed = time.time() - t0
    print(f"\n1M-event generation: {len(res.stream)} events in {elapsed:.1f}s")
    assert 0.99e6 < len(res.stream) < 1.01e6
    assert elapsed < 120


@pytest.mark.slow
def test_tgn_smoke_ten_epochs_small_vram():
    import torch

    from gdb.models.tgn_smoke import run_tgn_smoke

    cfg = DSBMConfig.from_yaml(ROOT / "configs" / "synth" / "tiny_fast.yaml")
    # Heavy-tailed popularity gives a learnable signal. On the uniform block SBM a featureless TGN
    # stays at chance (loss = 2 ln 2, AP = 0.5) for 40+ epochs: see docs/progress/step-1.md.
    cfg = replace(
        cfg, pop_dst={"kind": "pareto", "alpha": 1.2}, pop_src={"kind": "pareto", "alpha": 1.5}
    )
    stream = generate_stream(cfg, sch.abrupt("gamma", 50.0, 0.2)).stream
    out = run_tgn_smoke(stream, epochs=10)
    print(f"\nTGN smoke: {out}")
    assert all(map(lambda x: x == x, out["losses"]))  # no NaN
    assert out["losses"][0] - out["losses"][-1] > 0.02
    assert out["ap"] > 0.6
    if out["device"] == "cuda":
        assert out["peak_gb"] < 2.0
        assert torch.cuda.is_available()
