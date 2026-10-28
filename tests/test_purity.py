"""Generator/injector purity tests (random tests: pass rules are counts over 20 seeds, see gdb.synth.purity)."""

from pathlib import Path

import numpy as np
import pytest

from gdb.synth import purity as P

ROOT = Path(__file__).resolve().parents[1]
BAND_FILE = ROOT / "results" / "null_band_small_bipartite.json"


def test_null_band_is_reproducible_and_positive(small_cfg):
    a = P.build_null_band(small_cfg, 2000, n_null=8)
    b = P.build_null_band(small_cfg, 2000, n_null=8)
    assert a["band"] == b["band"]
    assert all(v >= 0 for v in a["band"].values()) and a["window_events"] == 2000


def test_purity_fast_sanity_three_seeds(small_cfg):
    """Cheap smoke version of the 20-seed check: the designed shifts are visible, the null is not."""
    band = P.build_null_band(small_cfg, 2000, n_null=40)
    seeds = range(3)
    for ty, stat in [
        ("alpha", "mark_mmd"),
        ("gamma", "new_node_rate_src"),
        ("delta", "blockpair_chi2"),
    ]:
        df = P.run_injection(small_cfg, 2000, ty, 0.2, seeds)
        assert (df[stat] > band["band"][stat]).all(), (ty, stat)
    # the shifted statistic of one type must not move for another type
    df = P.run_injection(small_cfg, 2000, "gamma", 0.2, seeds)
    assert (df["mark_mmd"] <= band["band"]["mark_mmd"]).sum() >= 2
    assert (df["blockpair_chi2"] <= band["band"]["blockpair_chi2"]).sum() >= 2


def test_pass_rate_rules_are_the_specified_counts():
    assert (P.N_SEEDS, P.NULL_MIN_INSIDE, P.SHIFT_MIN_DETECT) == (20, 17, 18)
    import pandas as pd

    band = {"band": {s: 1.0 for s in P.STATS}}
    df = pd.DataFrame({s: [0.5] * 17 + [2.0] * 3 for s in P.STATS})  # 17 inside, 3 outside
    t = P.pass_rates(df, band, "alpha").set_index("statistic")
    assert bool(t.loc["deg_ks_src", "pass"]) is True  # null cell: 17/20 inside is enough
    assert bool(t.loc["mark_mmd", "pass"]) is False  # shift cell: needs 18 outside, has 3


@pytest.mark.slow
@pytest.mark.skipif(
    not BAND_FILE.exists(), reason="run scripts/purity_report.py --preset small_bipartite first"
)
@pytest.mark.parametrize(
    "label,ty,params,primary,expected", P.VARIANTS, ids=[v[0] for v in P.VARIANTS]
)
def test_purity_20_seeds(small_cfg, label, ty, params, primary, expected):
    """Primary rows must pass; non-primary rows are exploratory and only checked for being computable."""
    band = P.load_band(BAND_FILE)
    df = P.run_injection(small_cfg, band["window_events"], ty, 0.2, range(P.N_SEEDS), **params)
    table = P.pass_rates(df, band, ty, label=label, expected=expected)
    assert np.isfinite(table["median_value"]).all()
    if not primary:
        return
    if label == "beta[hub]":
        # Known, reported result: DoS detects beta[hub] in 17/20 seeds at eps=0.2, w=2000 (needs 18).
        # Not tuned away; see docs/progress/step-1.md. strict=True flags it if the number ever changes.
        failed = table[table["pass"].eq(False)]["statistic"].tolist()
        assert failed == ["dos_w1"], failed
        return
    failed = table[table["pass"].eq(False)]["statistic"].tolist()
    assert not failed, failed
