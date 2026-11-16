"""Scorer contract tests on development fixtures only (tmp dirs shaped like results/g1_v2/official)."""

import json
from pathlib import Path

import numpy as np
import pytest

from gdb.eval import g1
from gdb.eval import g1_v2_score as sc
from gdb.eval.g1_raw import calibrate as g1_calibrate

PRESET = "small_bipartite"


def _row(preset, drift, seed, z=None, status="ok", w=2000):
    row = {"preset": preset, "drift": drift, "seed": seed, "status": status, "w": w}
    if status == "ok":
        for m in range(3):
            for t, v in zip(g1.TYPES, z):
                row[f"z_{t}_w{m}"] = float(v)
    return row


def _fixture(root: Path, preset=PRESET, mutate=None):
    d = root / "g1_v2" / "official" / preset
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for drift, seeds in sc.EXPECTED.items():
        if drift == "delta_mix":
            continue
        for s in seeds:
            z = rng.normal(0, 0.5, 4)
            if drift in g1.TYPES:
                z[g1.TYPES.index(drift)] += 10.0
            row = _row(preset, drift, s, z)
            if mutate:
                mutate(drift, s, row)
            (d / f"{drift}_{s}.json").write_text(json.dumps(row))
    return d


def _score(d, preset=PRESET, **kw):
    return sc.score_preset(sc.load_rows(d, preset), n_boot=200, **kw)


def _set(row, **zs):
    for m in range(3):
        for k, v in zs.items():
            row[f"z_{k}_w{m}"] = v


def test_perfect_fixture_passes_all_and_counts(tmp_path):
    r = _score(_fixture(tmp_path))
    assert r["pass_all"] and r["macro_f1_argmax"] == 1.0 and r["delta_fire_pooled"] == 0.0
    assert r["n_expected_drift"] == 80 and r["n_expected_stationary"] == 20
    assert r["delta_fire_n_pure_streams"] == 60 and r["n_failed_or_missing"] == 0


def test_delta_fire_threshold_is_inclusive_at_15_percent(tmp_path):
    def bump(n):
        cnt = {"k": 0}

        def f(drift, s, row):
            if drift in ("alpha", "beta", "gamma") and cnt["k"] < n:
                cnt["k"] += 1
                _set(row, delta=5.0)

        return f

    assert _score(_fixture(tmp_path / "a", mutate=bump(9)))["pass_c_delta_fire"]  # 9/60 = 0.15
    assert not _score(_fixture(tmp_path / "b", mutate=bump(10)))["pass_c_delta_fire"]


def test_exact_tie_is_an_error_and_not_resolved_by_index(tmp_path):
    def f(drift, s, row):
        if (drift, s) == ("beta", 3000):
            _set(row, alpha=9.0, beta=9.0)

    r = _score(_fixture(tmp_path, mutate=f))
    assert r["confusion_argmax"]["tie"]["beta"] == 1 and r["macro_f1_argmax"] < 1.0


def test_audit_requires_unique_diagonal_max(tmp_path):
    def f(drift, s, row):
        if drift == "alpha":
            _set(row, alpha=5.0, beta=5.0)

    r = _score(_fixture(tmp_path, mutate=f))
    assert r["diagonal_unique_max_rows"]["alpha"] is False and not r["pass_a_diagonal_unique_max"]


def test_missing_failed_and_nonfinite_rows_stay_visible_and_block_pass(tmp_path):
    d = _fixture(tmp_path)
    (d / "gamma_3001.json").unlink()
    bad = _row(PRESET, "alpha", 3002, status="error:ValueError:x")
    (d / "alpha_3002.json").write_text(json.dumps(bad))
    r = json.loads((d / "beta_3003.json").read_text())
    r["z_beta_w1"] = float("nan")
    (d / "beta_3003.json").write_text(json.dumps(r))
    out = _score(d)
    assert out["n_failed_or_missing"] == 3 and not out["pass_all"]
    assert {x["status"] for x in out["failed_streams"]} == {
        "missing",
        "error:ValueError:x",
        "nonfinite",
    }
    assert out["confusion_argmax"]["fail"]["gamma"] == 1  # labelled 'fail', not dropped
    assert out["macro_f1_argmax"] < 1.0


def test_wrong_preset_duplicate_unknown_and_quarantined_seed_raise(tmp_path):
    d = _fixture(tmp_path / "w")
    bad = json.loads((d / "alpha_3000.json").read_text())
    bad["preset"] = "wiki_like"
    (d / "alpha_3000.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        sc.load_rows(d, PRESET)
    d2 = _fixture(tmp_path / "d")
    (d2 / "alpha_03000.json").write_text((d2 / "alpha_3000.json").read_text())
    with pytest.raises(ValueError, match="duplicate"):
        sc.load_rows(d2, PRESET)
    d3 = _fixture(tmp_path / "q")
    (d3 / "none_3100.json").write_text(json.dumps(_row(PRESET, "none", 3100, np.zeros(4))))
    with pytest.raises(ValueError, match="allocation"):
        sc.load_rows(d3, PRESET)
    d4 = _fixture(tmp_path / "u")
    (d4 / "weird_3000.json").write_text("{}")
    with pytest.raises(ValueError):
        sc.load_rows(d4, PRESET)
    with pytest.raises(ValueError, match="official"):
        sc.load_rows(_fixture(tmp_path / "p"), "wiki_like")  # directory name != preset


@pytest.mark.parametrize(
    "bad_parent",
    ["official_rerun", "dev_raw", "quarantine_stationary_3100_3119_pre_approval", "official_x"],
)
def test_refuses_non_official_trees(tmp_path, bad_parent):
    d = tmp_path / "g1_v2" / bad_parent / PRESET
    d.mkdir(parents=True)
    with pytest.raises(ValueError, match="official"):
        sc.load_rows(d, PRESET)
    d1 = tmp_path / "g1" / "official" / PRESET  # a v1-shaped tree
    d1.mkdir(parents=True)
    with pytest.raises(ValueError, match="official"):
        sc.load_rows(d1, PRESET)


def test_stationary_streams_do_not_enter_macro_f1_and_are_reported_separately(tmp_path):
    def f(drift, s, row):
        if drift == "none" and s < 3205:
            _set(row, gamma=50.0)

    base = _score(_fixture(tmp_path / "a"))
    noisy = _score(_fixture(tmp_path / "b", mutate=f))
    assert base["macro_f1_argmax"] == noisy["macro_f1_argmax"] == 1.0
    assert noisy["stationary_false_alarm_streams"] == 5
    assert base["stationary_false_alarm_streams"] == 0
    lo, hi = noisy["stationary_false_alarm_rate_ci95_stream_bootstrap_conditional_on_valid"]
    assert lo <= 0.25 <= hi


def test_bootstrap_is_deterministic_and_seed_clustered(tmp_path):
    d = _fixture(tmp_path)
    a, b = _score(d), _score(d)
    assert (
        a["macro_f1_argmax_ci95_seed_clustered_failure_aware"]
        == b["macro_f1_argmax_ci95_seed_clustered_failure_aware"]
    )
    assert (
        a["n_seed_clusters"] == 20
        and a["audit_lo_conditional_on_valid"] == b["audit_lo_conditional_on_valid"]
    )


def test_scorer_does_not_modify_source_rows(tmp_path):
    d = _fixture(tmp_path)
    before = {f.name: f.read_bytes() for f in d.glob("*.json")}
    _score(d)
    after = {f.name: f.read_bytes() for f in d.glob("*.json")}
    assert before == after and not list(d.glob("*summary*"))


def test_repeatability_reports_flips_and_never_merges(tmp_path):
    d = _fixture(tmp_path)
    rr = tmp_path / "g1_v2" / "official_rerun" / PRESET
    rr.mkdir(parents=True)
    for name, flip in (("alpha_3000", False), ("beta_3000", True)):
        row = json.loads((d / f"{name}.json").read_text())
        if flip:
            _set(row, gamma=50.0)
        (rr / f"{name}.json").write_text(json.dumps(row))
    failed = _row(PRESET, "gamma", 3000, status="error:x")
    (rr / "gamma_3000.json").write_text(json.dumps(failed))
    rep = sc.repeatability(d, rr, PRESET)
    assert rep["n_pairs"] == 2 and rep["label_flips"] == 1
    assert rep["not_comparable_or_failed"] == ["gamma_3000.json"]
    assert _score(d)["pass_all"]  # first-run score unchanged by reruns
    with pytest.raises(ValueError):
        sc.repeatability(d, d, PRESET)


def test_overall_requires_both_gating_configs_to_pass():
    base = {"n_expected_drift": 80, "n_expected_stationary": 20}
    ok, bad = {"pass_all": True, **base}, {"pass_all": False, **base}
    assert sc.overall({"small_bipartite": ok, "wiki_like": ok})["g1_v2_passed"]
    assert not sc.overall({"small_bipartite": ok, "wiki_like": bad})["g1_v2_passed"]
    with pytest.raises(ValueError):
        sc.overall({"small_bipartite": ok})


def test_failed_stream_stays_an_error_in_the_expected_denominator(tmp_path):
    d = _fixture(tmp_path)
    (d / "gamma_3001.json").unlink()
    r = _score(d)
    # 19 of 20 gamma streams correct, 1 failed: F1_gamma = 2*19 / (2*19 + 0 + 1)
    assert r["f1_by_type"]["gamma"] == pytest.approx(38 / 39)
    assert r["n_expected_drift"] == 80 and r["confusion_argmax"]["fail"]["gamma"] == 1
    assert (
        "failure_aware" in " ".join(r)
        and r["macro_f1_argmax_ci95_seed_clustered_failure_aware"][1] < 1.0 + 1e-12
    )


def test_low_max_z_still_gets_argmax_label_in_primary_f1_but_not_in_thresholded(tmp_path):
    def f(drift, s, row):
        if (drift, s) == ("alpha", 3000):
            _set(row, alpha=2.0, beta=0.1, gamma=0.1, delta=0.1)  # max z <= 3, argmax still alpha

    r = _score(_fixture(tmp_path, mutate=f))
    assert r["macro_f1_argmax"] == 1.0 and r["macro_f1_thresholded_nongating"] < 1.0
    assert r["missed_detections_nongating"] == 1


def test_stationary_rate_per_1e5_events_formula_and_failed_stream_handling(tmp_path):
    def f(drift, s, row):
        if drift == "none" and s < 3204:
            _set(row, beta=9.0)  # 4 alarms

    d = _fixture(tmp_path, mutate=f)
    (d / "none_3219.json").unlink()  # one failed stationary stream: not in the event denominator
    r = _score(d)
    assert r["stationary_false_alarm_streams"] == 4 and r["stationary_valid_streams"] == 19
    assert r["stationary_tested_events"] == 19 * 2000
    assert r["stationary_stream_alarms_per_1e5_tested_events"] == pytest.approx(
        4 / (19 * 2000) * 1e5
    )
    assert r["stationary_false_alarm_rate_if_failed_streams_alarmed"] == pytest.approx(5 / 20)
    assert not r["pass_all"]  # a failed stationary stream also blocks the pass


def test_incomplete_or_empty_inputs_never_pass_and_do_not_crash(tmp_path):
    empty = tmp_path / "e" / "g1_v2" / "official" / PRESET
    empty.mkdir(parents=True)
    r = _score(empty)
    assert not r["pass_all"] and r["n_failed_or_missing"] == 100
    d = _fixture(tmp_path / "half")
    for f in list(d.glob("beta_*.json")):
        f.unlink()
    r2 = _score(d)
    assert not r2["pass_all"] and r2["n_failed_or_missing"] == 20
    assert r2["delta_fire_upper_if_failed_pure_streams_fired"] >= r2["delta_fire_pooled"]


def test_failure_blocks_pass_even_if_every_valid_criterion_holds(tmp_path):
    d = _fixture(tmp_path)
    assert _score(d)["pass_all"]
    (d / "delta_3000.json").write_text(json.dumps(_row(PRESET, "delta", 3000, status="error:x")))
    r = _score(d)
    assert r["pass_a_diagonal_unique_max"] and r["pass_c_delta_fire"] and not r["pass_all"]


def test_overall_rejects_summaries_that_do_not_cover_expected_streams():
    ok = {"pass_all": True, "n_expected_drift": 80, "n_expected_stationary": 20}
    assert sc.overall({"small_bipartite": ok, "wiki_like": ok})["g1_v2_passed"]
    with pytest.raises(ValueError, match=r"80 \+ 20"):
        sc.overall({"small_bipartite": ok, "wiki_like": {**ok, "n_expected_drift": 79}})


def test_zero_valid_class_marks_audit_row_unavailable_reports_counts_and_fails(tmp_path):
    d = _fixture(tmp_path)
    for f in d.glob("gamma_*.json"):
        f.write_text(json.dumps(_row(PRESET, "gamma", int(f.stem.split("_")[1]), status="error:x")))
    r = _score(d)
    assert r["audit_row_available"]["gamma"] is False and r["audit_row_available"]["alpha"] is True
    assert (
        r["valid_streams_by_type"]["gamma"] == 0
        and r["failed_or_missing_streams_by_type"]["gamma"] == 20
    )
    assert (
        r["valid_streams_by_type"]["alpha"] == 20
        and r["failed_or_missing_streams_by_type"]["alpha"] == 0
    )
    assert r["diagonal_unique_max_rows"]["gamma"] is False and not r["pass_all"]
    assert all(np.isnan(v) for v in (r["audit_mean"][t]["gamma"] for t in g1.TYPES))


def test_failure_forces_fail_even_with_perfect_audit_f1_and_intervals(tmp_path):
    d = _fixture(tmp_path)
    (
        d / "none_3205.json"
    ).unlink()  # only a stationary stream fails; every drift criterion is perfect
    r = _score(d)
    assert r["pass_a_diagonal_unique_max"] and r["pass_b_macro_f1"] and r["pass_c_delta_fire"]
    assert r["n_failed_or_missing"] == 1 and not r["pass_all"]


def test_stationary_summary_states_its_definition(tmp_path):
    r = _score(_fixture(tmp_path))
    s = r["stationary_alarm_definition"]
    assert "NOT an online detector false-alarm rate" in s and "A/E*1e5" in s


def test_bootstrap_replicate_counts_are_the_frozen_ones():
    import inspect

    assert g1.PROTO.n_boot == 500 and sc.N_BOOT == 2000
    assert inspect.signature(sc.score_preset).parameters["n_boot"].default == 2000
    assert inspect.signature(g1_calibrate).parameters["proto"].default.n_boot == 500
