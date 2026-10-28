import numpy as np
import pandas as pd
import pytest

from gdb.detect import standardise as sd
from gdb.detect import window_stats as ws
from gdb.eval import g1


def test_block_bootstrap_moments_recover_iid_series():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 2.0, 200)
    m = sd.moving_block_moments(x, block=2, n_boot=500, seed=1)
    assert m.mu == pytest.approx(x.mean(), abs=0.05) and m.sigma == pytest.approx(
        x.std(ddof=1), rel=0.05
    )
    assert m.mu_ci[0] < m.mu < m.mu_ci[1] and m.sigma_ci[0] < m.sigma < m.sigma_ci[1]
    assert sd.moving_block_moments(x, block=2, n_boot=500, seed=1) == m  # deterministic


def test_block_bootstrap_rejects_degenerate_and_nonfinite():
    with pytest.raises(ValueError):
        sd.moving_block_moments(np.ones(10))
    with pytest.raises(ValueError):
        sd.moving_block_moments(np.array([1.0, np.nan, 2.0, 3.0]))


def test_huber_recovers_linear_loss_and_downweights_outlier():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(40, 3))
    loss = f @ np.array([0.5, -0.2, 0.1]) + 1.0 + 0.01 * rng.normal(size=40)
    loss[3] += 5.0  # outlier
    w = sd.huber_fit(f, loss)
    assert np.allclose(w[:3], [0.5, -0.2, 0.1], atol=0.08) and w[3] == pytest.approx(1.0, abs=0.15)
    assert abs(sd.loo_residuals(f, loss)[3]) > 3.0


def _win(rng, nodes_s, nodes_d, n=300, d=8, shift=0.0):
    return ws.Window(
        rng.choice(nodes_s, n),
        rng.choice(nodes_d, n),
        (rng.normal(size=(n, d)) + shift).astype(np.float32),
    )


def test_gamma_components_detect_new_nodes_only():
    rng = np.random.default_rng(0)
    s, d = np.arange(0, 60), np.arange(100, 130)
    pool = [_win(rng, s, d) for _ in range(5)]
    same = ws.gamma_components(_win(rng, s, d), pool)
    new = ws.gamma_components(_win(rng, np.arange(1000, 1060), d), pool)
    assert new[1] > same[1] + 0.3 and new[0] > same[0] + 0.1


def test_alpha_statistic_responds_to_mark_shift_not_to_identical_marks():
    rng = np.random.default_rng(1)
    s, d = np.arange(0, 60), np.arange(100, 130)
    pool = [_win(rng, s, d, n=800) for _ in range(5)]
    p = ws.persistent_nodes(_win(rng, s, d), pool)
    a0 = ws.alpha_statistic(_win(rng, s, d, n=800), pool, p)
    a1 = ws.alpha_statistic(_win(rng, s, d, n=800, shift=0.6), pool, p)
    assert a1 > a0 + 0.001


def test_mean_spectrum_requires_common_grid():
    from gdb.utils import graph_stats as gs

    a = gs.Spectrum(np.linspace(0, 2, 5), np.full(5, 0.2))
    b = gs.Spectrum(np.linspace(0, 1, 5), np.full(5, 0.2))
    with pytest.raises(ValueError):
        ws.mean_spectrum([a, b])
    assert ws.mean_spectrum([a, a]).weights.sum() == pytest.approx(1.0)


def _toy_df():
    rows = []
    for c, hot in zip(g1.TYPES, range(4)):
        for _ in range(5):
            z = np.full(4, 0.5)
            z[hot] = 6.0
            rows.append(
                {
                    "drift": c,
                    "status": "ok",
                    **{f"z_{t}_w{m}": z[k] for m in range(3) for k, t in enumerate(g1.TYPES)},
                }
            )
    for _ in range(5):
        rows.append(
            {
                "drift": "none",
                "status": "ok",
                **{f"z_{t}_w{m}": 0.1 for m in range(3) for t in g1.TYPES},
            }
        )
    return pd.DataFrame(rows)


def test_scoring_toy_perfect_case_and_no_detection_scoring():
    df = _toy_df()
    r = g1.score_g1(df, 0, w=1000)
    assert r["macro_f1_argmax"] == 1.0 and r["pass_a_diag_dominant"] and r["pass_all"]
    assert r["detection_recall"] == 1.0 and r["false_alarm_streams"] == 0 and r["missed"] == 0
    # make half of the delta streams undetected: low z everywhere -> 'none' in the thresholded score
    m = (df.drift == "delta") & (df.index % 2 == 0)
    for col in [c for c in df if c.startswith("z_")]:
        df.loc[m, col] = 0.2
    r2 = g1.score_g1(df, 0, w=1000)
    assert r2["missed"] == int(m.sum()) and r2["macro_f1_thresholded"] < 1.0
    assert r2["confusion_thresholded"].loc["delta", "none"] == int(m.sum())


def test_ties_and_failed_streams_are_errors_not_dropped():
    df = _toy_df()
    df.loc[0, [f"z_{t}_w0" for t in g1.TYPES]] = 2.0  # exact tie
    r = g1.score_g1(df, 0, w=1000)
    assert r["confusion_argmax"].loc["alpha", "tie"] == 1 and r["macro_f1_argmax"] < 1.0
    df2 = _toy_df()
    df2.loc[1, "status"] = "error:X"
    r2 = g1.score_g1(df2, 0, w=1000)
    assert r2["n_failed"] == 1 and not r2["pass_all"]


def test_delta_fire_rule_uses_only_pure_types():
    df = _toy_df()
    df.loc[df.drift == "alpha", "z_delta_w0"] = 5.0  # delta fires on pure alpha
    r = g1.score_g1(df, 0, w=1000)
    assert r["delta_fire_by_type"]["alpha"] == 1.0 and not r["pass_c_delta_fire"]


def test_sigma_floor_only_when_given():
    x = np.zeros(10)
    with pytest.raises(ValueError):
        sd.moving_block_moments(x)
    m = sd.moving_block_moments(x, sigma_floor=0.01)
    assert m.sigma == 0.01 and sd.zscore(0.1, m) == pytest.approx(10.0)
    y = np.random.default_rng(0).normal(0, 1, 10)
    assert sd.moving_block_moments(y, sigma_floor=1e-6).sigma == sd.moving_block_moments(y).sigma
