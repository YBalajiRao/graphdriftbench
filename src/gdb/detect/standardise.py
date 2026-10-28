r"""Moving-block bootstrap standardisation and the Huber loss-residual (Blueprint §5.1).

$z_c(k)=(s_c(k)-\mu_c^R)/\sigma_c^R$ with $\mu_c^R,\sigma_c^R$ from a moving-block bootstrap of the
reference windows' statistic series $s_c(1..r)$ (each computed leave-one-out). Moving blocks of
length $L$ (overlapping, $r-L+1$ possible) are drawn with replacement until $r$ values are
collected; per replicate the mean and the sample variance are stored;

$$\hat\mu=\overline{\mu^{*}},\qquad \hat\sigma=\sqrt{\overline{(\sigma^{*})^2}},$$

and the 2.5/97.5% percentiles of the replicate moments are returned as the uncertainty of the
standardisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import HuberRegressor


@dataclass(frozen=True)
class Moments:
    mu: float
    sigma: float
    mu_ci: tuple[float, float]
    sigma_ci: tuple[float, float]


def moving_block_moments(
    series: np.ndarray,
    *,
    block: int = 2,
    n_boot: int = 500,
    seed: int = 0,
    sigma_floor: float = 0.0,
) -> Moments:
    """Moving-block bootstrap moments of a reference statistic series (see module docstring).

    ``sigma_floor`` (default 0 = none) is a pre-registered resolution floor for count-based
    statistics that are identically 0 on dense reference windows (e.g. the new-node rate): the
    returned sigma is ``max(sigma, sigma_floor)``. Without a floor a zero-variance series raises.
    """
    s = np.asarray(series, dtype=float)
    r = len(s)
    if r < block + 1 or not np.all(np.isfinite(s)):
        raise ValueError("reference series too short or non-finite")
    rng = np.random.default_rng(seed)
    n_blocks = -(-r // block)
    starts_all = np.arange(r - block + 1)
    mus, vars_ = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        st = rng.choice(starts_all, size=n_blocks, replace=True)
        x = np.concatenate([s[i : i + block] for i in st])[:r]
        mus[b], vars_[b] = x.mean(), x.var(ddof=1)
    sig = float(np.sqrt(vars_.mean()))
    if not max(sig, sigma_floor) > 1e-12:
        raise ValueError("degenerate reference series (zero variance)")
    sig = max(sig, sigma_floor)
    q = lambda v: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))  # noqa: E731
    return Moments(
        float(mus.mean()),
        sig,
        q(mus),
        (max(q(np.sqrt(vars_))[0], sigma_floor), max(q(np.sqrt(vars_))[1], sigma_floor)),
    )


def zscore(x: float, m: Moments) -> float:
    return (x - m.mu) / m.sigma


def huber_fit(features: np.ndarray, loss: np.ndarray) -> np.ndarray:
    r"""Huber regression of the loss on $[z_\alpha,z_\beta,z_\gamma]$ plus intercept; returns $\mathbf w$."""
    h = HuberRegressor(alpha=1e-4, max_iter=1000).fit(features, loss)
    return np.r_[h.coef_, h.intercept_]


def residual(w: np.ndarray, features: np.ndarray, loss: np.ndarray) -> np.ndarray:
    r"""$\ell-\hat\ell=\ell-\mathbf w^\top[z,1]$."""
    return loss - features @ w[:-1] - w[-1]


def loo_residuals(features: np.ndarray, loss: np.ndarray) -> np.ndarray:
    """Out-of-sample residuals of the reference windows (each window predicted by a fit without it)."""
    out = np.empty(len(loss))
    for j in range(len(loss)):
        keep = np.arange(len(loss)) != j
        w = huber_fit(features[keep], loss[keep])
        out[j] = residual(w, features[j : j + 1], loss[j : j + 1])[0]
    return out
