r"""Revised relational-semantic residual (G1 protocol v2 candidate; the v1 residual is kept).

v1 failure. $r_\delta=\ell-\mathbf w^\top[z_\alpha,z_\beta,z_\gamma,1]$ with $\mathbf w$ fitted on ten
reference windows whose $z$ values are, by construction, $O(1)$ noise. A test window with
$z_\gamma\approx10^2$ evaluates the line a hundred reference-ranges away, so $r_\delta$ is dominated
by the fitted slope (noise) times an out-of-range input, not by the loss. Also, reference residuals
were leave-one-out while the test residual used the full fit.

v2 residual (all inputs come from reference windows and the test window itself).

1. *Bounded nuisance prediction.* For a model fitted on a training set $T$ of reference windows,
   the features are projected onto the training support,
   $\tilde z=\Pi_{[\min_T z,\ \max_T z]}(z)$, and $\hat\ell=\mathbf w_T^\top[\tilde z,1]$. The nuisance
   explanation of a loss change therefore never exceeds what the reference data can support
   (|$\hat\ell-\bar\ell$| $\le$ |$\mathbf w_T$|·range), whatever the size of $z$.
2. *Comparable calibration (jackknife).* Reference residual $j$ is $\ell_j-\hat\ell_{-j}(z_j)$ with the
   model fitted on the other $r-1$ windows. A test residual is the average over $j$ of
   $\ell-\hat\ell_{-j}(z)$: every prediction used on either side comes from an $(r-1)$-window model.
3. ``nuisance="none"`` replaces the regression by the training-window median loss (the
   "generic loss shift" ablation).
4. *Attribution gate (optional).* With $z_{\max}=\max(z_\alpha,z_\beta,z_\gamma)$,
   $\tilde z_\delta=z_\delta$ if $z_{\max}\le z_{\rm fire}$ else $\min(z_\delta,0)$: a loss increase
   that coincides with a *detected* nuisance drift is not credited to $\delta$. Consequence (stated as
   a limitation, not hidden): $\delta$ concurrent with a detected $\alpha/\beta/\gamma$ is not
   identifiable by this statistic.

The regression slopes cannot be learned from the reference alone because the reference contains no
drift; item 1 makes the slopes' effect *bounded* rather than correct. Item 4 is the explicit
assumption that replaces the missing nuisance model.
"""

from __future__ import annotations

import numpy as np

from gdb.detect.standardise import huber_fit

Nuisance = str  # "bounded" | "none"


def project_to_support(z: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    r"""Coordinate-wise projection $\Pi_{[lo,hi]}(z)$."""
    return np.clip(z, lo, hi)


def _predict(
    train_f: np.ndarray, train_l: np.ndarray, f: np.ndarray, nuisance: Nuisance
) -> np.ndarray:
    if nuisance == "none":
        return np.full(len(f), float(np.median(train_l)))
    if nuisance != "bounded":
        raise ValueError(nuisance)
    w = huber_fit(train_f, train_l)
    zt = project_to_support(f, train_f.min(axis=0), train_f.max(axis=0))
    return zt @ w[:-1] + w[-1]


def jackknife_residuals(
    ref_feats: np.ndarray,
    ref_loss: np.ndarray,
    test_feats: np.ndarray,
    test_loss: np.ndarray,
    *,
    nuisance: Nuisance = "bounded",
) -> tuple[np.ndarray, np.ndarray]:
    r"""``(reference residuals (r,), test residuals (n_test,))`` under the v2 calibration."""
    r = len(ref_loss)
    res_ref = np.empty(r)
    preds_test = np.empty((r, len(test_loss)))
    for j in range(r):
        keep = np.arange(r) != j
        res_ref[j] = (
            ref_loss[j]
            - _predict(ref_feats[keep], ref_loss[keep], ref_feats[j : j + 1], nuisance)[0]
        )
        preds_test[j] = _predict(ref_feats[keep], ref_loss[keep], test_feats, nuisance)
    return res_ref, test_loss - preds_test.mean(axis=0)


def attribution_gate(z_delta: np.ndarray, z_nuisance_max: np.ndarray, z_fire: float) -> np.ndarray:
    r"""$\tilde z_\delta=z_\delta$ if $z_{\max}\le z_{\rm fire}$ else $\min(z_\delta,0)$."""
    return np.where(z_nuisance_max <= z_fire, z_delta, np.minimum(z_delta, 0.0))
