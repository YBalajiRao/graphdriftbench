---
name: drift-spec
description: Use when implementing or testing any drift injector, generator or detector statistic.
---

# Drift specification (Blueprint §4, §5.1, §6.1, §6.3, §6.4)

Source: docs/Paper1_Blueprint.md. Re-read the cited sections before coding; do not rely on this summary alone.

## Four definitions
Window law P_k(V,A,X,Y) = P(V) P(A|V) P(X|A,V) P(Y|A,X,V).
- **gamma** (entity turnover): P(V) changes.
- **beta** (topology): P(A|V) changes on persistent nodes.
- **alpha** (features): P(X|A,V) changes.
- **delta** (relational-semantic): P(Y|A,X,V) changes (the link rule).
A pure drift changes exactly one factor. Mixed = any non-empty subset.

## Pure-injection recipes
Synthetic dynamic SBM (§6.1), at time tau, magnitude eps:
- alpha: edge-mark mean += eps*d. B, node sets, link rule unchanged.
- beta: B <- (1-eps)B + eps B' (merge/split/densify). Marks and beta_h unchanged.
- gamma: replace fraction eps of nodes with fresh IDs from the same block/feature law.
- delta: B <- P B P^T (permutation over blocks with equal size and row sums) and/or beta_h <- -beta_h.
Semi-synthetic TGB (§6.3), hidden from the learner:
- gamma: remap fraction eps of active sources to fresh IDs consistently.
- delta: permute destinations inside degree-matched buckets (bijection).
- alpha: wiki = shift/rotate edge features by eps*sigma; review = remap ratings for fraction eps.
- beta: dense community burst among persistent nodes, or rewire fraction eps to a sharpened popularity law.

## Purity checks (must be tests)
- gamma, delta: degree distribution (KS p > 0.05) and KPM density-of-states distance within the stationary null band; edge-mark MMD within null band.
- alpha: structure untouched (degree, DoS in null band); mark MMD significant.
- beta: DoS distance significant; edge marks untouched on synthetic.
- delta: block-pair count matrix permuted, multiset of entries unchanged.
Null bands come from >= 20 stationary seeds, never hard-coded.

## Orthogonality audit (§6.4)
For each pure injection c, report the 4x4 matrix M[c, c'] = E[z_{c'} | inject c] of mean standardised statistic shifts, across eps in {0.05, 0.1, 0.2, 0.4} and seeds 0-4. Rows = injected type, columns = statistic. Diagonal-dominant supports Proposition 1; report off-diagonal leakage honestly (as a function of eps). Write to results/orthogonality_audit.csv via a script.
