# Gate G1 — official results (2026-10-28)

Protocol: `docs/progress/g1-protocol.md`, locked in commit `5e240d9` before any official stream ran. Official seeds 2000–2019 (α, β, γ, δ, δ[mix]) and 2100–2119 (stationary). Nothing in the protocol was changed after the first official row. All 340 official streams ran (0 failed, 0 NaN, 0 re-runs), all on CUDA (≥ 6.66 GB free before each run, peak ≤ 0.22 GB). Raw rows: `results/g1/official/<preset>/*.json`; scored summaries: `results/g1/official/<preset>_summary.{json,md}` (full audit matrices with CIs, confusion matrices, three-window sensitivity).

## Verdict
**G1 is NOT passed.** The required configurations P1 and P2 both fail; the optional P3 also fails. Per Blueprint §13 the consequence is: redesign the weak statistics, narrow the taxonomy claim, and do not build replay yet.

| config | streams | (a) diag-dominant rows | (b) macro-F1 ≥ 0.80 (95% CI) | (c) δ fires on pure α/β/γ ≤ 15% | pass |
|---|---|---|---|---|---|
| P1 `small_bipartite` (primary) | 120 | PASS (4/4 rows) | PASS 0.938 (0.88–0.99) | **FAIL 18.3%** (α 10%, β 5%, γ 40%) | **FAIL** |
| P2 `wiki_like` (required stress) | 100 | **FAIL** (β row: argmax is γ) | **FAIL 0.641 (0.59–0.68)** | **FAIL 28.3%** (α 20%, β 5%, γ 60%) | **FAIL** |
| P3 `g2_learnable` (additional) | 120 | PASS (4/4) | PASS 1.000 (1.00–1.00) | **FAIL 23.3%** (α 0%, β 10%, γ 60%) | **FAIL** |

Each configuration was scored separately; no pooled F1 was computed.

## Audit matrices (mean z of statistic given injected type, T₀; CIs in the summary files)
P1: α row [4.42, 0.73, 0.45, −0.16]; β row [−0.44, 7.84, 0.49, −3.60]; γ row [0.06, **46.15**, 105.47, 1.88]; δ row [−0.10, 0.41, 0.29, 10.37].
P2: α [14.91, −0.05, −0.05, −1.66]; β [0.04, **0.24**, **5.59**, −0.61]; γ [0.22, 22.40, 130.03, 7.67]; δ [0.26, −0.03, 0.47, **3.24**].
P3: α [4.96, 0.03, 0.49, −0.37]; β [0.18, 9.92, 1.39, −1.36]; γ [0.57, 40.23, 100.44, 7.22]; δ [0.05, 0.65, 0.87, 29.86].
(The γ column is large because of the pre-registered σ floor; compare only within rows.)

## Confusion matrices (argmax, rows = true; order α β γ δ)
P1: α [19,1,0,0]; β [0,20,0,0]; γ [0,0,17,3]; δ [0,1,0,19]. P2: α [20,0,0,0]; β [0,**0**,**20**,0]; γ [0,0,20,0]; δ [0,1,2,17]. P3: perfect (20 on each diagonal).
Per-type F1: P1 α 0.97, β 0.95, γ 0.92, δ 0.90; P2 α 1.00, **β 0.00**, γ 0.65, δ 0.92; P3 all 1.00.

## Detection, delay, false alarms (z > 3 at T₀; supplementary; missed detections shown)
| config | precision | recall | missed (of 80) | stationary false-alarm streams (of 20) | first detected at T₀/T₁/T₂ | never within 3 windows | false alarms /10⁵ events | thresholded macro-F1 | δ-stat AUROC |
|---|---|---|---|---|---|---|---|---|---|
| P1 | 0.986 | 0.863 | 11 | 1 | 69/5/4 | 2 | 0.83 | 0.868 | 0.889 |
| P2 | 0.970 | 0.812 | 15 | 2 | 65/3/1 | 11 | 2.33 | 0.568 | 0.769 |
| P3 | 0.962 | 0.950 | 4 | 3 | 76/2/1 | 1 | 5.00 | 0.972 | 0.891 |
Missed detections concentrate in α on P1 (7 of 20 streams with no statistic above 3: the thresholded confusion matrix has them in the `none` column), in δ on P2 (12 of 20) and in β on P2 (3 of 20 detected as γ, 17 of 20 labelled γ by argmax). The mean delay over detected streams is 0.17 windows (P1), 0.07 (P2), 0.05 (P3); it is conditioned on detection at the first window and is therefore not informative about the missed cases. The three-window sensitivity changes macro-F1 to 0.950 (P1) and leaves the pass/fail outcomes unchanged (summaries). ε severity (0.05/0.1/0.4) was not run (not part of G1 and not in the locked protocol).

## What failed, and what the data say about why
1. **(c) The δ residual fires on γ streams (40% / 60% / 60%) in all three configurations.** On P1 the mean prequential loss of γ streams is unchanged (1.27 vs reference 1.26) while $z_\delta$ ranges from −130 to +165 (γ rows, P1): the residual is not responding to a loss increase; it is the Huber regression, fitted on reference $z_\gamma\approx0\text{–}2$, extrapolating to a test $z_\gamma\approx105$ (floor-set scale). This is a specification property of §5.1's $\hat\ell=\mathbf w^\top[z_\alpha,z_\beta,z_\gamma,1]$, amplified by my disclosed σ floor; I did not verify that a bounded regression would fix it (not tested, not tuned). It means the δ residual in its literal form is **not γ-invariant**, so Proposition 1's claim "the δ residual is unchanged under pure α/β/γ to first order" is not supported by these data. In P2 it also fires on α in 20% of streams.
2. **β leaks under γ (z_β = 46, 22, 40).** On γ streams the persistent subgraph holds exactly 80% of the reference event count (`pers_events_w0` / reference mean = 0.80 on P1 and P2), consistent with the proposed mechanism: removing churned nodes sparsifies $G_a[P]$ and shifts its spectrum. The persistent-node definition therefore does not make $s_\beta$ invariant to γ in this implementation. A size-matched variant was not run.
3. **β is not identified on P2.** DoS under β[hub] has mean z 0.24 (essentially the null), reproducing the exploratory finding (1/20) on official seeds; β streams are classified as γ (20/20) because the hub injection shifts the new-node rate (z_γ = 5.59) while DoS is blind. This is the failure on the realistic preset: β[hub] is **not** detected by the locked β statistic, and the type is mis-assigned.
4. **δ is only weakly identified on P2:** mean $z_\delta$ = 3.24, AUROC 0.77, 12 of 20 δ streams below the fire threshold. The TGN reaches reference loss ≈ 1.10 (heavy-tail popularity, little block structure learned), which is the learnability limitation seen in development. On P1 and P3 the δ residual responds strongly to δ (10.4 and 29.9) and to δ[mix, ε = 0.2] (secondary: P1 mean $z_\delta$ 4.09, fires in 60% of streams; P3 10.62, 95%), consistent with the permutation being the stronger perturbation. Whether the δ response reflects relational-semantic change rather than generic loss increase: on stationary streams it stays at the null (P1 stationary false alarms 1/20) and on α/β it is negative or near zero, so it does not respond to generic structural/feature changes there, but it does respond (erratically) to γ for the extrapolation reason above. **I do not claim successful δ identification**: the residual identifies δ against α, β and stationary on P1/P3, fails the γ control, and is weak on P2.
5. **γ is identified (rows pass) but its z-scale is set by the floor**, and 3 of 20 γ streams are labelled δ on P1 (0 on P2, P3) because $z_\delta$ is erratic there; on P2, 2 of 20 δ streams are labelled γ.
6. **α**: P2 α detection is strong (14.9); on P1 it is detected at z 4.4 but 7 of 20 α streams stay below the fire threshold (thresholded matrix), i.e. α is a weak-power case at ε = 0.2 with the 32-d marks.

## What passed, and what it does and does not mean
P1 and P3 pass (a) and (b): on the dense, learnable configurations the argmax of the four statistics identifies the injected type in 94–100% of streams, and each row's maximum is the diagonal entry. This is evidence that the four statistics carry type information in an easy setting. It is **not** evidence of detector validity: (c) fails everywhere, P2 fails (a) and (b), purity of the injectors is a separate question, and the Blueprint's CUSUM/FWER detector (Step 3) has not been built or evaluated.

## Deviations and fills (all in the protocol; repeated for the record)
r = 10 windows, block length 2, 500 replicates (your values; the Blueprint gives none); leave-one-out reference series; pool of 9 windows for both reference and test statistics; σ resolution floor for the two γ components (not in the Blueprint; needed because the new-node rate is identically 0 on dense windows — the first 15 development streams all failed without it); Huber(α = 1e-4) with leave-one-out standardisation of the δ residual; β target `hub` (boost 10); fire threshold z > 3; 200 KPM probes. Development seeds were used to find the zero-variance failure and to look at the γ behaviour before the lock; none of the constants was tuned on them.

## Unresolved risks and limitations
- δ-residual and β definitions as written in the Blueprint are not γ-invariant (items 1–2); fixing them is a **method change** (bounded/clipped regression features, size-matched persistent subgraphs, or a different γ-invariant β) that needs your approval and **fresh confirmatory seeds** — seeds 2000–2019/2100–2119 are now used for G1 and may not be reused to claim validation of a revised statistic.
- The floor affects the magnitude of $z_\gamma$ and indirectly $z_\delta$; its effect was not isolated.
- P2 realism limitation: the TGN barely learns block structure there, so δ cannot be identified regardless of the statistic's quality; a "learnable wiki-shaped" configuration (sharper blocks, informative node features, same size and heavy tails) is an alternative to P2 that would need your approval; I did not substitute it.
- Single learner configuration (TGN, memory 32, one hyperparameter setting, 30 warm-up windows); the learning curve is still falling slowly during the reference windows on P1 (loss trend inflates the residual σ, which is conservative for δ).
- Window-level statistics only for ε = 0.2 and abrupt onset; mixed/recurring/gradual drift and severity curves are untested.
- Hypotheses: H1 (type macro-F1) cannot yet be assessed (no CUSUM detector, no baseline detectors); H5 (post-δ plasticity) untestable before δ identification is fixed; the Blueprint's orthogonality claim is partly refuted for β↔γ and δ↔γ in this implementation.

## Recommendation (evidence-based, not yet approved)
1. Do **not** start Step 3/4 on the current statistics. Follow §13's failure branch.
2. Decide among: (A) revise δ-residual (bounded features / no extrapolation) and β (size-matched or γ-orthogonalised persistent subgraph), re-run G1 on **new seeds** with a new locked protocol version; (B) narrow the claim to three identifiable types plus a γ-conditional δ; (C) keep the statistics and report the leakage as a limitation (not recommended: G1 says redesign).
3. Replace or supplement P2 by a learnable wiki-shaped stress configuration only with your approval; keep P2 in the paper as the realistic negative case.
4. Before any of that: supply the SSM, TEM and ContinualGNN PDFs (`docs/progress/baseline-fulltext-notes.md`), because the baseline definitions are still unread.

## Reproduction
`python scripts/run_g1.py --mode official --preset <p> --drifts none,alpha,beta,gamma,delta[,delta_mix]` (resumable, never overwrites) then `python scripts/score_g1.py --mode official --preset <p>`. Dev artefacts: `results/g1/dev`, `results/g1/dev_failed_v0` (first, failed attempt with the zero-variance error).
