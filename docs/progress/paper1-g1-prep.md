# Paper 1 — decisions applied, G1 preparation (2026-10-28, second note)

Supplements `step-1.md` and `paper1-baseline-and-g1-readiness.md` (both preserved). **Gate G1 was not run**: three choices that the Blueprint leaves open materially affect the result (§6). Everything below labelled *exploratory* uses seeds that are not G1 seeds and may not be reported as confirmation.

## 1. Git
- `ba84fd5` `fix(repo): track src/gdb/data and configs/data (anchor /data/ ignore rule)` — files: `.gitignore`, `configs/data/tgb.yaml`, `src/gdb/data/{__init__,stream,tgb_loader}.py`. Verified before committing: staged diff listed only those five paths, no cache/data blobs. **Clean-clone check:** `git clone` of that commit → `python -m pytest -m "not slow"` → 69 passed (before this fix the clone could not import `gdb.data`).
- A second commit with the work below follows this note (hash in §8). `181a5e2`, `5e57afc` untouched.

## 2. Decision conflicts with the Blueprint (nothing was overridden silently)
1. **δ magnitude.** Blueprint §6.1 defines δ as $B\to\Pi B\Pi^\top$ (a full block permutation, no ε) and/or $\beta_h\to-\beta_h$, but the §6.1 grid ("ε ∈ {0.05, 0.1, 0.2, 0.4}") and the G1 criterion ("at ε = 0.2") apply one ε to all four types. A full permutation has no magnitude, so **"δ at ε = 0.2" is undefined in the Blueprint**. Applied per your decision: primary δ = full permutation (magnitude-free), mixing kept as a separately labelled `mode="mix"` sensitivity. For G1 this means the δ row is evaluated at the full permutation while α, β, γ are at ε = 0.2: a reading of the Blueprint I want you to confirm (§6, Q3).
2. **z-standardisation.** Blueprint §5.1 specifies a moving-block bootstrap of the reference windows (not stationary-stream calibration). I did not substitute anything. The Blueprint gives no reference length $r$, no bootstrap block length, no number of replicates (§6, Q1).
3. **Blueprint reference corrections (not edited in the Blueprint):** Batagelj–Zaveršnik: the DOI 10.1007/s11634-010-0079-y resolves to *Fast algorithms for determining (generalized) core groups in social networks* (Adv. Data Anal. Classif.), not the title quoted in §15; LwF record is year 2017 online (check volume year).

## 3. δ redefinition (implemented, tested)
- `inject_delta`: `mode="permute"` (default) → $B\to BP^\top$ with $m=1$ whenever active; requires `eps == 1.0` and an abrupt onset (else `ValueError`); `mode="mix"` → $B\to(1-\varepsilon)B+\varepsilon BP^\top$, $\beta_h\to(1-2\varepsilon)\beta_h$. `drift_truth.csv` records `mode` in `resolved`.
- Purity variants (`gdb.synth.purity.VARIANTS`): `delta` = permute (primary), `delta[mix]` (secondary, non-verdict), `delta[+flip]` = permute + homophily flip. **The Step 1 purity tables used mixing at ε = 0.2 and are unchanged (historical, not regenerated: their files were not overwritten).**
- Tests (`tests/test_delta_modes.py`, 5): default permute and its magnitude/onset restrictions; mix labelled; permutation changes the assignment (`perm ≠ id`, `B[:,perm] ≠ B`); singular values, row sums, column sums of $B$ unchanged; block-pair counts permute within 6σ; node sets, mark mean and degree KS unchanged; destination popularity rank-correlation > 0.8 with Pareto popularity (per-node popularity preserved); recurring permute restores the law.
- **Verified invariants (what "preserving" means here):** margins of $B$, spectrum of $B$, block sizes, node sets, mark law, and (in expectation) per-node degree, **given** swapped blocks have equal size and equal weighted column sum (enforced by `delta_permutation`). It is *not* claimed to preserve anything beyond these; in particular the realised graph's isomorphism class is not exactly preserved (finite sample).

### Effect on every available statistic (exploratory; `scripts/delta_effect.py`, injection seeds 7000–7019, null = q95 of 200 stationary streams, seeds 8000–8199; cell = seeds outside the band / 20)

`small_bipartite` (w = 2000):

| variant | deg_ks src | deg_ks dst | DoS W1 | mark MMD | block-pair χ² (labels) | new-node src | new-node dst | SV W1 | HHI dst | HHI src |
|---|---|---|---|---|---|---|---|---|---|---|
| stationary | 1 | 2 | 1 | 1 | 0 | 1 | 0 | 0 | 2 | 2 |
| δ permute | 1 | 2 | 1 | 1 | **20** | 0 | 0 | 0 | 0 | 1 |
| δ permute + flip | 4 | 0 | 2 | 2 | **20** | 0 | 0 | 2 | 2 | 4 |
| δ mix 0.2 | 2 | 3 | 1 | 2 | 20 | 0 | 0 | **15** | 1 | 1 |
| β hub 0.2 | 19 | 20 | 20 | 2 | 20 | 0 | 0 | 20 | 20 | 20 |

`wiki_like` (w = 5000):

| variant | deg_ks src | deg_ks dst | DoS W1 | mark MMD | block-pair χ² (labels) | new-node src | new-node dst | SV W1 | HHI dst | HHI src |
|---|---|---|---|---|---|---|---|---|---|---|
| stationary | 1 | 3 | 0 | 2 | 1 | 1 | 0 | 1 | 0 | 0 |
| δ permute | 0 | 1 | 0 | 2 | **20** | 1 | 0 | 1 | 0 | 1 |
| δ permute + flip | 0 | 3 | 0 | 2 | **20** | 0 | 0 | 2 | 0 | 0 |
| δ mix 0.2 | 2 | 0 | 2 | 1 | 20 | 0 | 0 | 2 | 2 | 0 |
| β hub 0.2 | 10 | 3 | 1 | 2 | 20 | 0 | 0 | 10 | 8 | 2 |

Reading: (i) the primary δ moves **no** label-free structural statistic on either preset (all at the stationary rate); the only statistic it moves is the label-based block-pair χ², which a detector cannot use, so **δ is observable only through the learner's predictions** — consistent with Blueprint §4.2, and it means G1(c) and the δ column of the audit depend entirely on the prequential-loss residual, which does not exist yet. (ii) Mixing at 0.2 leaks into SV W1 (15/20) on `small_bipartite`, not on `wiki_like`; the permute form does not, supporting the choice. (iii) δ vs β are separable on the structural statistics on `small_bipartite` (β: DoS/SV/HHI 20/20, δ: 0–1) but on `wiki_like` β is itself barely visible (DoS 1/20), so there δ-vs-β separation rests on SV/HHI (β 10/20, 8/20) and the residual. (iv) This shows the injector is type-pure on the *tested* statistics; it is not evidence that the detector separates the types (G1).

## 4. Reserved ids as default (implemented, tested)
- `DSBMConfig.id_layout` default is now `"reserved"`; `"legacy"` is retained and tested. Reserved layout: sources own $[0,C_s)$, destinations $[C_s,C_s+C_d)$, $C=\lceil(1+h)n\rceil$ from the config only (`id_headroom` h = 1.0); churn ids are allocated inside the owner's range; overflow raises.
- **Learner-facing representation** (`gdb.data.stream.StreamReader`): `initial_table()` returns the fixed-size node table at time 0 (initial entities' features, zeros elsewhere, `role`, `initial`); `batches(size)` yields events plus the features of entities appearing for the first time in that batch. In the container, fresh entities' features live in `late_ids/late_feat` and the `node_feat` rows for them are zero. These arrays still exist in the offline container (an in-memory object cannot hide its own contents), so safety rests on the rule that `gdb.cl`/`gdb.detect` only use `StreamReader`; `tests/test_no_leakage.py` now greps those packages for `late_ids`, `late_feat`, `node_initial`, `.node_feat`. This is a source-level guard, not a runtime guarantee.
- Tests (`tests/test_id_layout.py`, 17 total): ranges disjoint under src/dst/both churn; table shape, role and initial flags identical across no churn, ε = 0.1 and ε = 0.4 for each churn side; initial ids never remapped; reader releases an entity's features only in the batch of its first event, never before onset, and exactly the set that appears; raw `node_feat` holds no fresh features; headroom overflow refused; mixed and recurring drift on the reserved layout; determinism; legacy hash pinned (`1e04b330…`, seed 0, γ+α) and default-layout hash differs.
- **TGB loader unchanged:** `load_tgb` returns `node_initial=None`, dense TGB ids (sources < 8227 ≤ destinations < 9227 asserted in the slow loader test, which passed).
- **Migration:** reserved streams are not hash- or file-compatible with legacy ones (destination ids shift by $C_s-n_{src}$, new fields in `stream.pt`; `load` of old files still works because fields are optional). All Step 1 example outputs (`data/synth`, git-ignored) are legacy. Purity statistics are invariant to the relabelling (it is order-preserving within each side). Existing legacy-convention tests now request `id_layout="legacy"` explicitly.

## 5. Learnable streams for G2 (preset `configs/synth/g2_learnable.yaml`; exploratory, development seeds 0–4)
`scripts/g2_learnability.py` → `results/exploratory/g2_learnability.csv` (CUDA, 20 epochs, lr 3e-3, held-out last 15%, 5 seeds, mean ± sd):

| preset | block-oracle AP | TGN AP, no node features | TGN AP, node features |
|---|---|---|---|
| `small_bipartite` (original) | 0.659 ± 0.004 | 0.502 ± 0.005 | 0.665 ± 0.009 |
| `g2_learnable` (p_out 0.02, σ_node 0.3, β_h 2) | 0.774 ± 0.008 | **0.602 ± 0.084** | **0.790 ± 0.007** |

- Predictive signal exists before drift and is above chance with node features on both presets; it is strongest and tightest on `g2_learnable`.
- **Correction of earlier claims:** the audit note said a featureless TGN "stays at chance". On `g2_learnable` it does not (0.60 ± 0.08, seed-dependent) and in one slow-test run with the new id layout reached 0.68. Chance-level behaviour on the *original* preset is reproduced (0.502). The G2 design should therefore state that the learner receives node features (via the reader) and not depend on a featureless baseline being at chance. I changed the slow test to assert only "with features > 0.68 and > featureless".
- The oracle ceiling ignores the homophily term, so TGN-with-features can exceed it; this is not a Bayes bound.
- Factor checks on `g2_learnable` (5 seeds): γ(0.2, both sides) keeps the active-node counts constant (300/300 sources, 150/150 destinations before and after), i.e. turnover replaces entities rather than shrinking or growing the pool; δ's popularity check is **uninformative** on this preset (no popularity heterogeneity: rank correlation ≈ 0.00 ± 0.07 is noise); the popularity-preservation test uses the Pareto preset (`tests/test_delta_modes.py`, ρ > 0.8). Mixed and recurring drift are tested for ids/onset/restoration, not yet for learner-level effects.
- Not done: a learner-level check that δ actually degrades a trained TGN's AP (needs the Step 4 loop); AP numbers are development numbers, not benchmark results.

## 6. Gate G1: reconciliation with the Blueprint and the choices that block the run
**Blueprint G1 (§13), exactly:** "Can the four raw statistics tell the four pure drift types apart on synthetic streams?" Pass: 4×4 audit matrix diagonal-dominant; simple type classifier (argmax of standardised z) macro-F1 ≥ 0.80 at ε = 0.2; δ residual does not fire on pure α/β/γ (≤ 15% of seeds). Fail: redesign the weak statistic, narrow the claim, do not build replay.
**Not part of G1:** CUSUM, Bonferroni/ARL₀, detection precision/recall/delay, mixed drift. They belong to Step 3 (§13) and the detector evaluation (§7.2); I will report detection F1/delay/false alarms only where they can be defined (they cannot without a CUSUM), and I do not stretch G1 to include them. Your list of reported items is therefore split: *required for G1*: audit matrix, macro-F1 + confusion matrix, δ false-fire rate, per type/preset, seed-level values, runtime/memory; *deferred to Step 3*: precision/recall/F1 with tolerance, delay, false alarms/10⁵ events, mixed-drift type sets.

**Fixed independently of your answers** (can be locked now): presets `small_bipartite` (primary development/reference) and `wiki_like` (required stress run, reported even if it fails); reserved id layout; α, β[hub, boost 10], γ[src] at ε = 0.2, δ[permute]; window sizes from the presets (2000 / 5000); locked statistics (γ: max of standardised Jaccard distance and new-node rate; β: DoS W1, 100 moments, 200 probes; α: degree-stratified linear MMD; δ: residual per §5.1); non-finite value ⇒ run recorded as failed, not dropped; tied argmax counted as misclassified; free VRAM checked via `mem_get_info`, CPU fallback below 2.5 GB. **Seeds** (none used so far; will be frozen before any official statistic is computed): calibration/bootstrap-reference streams 3000–3199, official evaluation 2000–2019 per type; development 0–4, exploratory 7000–8199, purity 0–19/1000+ are excluded.

**Choices the Blueprint leaves open and that change the outcome — need your decision:**
- **Q1 — reference and bootstrap.** Blueprint: "last $r$ windows since the last alarm", moving-block bootstrap for $\mu_c^R,\sigma_c^R$; $r$, block length, replicates unspecified. Smallest alternatives: **(A)** $r$ = 10 windows, block length 2, 500 replicates (faithful to §5.1; needs 11 windows of stationary data per stream, longer streams), **(B)** stationary-calibration z (cheaper; a deviation from §5.1). Recommendation: A; G1 is not a place to save compute.
- **Q2 — what a "test case" is.** One statistic vector per drifted stream from the **first post-onset window**, or from the mean z over the first 3 post-onset windows. Recommendation: first window (strictest, matches "detect at onset"); also report the 3-window mean as a non-gating sensitivity.
- **Q3 — δ's ε.** Confirm: δ row = full permutation (no ε), α/β/γ at ε = 0.2, mixing 0.2 reported as secondary.
- **Q4 — δ residual learner.** The residual needs a learner with a link rule it can learn. Alternatives: **(A)** feature-aware TGN (prequential, `g2_learnable`, node features via the reader), faithful, cost ≈ 15–30 s/stream × ≈ 300 streams on the RTX 3070 (≈ 1.5–3 h; extrapolated from the smoke timing, not measured for this protocol); **(B)** a cheap frozen scorer as a G1-only proxy (deviation). Recommendation: A. Also unspecified: Huber settings for $\mathbf w$ (default scikit-learn-style ε = 1.35) and whether $z_\alpha,z_\beta,z_\gamma$ enter as the same statistics used for the audit.
- **Q5 — "none" class.** Macro-F1 over the four drift types only (the Blueprint says "drift types"), with stationary streams used only for the δ false-fire count and a separate false-alarm report. Recommendation: that.
- **Q6 — preset parameters for the learner.** `g2_learnable` (sharp block, informative node features) as the G1 reference preset rather than `small_bipartite` for the δ column, since the δ residual needs a learnable rule while the other three statistics are label-free. This differs from your "small_bipartite as primary" unless you accept it; the pure-structure statistics can still be computed on `small_bipartite`.

Until Q1–Q6 are answered, running G1 would require me to choose them after having seen related exploratory results, which is what you asked me not to do. The exploratory results so far already show: β via DoS is expected to be weak on `wiki_like` (1/20), and the pure δ is invisible to structural statistics, so a G1 failure on δ-or-β is a realistic outcome that must be reported as such.

## 7. Baselines: literature status (no change to the previous matrix)
- DOIs resolved to the expected titles in this and the previous session: ER-GNN (10.1609/aaai.v35i5.16602), SSM (10.1109/ICDM54844.2022.00177), TEM (10.1145/3637528.3671732, *Topology-aware Embedding Memory for Continual Learning on Expanding Networks*), TWP (10.1609/aaai.v35i10.17049), ContinualGNN (10.1145/3340531.3411963), CaT (10.1109/ICDM58522.2023.00141), Malialis (10.1109/IJCNN60899.2024.10650724), EWC, LwF, Page, Lorden, SCPD.
- **Full texts were NOT read.** I downloaded the ER-GNN PDF from the AAAI site but cannot extract text: `pypdf`/poppler are not installed and installing needs your approval; SSM, TEM and ContinualGNN are behind publisher paywalls (IEEE/ACM). A first attempt to fetch TEM by a guessed arXiv id returned an unrelated paper; I discarded it and will not guess identifiers. All statements about those methods' internals in the previous note remain **[verify]**. Plan once approved: install `pypdf` (pure Python) or you supply the four PDFs (university VPN, `acquire-sources` worklist), then write a per-method definition note before any implementation.
- Decisions applied: keep TGN primary / DyGFormer second; type-agnostic structural replay at equal memory and replay steps (the decisive control); fine-tune, reservoir ER; periodic (matched refresh count) and always-on refresh controls; oracle-trigger run as synthetic-only labelled diagnostic implemented outside `gdb.cl`/`gdb.detect`; TWP to appendix **only if** the adaptation is judged too costly and the decision is recorded (not decided now); CaT/PromptGNN/OTGNet discussion-only after relevance is verified; adapted baselines named "-style", never "reproductions".

## 8. Tests, lint, repository state
- `python -m pytest`: first full run 104 passed, 1 failed (the learnability test's assumption that a featureless TGN stays below 0.56, shown false above); after correcting the test (§5) it passes. Final full-suite count is reported in the chat summary.
- Ruff: clean on `src`, `scripts`, `tests` except the E731 lambda in `tests/test_metrics_audit_step1.py` (auditor's file, left as written).
- Not run on this diff: code-reviewer / metrics-auditor agents (not requested here); `StreamReader`, SV/HHI and the δ modes have only my tests.
- Exploratory outputs: `results/exploratory/*.csv` (+ per-script logs). No earlier result file was overwritten; `results/purity_*` are the Step 1 (mixing-δ) files.

## 9. Unresolved risks / failed criteria
- Step 1 purity failures (β via DoS in all presets; unipartite δ DoS 16/20) stand; they were produced with the then-primary mixing δ and are not re-run.
- Expected G1 risks: β separability on `wiki_like` via DoS (1/20 exploratory); δ is label-free-invisible by design, so G1 hinges on the residual; the learner's features and id layout are not yet used by any detector.
- Leakage guard for the reader is a convention plus a grep, not an enforced runtime barrier.
- Single-preset learnability numbers (5 seeds); no learner-level evidence that δ/β/γ/α hurt a trained TGN differently.
- Baseline definitions unverified against full texts.

## 10. Next experiment recommended
Answer Q1–Q6, approve a PDF reader install, then (1) build the prequential-loss pipeline on `g2_learnable` and check at the learner level that δ[permute] raises loss and α/β/γ do not (a development check on seeds 0–4), (2) freeze the G1 protocol and seeds in a signed-off note, (3) run G1 once and report whatever happens.
