# Paper 1 — baseline strategy, Step 1 audit and Gate G1 readiness (2026-10-28)

Status: **audit done, safe corrections made and tested; Gate G1 was NOT executed** (§7 explains which decisions block it). No Step 2 work, no full training. `step-1.md` is untouched; this note supplements it.

Evidence labels: **measured** = produced by a script in `scripts/`, files in `results/exploratory/`; **cited** = DOI resolved through OpenAlex in this session (existence, title, authors, year only; I did not read full texts, so statements about what a paper's method does are **[verify against full text]**); **proposed** = my recommendation, not tested.

## 1. Current project state

- Branch history: `5e57afc` (setup), `181a5e2` (Step 1). Working tree has uncommitted changes from this audit (listed in §6); nothing was committed, amended or reset.
- **Defect in the Step 1 commit (fixed in the working tree, not yet committed):** `.gitignore` line 1 was `data/`, which also matched `src/gdb/data/`. `stream.py`, `tgb_loader.py` and `__init__.py` were therefore **never committed** in `181a5e2`; a fresh clone could not import `gdb.data`. The rule is now `/data/` (repo-root only), and `src/gdb/data/` shows as untracked. It needs a commit; I did not commit without being asked.
- Implemented and tested (Step 1): generator, four injectors, schedules, purity machinery, graph statistics, TGB loader, TGN smoke. **Not implemented:** the evaluation protocol (Step 2), the detector (Step 3: z-standardisation, bootstrap, CUSUM, Bonferroni, δ residual), memory/policy/CSEE (Step 4), every baseline (Step 5). `src/gdb/{cl,detect,memory,eval,inject}` are empty stubs.
- Fast suite after this audit: 77 passed (69 + 6 id-layout + 2 candidate-statistic tests); full run in §6.

## 2. Baseline strategy

### 2.1 Separate the two comparisons
- **Backbone comparison (context row only, Blueprint §9E):** TGN (primary), DyGFormer (second backbone), plus JODIE/DyRep/TGAT/GraphMixer as no-CL context. Backbones are compared with *no* CL strategy, on the same chronology, negatives and seeds. They never appear in CL-strategy tables.
- **CL-strategy comparison (all claims H1–H5):** one fixed backbone per table (TGN first; DyGFormer repeated only after G2), same stream, same negatives, same seeds 0–4, same memory budget M ∈ {1,2,5}% of stream events.

### 2.2 Baseline matrix

Memory budget M is in *events* for every replay method; a method that stores ego-nets, embeddings or condensed graphs is charged the same byte-equivalent (events × (2 ids + t + d_e floats)); this charging rule must be fixed before Step 5. "Impl." = whether code exists (none does).

| # | Method | Scientific purpose | Reference (resolved this session) | Directly applicable to CTDG link prediction? | Adaptation needed (changes the original?) | Impl. | Budget / cost | Likely reviewer objection |
|---|---|---|---|---|---|---|---|---|
| 1 | Fine-tune (no replay) | forgetting lower bound | none needed | yes | none | to build | M = 0 | none |
| 2 | Joint retraining on all past data | upper bound | none needed | yes | none | to build | unbounded; label as non-comparable in budget terms | upper bound ignores budget |
| 3 | Reservoir ER (random events) | equal-budget random memory; the null for structure-aware selection (H4, A3) | Vitter 1985 reservoir sampling **[verify DOI]** | yes | none | to build | M events; replay steps matched to CSEE | "unstructured control is too weak" → also run type-agnostic structural (#6) |
| 4 | **ER-GNN** (Zhou & Cao, AAAI 2021) DOI 10.1609/aaai.v35i5.16602 — title verified: *Overcoming Catastrophic Forgetting in Graph Neural Networks with Experience Replay* | structure/feature-based node selection for a replay buffer | cited ✓ | published for **node classification with task boundaries** | its buffer-selection rule (representative/coverage-style node selection **[verify against full text]**) must be applied to *events or ego-nets* of a TGN; loss = TGN BCE on replayed events. This is a different method from the original (task: link prediction, no task ids) → report as "ER-GNN-style selection" | to build | M events | "not the original ER-GNN" — state it in the appendix, as Blueprint §9B already requires |
| 5 | **SSM** (Zhang, Song, Tao, ICDM 2022) DOI 10.1109/ICDM54844.2022.00177 — title verified: *Sparsified Subgraph Memory for Continual Graph Representation Learning* | closest published ancestor of our unit (sparsified subgraph memory); isolates whether *k-core centre selection + drift-triggered refresh* beats plain sparsified subgraph memory | cited ✓ | node classification, task-incremental | store sparsified ego-subgraphs around selected events' endpoints, sparsifier as published **[verify against full text]**; budget charged per stored edge | to build | M events | "SSM is the closest competitor; is your sparsifier the published one?" → implement its published rule exactly, change only the target loss |
| 6 | **TEM / PDGNN** (Zhang, Song, Chen, Tao, KDD 2024) DOI 10.1145/3637528.3671732 — verified title: *Topology-aware Embedding Memory for Continual Learning on Expanding Networks*. (The Blueprint's "TEM-adapted" refers to this.) | memory of node embeddings that encodes topology, avoiding stored subgraphs | cited ✓ | node classification on expanding (snapshot) networks | The paper's central device is a specific GNN (PDGNN, **[verify]**) that cannot be swapped into TGN. A TGN adaptation can only keep the *idea* (replay stored node embeddings/memory states, no neighbourhood). That is a **substantial change**: label it "TEM-style embedding replay" and never "TEM" in the main table | to build | M embeddings (charged by bytes) | "this is not TEM" → true; keep it in the appendix unless a faithful port is feasible |
| 7 | **Type-agnostic structural replay** (ablation A2) | tests whether *type conditioning* contributes beyond detection + k-core memory; the key G2 comparison | internal | n/a | same detector, same memory, same replay steps, one fixed policy for every alarm | to build | M, **equal replay gradient steps** as CSEE | none if steps are matched |
| 8a | **Periodic refresh** (A1) matched to CSEE's alarm count | tests whether *when* we refresh matters, at equal refresh count | internal | n/a | refresh at evenly spaced windows with the same number of refreshes as CSEE on that stream | to build | M, equal refresh count | none |
| 8b | **Always-on refresh** (A1, H3) | upper compute reference for H3 ("≥50% fewer replay steps") | internal | n/a | refresh every window | to build | M, most compute | none; report as a compute reference, not a competitor |
| 9 | **Oracle-trigger diagnostic** (synthetic only) | separates detector error from policy value: same policy with true alarm times/types | internal | n/a | alarms and types injected from ground truth by the evaluation harness | to build | M | misuse: must be reported only as a labelled diagnostic, in a separate table, never in a fair-comparison table or the headline claims; implement outside `src/gdb/cl` and `src/gdb/detect` so the leakage test stays meaningful |
| 10 | Loss-PH / ADWIN-triggered ER (Malialis-style) DOI 10.1109/IJCNN60899.2024.10650724 — verified title: *Incremental Learning with Concept Drift Detection and Prototype-based Embeddings for Graph Stream Classification* | generic single-detector trigger; tests H1/A5 "typed vs generic detector" | cited ✓ | graph-stream **graph classification**, prototype embeddings | only the idea (loss-based detector triggers an update) is reused; it is a re-implementation of an idea, not of the paper | to build | M | name it "drift-triggered ER (loss detector)" |
| 11 | EWC (PNAS 2017, DOI 10.1073/pnas.1611835114 ✓), LwF (DOI 10.1109/TPAMI.2017.2773081 ✓; the record is **2017** online, the Blueprint says 2018 — check the volume year) | regularisation family | cited ✓ | yes (loss-level) | Fisher over TGN parameters; LwF needs outputs on old data = the current stream, so adaptation is non-trivial **[verify]** | to build | no replay memory; parameters extra | none |
| 12 | TWP (Liu, Yang, Wang, AAAI 2021) DOI 10.1609/aaai.v35i10.17049 ✓ *Overcoming Catastrophic Forgetting in Graph Neural Networks* | topology-aware regularisation | cited ✓ | node classification; relies on attention/aggregation importance | needs a topology-aware gradient on a message-passing layer; TGN uses one TransformerConv, so feasible but expensive | to build | extra parameter memory | **my recommendation:** appendix only; EWC already represents the regularisation family and TWP's port is a large change (decision in §9) |
| 13 | ContinualGNN (Wang et al., CIKM 2020) DOI 10.1145/3340531.3411963 ✓ *Streaming Graph Neural Networks via Continual Learning* | streaming-graph CL with replay and EWC-style regularisation **[verify against full text]** | cited ✓ | streaming graph, not CTDG memory models | adapt buffer + regulariser to TGN | to build | M | include if budget allows; the Blueprint lists it, I keep it |
| 14 | CaT, PromptGNN, OTGNet | graph condensation / prompts / open-world | CaT ICDM 2023 ✓ (DOI 10.1109/ICDM58522.2023.00141); PromptGNN, OTGNet **not verified** | doubtful for edge streams | condensation of an event stream is not defined by the original | no | n/a | **recommend: discuss only**, as the Blueprint allows for CaT; PromptGNN only if public code exists |

**Detector baselines (Blueprint §9D, H1):** LAD, SCPD, Fréchet-statistics CPD, LEM-CPD, loss-PH/ADWIN. SCPD DOI 10.1007/978-3-031-33374-3_2 verified (*Fast and Attributed Change Detection on Dynamic Graphs with Density of States*, the Blueprint's "LNCS 2023"). These produce *when*-alarms only; for H1 (type macro-F1) they need a documented type-assignment rule, or H1 must be restricted to detection F1 for them. This is an open design point (§9).

### 2.3 Recommended core suite (smallest set that tests each contribution)

| Contribution | Needed comparison |
|---|---|
| C2 detector | CSEE detector vs loss-PH/ADWIN, and vs a single-statistic detector built from each of its four statistics (A4); SCPD for β as the closest published DoS detector |
| C3 type-conditioned replay | #7 (type-agnostic, equal steps) — **the decisive control (G2)**; #8a periodic, #8b always-on; #9 oracle diagnostic on synthetic only |
| C3 structure-aware memory | #3 reservoir (A3), #5 SSM |
| Context for a CL reviewer | #1, #2, EWC, #4 ER-GNN-style, #6 TEM-style (appendix), #10 loss-triggered ER |
| C4 backbones | TGN primary; DyGFormer only after G1 and G2 |

Not recommended now: TWP (appendix), CaT/PromptGNN/OTGNet (discussion), the optional extra datasets.

### 2.4 Fair-comparison protocol (to be locked before Step 5)
Same chronological stream and the same ten *evaluation-only* segments; same official TGB negative samples (historical + random) for every method and seed; same TGN hyperparameters except the CL-specific ones, which get an equal tuning budget on validation segments only; same M; equal replay gradient steps in every comparison that claims compute-neutrality (G2, A2) and replay steps *reported* in every other table; seeds 0–4; Wilcoxon (paired across seeds × streams) for CSEE vs the best baseline, Friedman + Nemenyi across all methods; Holm correction over the hypothesis family (H1–H5). Peak GPU memory, throughput and adaptation time reported per method. **No numbers from other papers.**

**Limits of this recommendation:** I could not read full texts, so claims about what ER-GNN, SSM, TEM and ContinualGNN do internally are from memory and flagged [verify]. Their exact selection rules must be read from the papers before implementation.

## 3. Step 1 findings, separated

### 3.1 Defects
1. `.gitignore` `data/` swallowed `src/gdb/data/` (above). **Fixed in the working tree.**
2. `configs/data/tgb.yaml` was also never committed (same rule).
3. The Step 1 test suite did not check what the learner sees: `node_feat` has rows for ids that appear only later, and the table size depends on ε (§3.5).
4. δ with ε < 1 is not isomorphism-preserving (§3.3). That is a property of my Step 1 convention, not a coding error, but the Step 1 report described δ as "permutation, spectra unchanged" without saying this holds only at ε = 1.

### 3.2 Injector purity vs statistical power vs Gate G1 (three different claims)
- **Injector purity** (did only the intended factor change?) was checked on *block-pair counts, degree KS, marks and node sets*: Step 1 passed those cells except as listed in `step-1.md`. It was **not** checked on spectral statistics for δ at ε < 1, where it fails (§3.3).
- **Statistical power** (does a given statistic see the intended change?) is statistic-specific: measured in §3.4.
- **Gate G1** (can the *detector's* statistics separate the types?) has not been run.

### 3.3 δ at ε = 0.2 is not spectrum-preserving (new, measured)
`scripts/eps_effective.py` → `results/exploratory/eps_effective.csv`. For the block law $p_{ab}\propto B_{ab}n_an_b$ and the singular values of the scaled block matrix, at ε = 0.2 (small_bipartite):

| injector | TV of block-pair law | spectral change of B |
|---|---|---|
| β[hub, boost 10] | 0.142 | 0.100 |
| β[merge] | 0.030 | 0.032 |
| β[densify, p = 4] | 0.062 | 0.030 |
| δ (mixing) | 0.117 | **0.116** |
| δ at ε = 1 (full permutation) | 0.586 | **0.000** |

So at ε = 0.2 the mixing-δ changes the block spectrum as much as β[hub] does, and only the full permutation (ε = 1) leaves it exactly unchanged. Consistently, a singular-value statistic fires on δ at ε = 0.2 in 15/20 exploratory seeds on `small_bipartite` (§3.4). This contradicts Blueprint §6.1 ("graph law preserved up to isomorphism") for fractional δ and will matter for the G1 diagonal-dominance criterion. The DoS statistic did *not* fire on δ (1/20), so the locked statistic is unaffected on these data. Because the effective perturbation sizes differ across injectors (TV 0.03 for β[merge] vs 0.14 for β[hub] at the same nominal ε), **nominal ε is not comparable across types**; I did not rescale anything.

### 3.4 β: why DoS is weak, and what else sees it (exploratory, seeds disjoint from all earlier ones)
Setup (`scripts/beta_diagnostic.py`, prespecified before running): candidates dos_w1 (locked: 100 moments, 200 probes), sv_w1 (W1 between scaled singular-value spectra of the window biadjacency matrix, a LAD-style statistic = Blueprint ablation A5), and the absolute change of the destination/source Herfindahl concentration index. Null band = q95 of 200 stationary streams (seeds 8000–8199); 20 injection seeds 7000–7019; ε = 0.2 and window sizes unchanged. These seeds are disjoint from the purity seeds (0–19), purity null seeds (1000+), the earlier exploration (100–199, 5000+, 6000+) and are **not** the G1 seeds. Cell = seeds outside the band / 20.

`small_bipartite` (w = 2000; mean cost per window pair: DoS 0.084 s, SV 0.006 s):

| variant | dos_w1 | sv_w1 | hhi_dst | hhi_src |
|---|---|---|---|---|
| none (false alarm) | 1/20 | 0/20 | 2/20 | 2/20 |
| β[hub] | 20/20 | 20/20 | 20/20 | 20/20 |
| β[merge] | 1/20 | 2/20 | 0/20 | 4/20 |
| β[densify] | 0/20 | 6/20 | 3/20 | 2/20 |
| α | 1/20 | 1/20 | 0/20 | 1/20 |
| γ[src] | 0/20 | 2/20 | 3/20 | 0/20 |
| δ | 1/20 | **15/20** | 1/20 | 1/20 |

`unipartite_small` (w = 2000):

| variant | dos_w1 | sv_w1 | hhi_dst | hhi_src |
|---|---|---|---|---|
| none | 0/20 | 0/20 | 3/20 | 2/20 |
| β[hub] | 3/20 | 12/20 | 15/20 | 10/20 |
| β[merge] | 1/20 | 1/20 | 1/20 | 0/20 |
| β[densify] | 4/20 | 2/20 | 0/20 | 0/20 |
| α | 3/20 | 3/20 | 0/20 | 1/20 |
| γ[src] | 1/20 | 1/20 | 2/20 | 1/20 |
| δ | 1/20 | 1/20 | 0/20 | 0/20 |

`wiki_like` (w = 5000; mean cost per window pair: DoS 1.82 s, SV 0.26 s):

| variant | dos_w1 | sv_w1 | hhi_dst | hhi_src |
|---|---|---|---|---|
| none | 0/20 | 1/20 | 0/20 | 0/20 |
| β[hub] | **1/20** | 10/20 | 8/20 | 2/20 |
| β[merge] | 1/20 | 2/20 | 1/20 | 0/20 |
| β[densify] | 1/20 | 1/20 | 1/20 | 0/20 |
| α | 1/20 | 1/20 | 2/20 | 0/20 |
| γ[src] | 0/20 | 0/20 | 0/20 | 0/20 |
| δ | 2/20 | 2/20 | 2/20 | 0/20 |

On `wiki_like` the locked DoS statistic detects β[hub] in 1/20 seeds (the same as its false-alarm level), consistent with the Step 1 result (2/20); SV reaches 10/20 and the destination concentration 8/20. On the SV statistic δ does **not** leak here (2/20) although the block-law change of §3.3 exists, so the δ leak into SV is preset-dependent (15/20 on `small_bipartite`, 2/20 on `unipartite_small` and `wiki_like`); I did not investigate why.

Reading (limited to what the data support):
1. The generator *does* change β: β[hub] is detected by all four statistics on `small_bipartite` (20/20), so the injection is not the problem there. The earlier 17/20 purity result for DoS (88% true power on 100 other seeds) is sampling variation around a real but sub-100% power.
2. β[merge] (TV 0.03) and β[densify] (TV 0.06) are small perturbations of the block law at ε = 0.2; no label-free statistic here detects them reliably. This is a **power limit at the specified ε**, not evidence of leakage or of an implementation fault.
3. DoS has near-zero power on `unipartite_small` β[hub] (3/20) while the concentration index has 15/20 and SV 12/20. Whether this is a property of the statistic or of that preset's weak hub (the hub boost acts on one block of 4 in a symmetric B) is **not established**; no variant of the experiment separating the two explanations was run.
4. The Herfindahl index is O(E), 15× cheaper than DoS here, but it is a degree-concentration statistic: by construction it cannot see community-structure β, and its β[hub] power says nothing about β in general.
5. No candidate was tuned on these seeds; the result is exploratory and may not be used as confirmation. If any alternative replaces DoS for β, that is a method change needing your approval and a fresh confirmatory seed set.

### 3.5 Fresh-node ids and node-table leakage (audit of `dsbm.py`)
- Legacy layout: fresh ids are appended after all initial ids, so (i) a fresh *source* id exceeds every destination id, breaking the "sources 0..N_src−1, destinations after" convention for churned streams; (ii) `node_feat`/`node_ids`/`node_role` have `n_initial + (churned ids)` rows, so the **table length reveals ε and the sides churned**; (iii) feature rows of future entities are readable before they appear.
- `SynthResult.save` writes `node_block.npy` and `drift_truth.csv` next to `stream.pt`: acceptable because the learner API takes only `EventStream`, but the folder is not protected. The leakage test covers `src/gdb/cl` and `src/gdb/detect` by text search only, which is a weak check; a runtime check on the learner interface would be stronger (Step 4).

## 4. Recommended decisions for the five open questions of `step-1.md`

| # | Question | Recommendation | Basis |
|---|---|---|---|
| 1 | β / DoS | Run G1 with the **locked DoS**; report its failure modes. Pre-register sv_w1 and a concentration statistic as the *A5-style alternatives*, evaluated on a fresh confirmatory seed set, and decide on replacement only after seeing both. Do not enlarge ε or w. | §3.4 |
| 2 | ε conventions | **α:** keep the relative definition (shift = ε·σ·√d) but also report the unit-vector norm in every table (at d = 172, ε = 0.2 → 2.62 vs 0.2). **δ:** keep ε as a mixing weight for dose-response but treat **ε = 1 as the only isomorphism-preserving δ** and report the leakage of fractional δ into spectral statistics (§3.3). Needs your decision: for G1 "at ε = 0.2", δ is a mixing weight 0.2 (my current default, leaks into spectral statistics) or δ is evaluated at the full permutation (pure, but not "ε = 0.2"). | §3.3, `eps_effective.csv` |
| 3 | Learnability | Give the learner **informative static node features** (block-mean + noise), which the generator already produces, with a sharper block law for G2 streams. Measured (one seed, 20 epochs, exploratory): featureless TGN AP 0.50 on the uniform block SBM vs 0.70 with node features; sharp block (p_out 0.02) 0.50 → 0.77, with β_h = 2 0.80; heavy-tailed popularity is learnable without features (0.70) but popularity is not changed by δ, so it cannot show δ-forgetting. Block-oracle ceilings: 0.66–0.78 | `learnability.csv`; table below |
| 4 | Fresh ids | Adopt the **reserved layout** (implemented, opt-in, §5): fixed per-side id ranges chosen from the config only, fixed-size node table, `observed_node_table()` for the learner. Default stays `legacy` until you approve, so existing hashes and files do not change. | §5 |
| 5 | G1 reference preset | **`small_bipartite` (d_edge = 32) as the pre-registered primary** (power, speed), with `wiki_like` as a pre-registered secondary run. Expect the secondary to fail β via DoS; report that rather than drop it. | §3.4 |
| 6 | Lint in auditor's test | one E731 left in `tests/test_metrics_audit_step1.py`; harmless, left as is | — |

Learnability, TGN 20 epochs, lr 3e-3, memory dim 32, seed 7000, ~15k events (`results/exploratory/learnability.csv`):

| stationary link rule | block-oracle AP | TGN no node features | TGN + node features |
|---|---|---|---|
| uniform block, β_h 0.5 | 0.659 | 0.501 | 0.697 |
| uniform block, β_h 2 | 0.669 | 0.504 | 0.703 |
| sharp block (p_out 0.02) | 0.781 | 0.500 | 0.774 |
| sharp block + β_h 2 | 0.772 | 0.499 | 0.801 |
| Pareto popularity | 0.670 | 0.698 | 0.816 |

The "block oracle" scores pairs by the true $B_{ab}$ only, ignoring the homophily term and popularity, so it is a reference, not a Bayes bound (TGN with features exceeds it where the homophily term matters). One seed and one model size: this shows learnability exists, not how large the later forgetting effect will be.

## 5. Safe corrections made (all opt-in or additive; legacy output is byte-identical)
Verified: the legacy stream hash for `small_bipartite`, γ 0.2 + α 0.2 at t = 25, seed 0 is `1e04b330…` before and after the change, equal to the value recorded in `step-1.md`.

- `DSBMConfig.id_layout ∈ {legacy, reserved}` (default `legacy`) and `id_headroom` (default 1.0). Reserved layout: sources own `[0, C_s)`, destinations `[C_s, C_s + C_d)` with $C = \lceil(1+h)n\rceil$ from the config only; fresh ids are allocated inside the owner's range; the node table has a fixed size independent of ε and of the schedule; overflow raises. Unipartite uses one range.
- `EventStream.node_initial` (bool, `None` for legacy) and `EventStream.observed_node_table(upto)` (zeroes features of entities that are not initial and have not appeared in `events[:upto]`).
- **Migration/compatibility:** with `reserved`, every destination id shifts by $C_s - n_{src}$, so streams, hashes and `stream.pt` files are not interchangeable with legacy ones; `tgb_loader` output (dense TGB ids) is unchanged. Existing tests, presets and `data/synth` outputs keep the legacy layout. Switching the default is a one-line change that I have **not** made.
- `bipartite_sv_spectrum`, `sv_w1`, `concentration_index` in `graph_stats.py`: exploratory candidates, outside the locked detector.
- `run_tgn_smoke(use_node_feat=...)`: static node features added to the memory state.
- Scripts: `beta_diagnostic.py`, `learnability_study.py`, `eps_effective.py` (all write only under `results/exploratory/`).

Tests added: `tests/test_id_layout.py` (6: disjoint namespaces under two-sided churn, ε-independent table size/offset, no renumbering of initial ids, future-entity masking, headroom overflow, determinism + legacy unchanged), `tests/test_candidate_stats_and_learnability.py` (SV spectrum vs `numpy.linalg.svd`, Herfindahl extremes, slow learnability test).

## 6. Test and repository status
- Full suite (`python -m pytest`): **90 passed** in 86 s (77 fast + 13 slow). Ruff: clean on `src`, `scripts`, new tests; the one remaining finding is the E731 in the metrics-auditor's file (left as written).
- `git status` (nothing committed): modified `.gitignore`, `src/gdb/models/tgn_smoke.py`, `src/gdb/synth/dsbm.py`, `src/gdb/utils/graph_stats.py`; untracked: `src/gdb/data/` (**was missing from `181a5e2`**), `configs/data/` (**also missing**, hidden by the same ignore rule), `docs/progress/paper1-baseline-and-g1-readiness.md`, `results/exploratory/` (+ four `results/exploratory_*.log` files), `scripts/{beta_diagnostic,eps_effective,learnability_study}.py`, `tests/test_id_layout.py`, `tests/test_candidate_stats_and_learnability.py`.
- I did not run the code-reviewer or metrics-auditor agents on this diff (not requested); the SV/Herfindahl tests are mine and have not been independently audited.

## 7. Gate G1: protocol and why it was not executed

**Blueprint criterion (§13):** four raw statistics separate the four pure drift types on synthetic streams at ε = 0.2: (a) the 4×4 matrix $\mathbb E[z_{c'}\mid\text{inject }c]$ is diagonal-dominant; (b) the argmax-of-standardised-z classifier has macro-F1 ≥ 0.80; (c) the δ residual fires on pure α/β/γ in ≤ 15% of seeds. On failure: redesign the weak statistic, narrow the taxonomy claim, do not build replay.

**Unspecified in the Blueprint (so G1 cannot be run without choices):**
1. **The δ statistic.** δ is a prequential-loss residual of a learner. Step 1 has no detector and no learner that learns the block rule; the Step 1 statistic `blockpair_chi2` uses ground-truth block labels and is not admissible. A learner with node features (§4, Q3) is needed.
2. Window size w, reference length r, and the standardisation source ("moving-block bootstrap of the reference windows" is Step 3 machinery that does not exist).
3. Whether "pure drift" for δ means ε = 0.2 mixing or the full permutation (§3.3).
4. The β target (hub / merge / densify) and the preset (§4, Q5).
5. Tie handling and the macro-F1 class set (4 classes + "none"?).

**Proposed pre-registered protocol (needs your approval of the choices in §9):**
- Preset `small_bipartite`, reserved id layout, informative node features; w = 2000 (preset value), ε = 0.2, abrupt onset; β target = `hub` (boost 10) as the primary, `merge` and `densify` reported as additional rows that do **not** enter the verdict.
- Seeds: stationary calibration streams 3000–3199 (for the z-standardisation mean/sd per statistic); evaluation seeds 2000–2019 per type (20 seeds × {α, β, γ, δ} + 20 stationary streams). None of these has been used so far. They must not be reused for any tuning.
- z-score: $z_c=(s_c-\hat\mu_c)/\hat\sigma_c$ with $\hat\mu,\hat\sigma$ from the stationary calibration streams (a simplification of the Blueprint's block bootstrap; stated as a deviation).
- Statistics: γ = max of standardised (1 − Jaccard, new-node rate); β = DoS W1 (locked, 100 moments, 200 probes); α = degree-stratified linear MMD; δ = loss residual per §5.1 with $\mathbf w$ fitted by Huber regression on the calibration streams, loss from a TGN with node features trained on the reference windows only.
- Scoring: (a) the 4×4 mean-z matrix with 95% bootstrap CIs over seeds, diagonal-dominance = the diagonal entry is the row maximum in every row; (b) macro-F1 of argmax-z over the 80 labelled windows (+ a "none" class from the 20 stationary streams, reported separately with and without it); (c) fraction of pure α/β/γ seeds with δ-residual z above its q95 calibration threshold. Also reported but not gating: confusion matrix, detection delay in windows, false alarms per 10^5 events, AUROC per statistic.
- Missing/failed values: a non-finite statistic raises (existing behaviour) and the run is reported as failed, not dropped; tied argmax is counted as a misclassification.
- Cost estimate (extrapolated from the smoke test and the timings above, **not measured for this protocol**): window statistics ≈ 0.2 s per stream; the TGN pass dominates at roughly 15–30 s per stream on the RTX 3070 (peak ≈ 0.03 GB measured in the smoke test), ≈ 300 streams → ≈ 1.5–3 h; CPU fallback if free VRAM < 2.5 GB is feasible for small_bipartite but slower. `wiki_like`: DoS dominates; ≈ 1–2 s per window pair.
- Interpretation rule fixed in advance: a failure of (a), (b) or (c) is reported as a failure and triggers the Blueprint's "if it fails" branch; ε, w and thresholds are not changed afterwards.

**Not executed.** Reasons: items 1–4 above require your decisions; the δ learner depends on the unresolved learnability and id-layout choices; and the existing evidence already shows that the *locked* β statistic is expected to have limited power on `wiki_like` and `unipartite_small`. Running a "G1" now would require me to make method choices the brief tells me to bring to you.

## 8. Failed criteria and unresolved risks
- **Step 1 purity:** β via DoS fails in all three presets (17/20, 2/20, 2/20 vs the ≥ 18/20 rule); unipartite δ via DoS 16/20 (needs 17). Unchanged.
- **Expected G1 risks:** β separability on `wiki_like`/unipartite via DoS; δ-vs-β leakage in spectral statistics at ε < 1; δ residual not yet implementable.
- **Taxonomy claim risk:** Blueprint Proposition 1 assumes isomorphism-preserving δ; fractional δ violates it spectrally.
- **Pass-rule weakness:** a null cell passes with probability ≈ 0.4 even if 20% of seeds leave the band (noted in Step 1); G1 as I propose it adds CIs, which is sharper than the 17/20 rule.
- **Exploratory only:** every number in §3.4, §3.3 and §4 (learnability) is exploratory; none is a confirmatory result, none belongs in the paper's tables, and the learnability numbers are single-seed.
- **Citations:** verified by existence only; full-text claims unverified; DyGFormer, TGAT, JODIE, DyRep, GraphMixer, TGB, TGN, Vitter, Gretton, OTGNet, PromptGNN not re-verified (OpenAlex search was rate-limited, HTTP 429). The Batagelj–Zaveršnik DOI in the Blueprint should be the ADAC paper 10.1007/s11634-010-0079-y (*Fast algorithms for determining (generalized) core groups in social networks*), which differs from the "An O(m) algorithm for cores decomposition" title the Blueprint quotes (that title I believe exists only as a preprint). LwF year should be checked (2017 online).

## 9. Hypotheses H1–H5: can the planned experiments support them?
- **H1** (type macro-F1 > single-statistic detectors): supportable only if the comparison detectors are given a type rule; otherwise restrict to detection F1 for them. Depends on G1.
- **H2** (lower AF at equal budget, non-inferior AA): supportable on synthetic **only if** the stream has a learnable rule (§4 Q3); on TGB-wiki the effect may be small, report honestly.
- **H3** (≥ 50% fewer replay steps): measurable once alarms exist; compare against #8a/#8b with step counts reported.
- **H4** (k-core beats random events, mostly under β/γ): depends on the β/γ injections being detectable and on A3; a DoS-blind β would undermine the "mostly under β" half.
- **H5** (down-weighting after δ): needs a learnable δ and G1(c); not testable before then.
- Orthogonality claim: do not state "orthogonal" without the audit; §3.3 shows one concrete leak.

## 10. Recommended next step
1. You decide §11; I commit the `.gitignore` fix and this audit's code as a separate commit.
2. Switch the default id layout to `reserved` and give the learner node features (small change, tested).
3. Implement the minimal δ learner path (feature-aware TGN prequential loss) and a throwaway G1 harness exactly per §7; run it once on the pre-registered seeds; report whatever happens.
4. Only then Step 2 (metrics, pure and fast) and, after G1 passes or the taxonomy is narrowed, Step 3.

## 11. Decisions I need from you
1. Commit the `.gitignore` fix (and the audit changes) now? (Recommended: yes, as a new commit.)
2. Adopt `id_layout = "reserved"` as the default for new streams? (Recommended: yes; legacy stays available.)
3. δ at G1: fractional mixing ε = 0.2 (leaks spectrally), or full permutation, or both reported with the mixing row as the gating one? (My default: gating = mixing 0.2 as the Blueprint states ε = 0.2, plus the full-permutation row.)
4. β at G1: confirm `hub` as the gating target, `merge`/`densify` as non-gating, DoS as the locked statistic, sv_w1/HHI as pre-registered A5 alternatives only.
5. Learnability: confirm informative node features + a sharper block law (p_out ≈ 0.02) for G2/Step 4 streams.
6. G1 preset: `small_bipartite` primary, `wiki_like` secondary, as proposed.
7. Standardisation: stationary-calibration z-scores instead of the Blueprint's block bootstrap for G1 only (deviation), or build the bootstrap first.
8. Baselines: TWP to the appendix; CaT/PromptGNN/OTGNet discussed only; TEM labelled "TEM-style"; oracle-trigger diagnostic allowed on synthetic only.
9. Authorise reading the full texts of ER-GNN, SSM, TEM and ContinualGNN (via the literature skills) before implementing any of them.
