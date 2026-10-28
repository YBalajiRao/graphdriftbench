# Step 1 — synthetic dynamic-SBM generator, pure drift injectors, graph statistics, TGB loader

Status: **built and tested; 69 fast + 12 slow tests pass. One purity check fails (β detected through DoS) and is reported, not tuned away. Waiting for your decision before Gate G1 or Step 2.**

## 1. What was built

| Path | Content |
|---|---|
| `src/gdb/synth/dsbm.py` | `DSBMConfig`, `generate_stream()`, `SynthResult`; bipartite (default) and unipartite; ground-truth block labels and diagnostics |
| `src/gdb/synth/injectors.py` | `DriftEvent`, `DriftSchedule` (+ CSV round-trip), `inject_alpha/beta/gamma/delta`, `beta_target`, `delta_permutation` |
| `src/gdb/synth/schedules.py` | `abrupt`, `gradual` (staircase ramp), `mixed`, `recurring` (A→B→A) |
| `src/gdb/synth/purity.py` | window-pair statistics, stationary null band, pass-rate rules (17/20, 18/20), variant list |
| `src/gdb/utils/graph_stats.py` | degree distributions + KS, KPM density of states (Chebyshev, Rademacher, Jackson), W1 between spectra, `networkx` core numbers, linear-time RBF MMD (median heuristic), degree-stratified MMD, Jaccard / new-node rate. Pure, typed, LaTeX docstrings, inputs = arrays of one window |
| `src/gdb/data/stream.py` | `EventStream` (+ hash, save/load, PyG `TemporalData` export, optional integer time) |
| `src/gdb/data/tgb_loader.py` | `tgb_root()` context manager (patches and restores `PROJ_DIR`), `load_tgb()` → `EventStream` + official masks; raises (instead of prompting for a download) if the folder is missing |
| `src/gdb/models/tgn_smoke.py` | minimal PyG TGN loop with the free-VRAM check (`torch.cuda.mem_get_info()`, CPU below 2.5 GB) |
| `scripts/` | `gen_synth.py`, `purity_report.py`, `check_tgb_review.py` (manual, not in tests) |
| `configs/synth/` | `small_bipartite`, `tiny_fast`, `wiki_like`, `unipartite_small`; `configs/data/tgb.yaml` (data root) |
| `tests/` | 8 files; `tests/test_metrics_audit_step1.py` was written by the metrics-auditor agent |

Model (documented in `dsbm.py`): degree-corrected block model. Pair intensity ∝ `B[a,b] θ_u θ_v exp(β_h cos(x_u,x_v))`, with each block's activities rescaled so Θ_a = n_a, and the proposal rate divided by the **exact** mean acceptance `c_ab` of the homophily factor. Result: the expected events per block pair are exactly `λ0 B_ab n_a n_b / Z` for any β_h, so δ (a block permutation) leaves all margins and degree laws unchanged. Simulation is block-pair Poisson thinning, O(#events).

## 2. Presets

| Preset | Shape | Use |
|---|---|---|
| `small_bipartite` | 300 src / 150 dst, 4+4 blocks, 32-d marks, ~10K events, w = 2000 | purity tests (32-d because the linear-time MMD has little power at d ≤ 16, see §6) |
| `tiny_fast` | 200 / 100, 4-d marks, w = 1000 | the "small-d option" for fast tests |
| `wiki_like` | 8,227 src / 1,000 dst, 8+8 blocks, Pareto popularity (dst α = 1.2, src α = 2.5), 172-d marks, ~157K events, w = 5000 | shape of `tgbl-wiki`; block structure is **not** fitted to the data |
| `unipartite_small` | 300 nodes, 4 blocks with paired-block `B`, 32-d marks | unipartite mode |

## 3. Purity pass-rate tables (ε = 0.2, 20 seeds, abrupt onset)

Test seeds 0–19; null band = 95th percentile of 200 stationary streams (seeds 1000+), same preset and window size, saved in `results/null_band_<preset>.json`. Cells show seeds counted / 20: **null cells count seeds inside the band (pass ≥ 17)**, **shift cells count seeds outside it (pass ≥ 18)**, cells without ✓/✗ have no requirement. Expectations: α → only `mark_mmd` shifts; β → only `dos_w1` shifts (degree and block-pair statistics are informational); γ → only `new_node_rate_src` shifts; δ → only `blockpair_chi2` shifts. Rows `beta[merge]`, `beta[densify]`, `gamma[both]`, `delta[+flip]` are extra variants and not part of the verdict. All raw per-seed values are in `results/purity_raw_<preset>.csv`.

Primary checks failed: `small_bipartite` 1 of 32, `wiki_like` 1 of 32, `unipartite_small` 2 of 32.

### `small_bipartite`

| type | deg_ks_src | deg_ks_dst | dos_w1 | mark_mmd | blockpair_chi2 | new_node_rate_src | new_node_rate_dst |
|---|---|---|---|---|---|---|---|
| none | 20/20✓ | 18/20✓ | 20/20✓ | 18/20✓ | 19/20✓ | 19/20✓ | 20/20✓ |
| alpha | 19/20✓ | 18/20✓ | 19/20✓ | 20/20✓ | 20/20✓ | 20/20✓ | 20/20✓ |
| beta[hub] | 18/20 | 20/20 | 17/20✗ | 19/20✓ | 20/20 | 18/20✓ | 20/20✓ |
| beta[merge] | 2/20 | 2/20 | 1/20✗ | 19/20✓ | 14/20 | 20/20✓ | 20/20✓ |
| beta[densify] | 0/20 | 0/20 | 2/20✗ | 18/20✓ | 13/20 | 20/20✓ | 20/20✓ |
| gamma[src] | 20/20✓ | 20/20✓ | 19/20✓ | 17/20✓ | 20/20✓ | 20/20✓ | 20/20✓ |
| gamma[both] | 20/20✓ | 19/20✓ | 18/20✓ | 17/20✓ | 20/20✓ | 20/20✓ | 20/20✓ |
| delta | 20/20✓ | 19/20✓ | 19/20✓ | 18/20✓ | 20/20✓ | 20/20✓ | 20/20✓ |
| delta[+flip] | 0/20 | 0/20 | 2/20 | 18/20✓ | 20/20✓ | 19/20✓ | 20/20✓ |

### `wiki_like`

| type | deg_ks_src | deg_ks_dst | dos_w1 | mark_mmd | blockpair_chi2 | new_node_rate_src | new_node_rate_dst |
|---|---|---|---|---|---|---|---|
| none | 20/20✓ | 19/20✓ | 19/20✓ | 20/20✓ | 18/20✓ | 19/20✓ | 18/20✓ |
| alpha | 19/20✓ | 20/20✓ | 20/20✓ | 20/20✓ | 20/20✓ | 19/20✓ | 19/20✓ |
| beta[hub] | 5/20 | 4/20 | 2/20✗ | 18/20✓ | 20/20 | 19/20✓ | 19/20✓ |
| beta[merge] | 1/20 | 0/20 | 0/20✗ | 18/20✓ | 5/20 | 19/20✓ | 19/20✓ |
| beta[densify] | 1/20 | 0/20 | 0/20✗ | 17/20✓ | 20/20 | 18/20✓ | 19/20✓ |
| gamma[src] | 18/20✓ | 19/20✓ | 19/20✓ | 19/20✓ | 20/20✓ | 20/20✓ | 19/20✓ |
| gamma[both] | 19/20✓ | 20/20✓ | 18/20✓ | 18/20✓ | 20/20✓ | 20/20✓ | 20/20✓ |
| delta | 18/20✓ | 20/20✓ | 20/20✓ | 18/20✓ | 20/20✓ | 18/20✓ | 19/20✓ |
| delta[+flip] | 1/20 | 0/20 | 0/20 | 20/20✓ | 20/20✓ | 20/20✓ | 19/20✓ |

### `unipartite_small`

| type | deg_ks_src | deg_ks_dst | dos_w1 | mark_mmd | blockpair_chi2 | new_node_rate_src | new_node_rate_dst |
|---|---|---|---|---|---|---|---|
| none | 18/20✓ | 20/20✓ | 19/20✓ | 20/20✓ | 19/20✓ | 20/20✓ | 19/20✓ |
| alpha | 20/20✓ | 20/20✓ | 19/20✓ | 20/20✓ | 19/20✓ | 20/20✓ | 20/20✓ |
| beta[hub] | 2/20 | 5/20 | 2/20✗ | 19/20✓ | 20/20 | 19/20✓ | 19/20✓ |
| beta[merge] | 2/20 | 0/20 | 2/20✗ | 19/20✓ | 17/20 | 20/20✓ | 19/20✓ |
| beta[densify] | 0/20 | 0/20 | 2/20✗ | 20/20✓ | 19/20 | 20/20✓ | 20/20✓ |
| gamma[src] | 20/20✓ | 20/20✓ | 17/20✓ | 20/20✓ | 19/20✓ | 20/20✓ | 20/20✓ |
| delta | 20/20✓ | 19/20✓ | 16/20✗ | 19/20✓ | 20/20✓ | 19/20✓ | 19/20✓ |
| delta[+flip] | 0/20 | 0/20 | 1/20 | 20/20✓ | 20/20✓ | 20/20✓ | 20/20✓ |

### The failures, investigated (nothing was tuned)

1. **β[hub] through DoS fails in all three presets** (17/20, 2/20, 2/20). On 100 extra seeds (100–199, disjoint from null and test seeds) the true detection rate in `small_bipartite` is **88/100** (stationary control 3/100), so P(≥ 18/20) is only about 0.56 even for a correct implementation: that check is at the edge of the specified rule, not broken. In `unipartite_small` it is 12/100 and in `wiki_like` 2/20: **DoS W1 of the window graph is essentially blind to β there.** `merge` and `densify` (community-structure changes, mass-preserving) are invisible to it in every preset. The normalised Laplacian removes degree scale, and a K-block structure only moves K of n eigenvalues, so community changes barely move the W1. This concerns the statistic, not the generator: block-pair counts do shift under β.
2. **δ through DoS in `unipartite_small`: 16/20 inside** (needs 17). On 100 extra seeds δ exceeds the band in 8/100 vs 6/100 for stationary streams, which is the nominal 5%: no leak, a chance excursion.
3. **Earlier unipartite run (superseded).** A first unipartite run showed `deg_ks_src` at 16/20 for α, γ and δ, and δ's block-pair counts at 0/20. The block-pair 0/20 was a real preset defect: an assortative symmetric `B` is invariant under `P B Pᵀ`, so δ was a silent no-op (the α and δ rows had identical statistics). The generator now refuses a no-op δ (`ValueError`, with a test) and redraws the permutation until `B` changes; the unipartite preset uses a paired-block `B`. The `deg_ks_src` excursion disappeared with the new realisation, consistent with chance. Seeds, ε and w were not changed.
4. **Weakness of the rule itself** (code-reviewer): a null cell passes with probability ≈ 0.4 even if an impure injection pushes 20% of seeds out of the band. Observed counts are reported in every cell; a binomial test against the 5% expected exceedance would be sharper.

## 4. Other acceptance results

| Item | Result |
|---|---|
| Stationary block-pair counts vs Poisson expectation | ≥ 95% of cells within 3σ and χ² p > 1e-3, seeds 0–4 (`test_dsbm.py`) |
| Reproducibility | same seed → identical SHA-256 stream hash. Example outputs in `data/synth/` (git-ignored), `small_bipartite`, γ 0.2 + α 0.2 at t = 25, seeds 0–4: `1e04b330…`, `bd7c5d99…`, `136cd23b…`, `7fe2060a…`, `20158169…`; plus `wiki_like` δ 0.4 ramp (seed 0): 157,283 events |
| `drift_truth.csv` vs configured schedule | exact round trip for abrupt, gradual, mixed and recurring events (`test_injectors.py`); an extra `resolved` column records the seeded details (permutation, id ranges, shift norm) |
| KPM vs `numpy.linalg.eigvalsh` (n = 200–500, 100 moments, 30 probes) | required W1 error < 0.02; worst of 12 exploratory cases 0.0154; the test uses 3 graphs × 5 probe seeds. At 150–200 moments and 60–100 probes the errors were 0.003–0.009 |
| k-core | `networkx.core_number`, tested on a K4 + triangle + pendant-path graph |
| 1M events, CPU | **998,013 events in 2.8 s** (`wiki_like` shape, 172-d marks, mixed γ+α) |
| TGB loader | `tgbl-wiki`: 157,474 events, 8,227 sources, 1,000 destinations, 172-d float32 features, masks 110,232 / 23,621 / 23,621; `PROJ_DIR` restored afterwards (also tested under an exception). `tgbl-review` is not loaded in tests (`scripts/check_tgb_review.py` is the manual check; I did not run it) |
| PyG export + TGN smoke | `to_temporal_data()` works. 10 epochs, 10,146 events, memory dim 32: free VRAM before = 6.95 GB → ran on CUDA; **peak allocated 0.021 GB** (reserved 0.027 GB), 11 s, loss 1.347 → 1.305, held-out AP 0.655 |
| Leakage | `tests/test_no_leakage.py` greps `src/gdb/cl` and `src/gdb/detect` for `drift_truth`, schedule classes, `node_block`, `synth`, `resolved`, `config.yaml`. Both packages are still empty stubs, so it passes vacuously today and becomes real at Steps 3–4 |
| Suites | fast (`-m "not slow"`): 69 tests, ≈ 20 s; slow: 12 tests (loader, 9 purity variants, 1M runtime, TGN smoke), ≈ 45 s |

## 5. Agent reviews

- **metrics-auditor**: 17 tests; every formula re-derived independently and matching (Jackson, KPM moments, W1, linear MMD, stratified MMD, Jaccard/new-node rate, KS, k-core, block-pair χ²). One defect, **fixed**: pass thresholds were floored (`17n//20`), so runs with n < 20 seeds passed at 80%; now a ceiling. The official 20-seed numbers were unaffected.
- **code-reviewer**: generator math, gamma bookkeeping, ramp/recurring logic, injector purity, determinism and Windows constraints: clean. **Fixed:** NaN statistics now raise (a NaN counted as "outside the band" and made shift cells easier to pass); unseeded default RNG in `kpm_dos` replaced by a fixed seed; TGN negatives now drawn from destinations seen in training (not a min/max range that could include fresh source ids or test-period nodes); loader raises instead of prompting for a download; leakage patterns widened; non-primary failures are printed; unipartite γ expectation corrected; docstring corrections; peak reserved memory logged. **Not fixed, recorded as open questions:** §7 items 3.

## 6. Deviations from the Blueprint and decisions you should confirm

1. **Meaning of ε.** β: mixing weight (as specified). γ: fraction of nodes replaced (as specified). **δ: mixing weight** `B ← (1−ε)B + ε·BPᵀ` (a permutation has no natural magnitude; ε = 1 is the full permutation, where the multiset of entries is unchanged, tested). With `flip_homophily`, β_h ← (1−2ε)β_h. **α: shift = ε · σ_e · √d_e** (relative to the mark norm) instead of ε · (unit vector); a literal unit-norm shift at ε = 0.2 in 172-d would be 1.5% of the mark norm.
2. **Degree-corrected SBM.** Blueprint §6.1 is a plain SBM; popularity θ is optional (`pop_*: none` recovers it) and needed for the `wiki_like` heavy tails.
3. **γ keeps the replaced node's block and activity θ** and draws fresh features from the block law (Blueprint: "same block/feature law"). Re-drawing θ would leak γ into the degree distribution.
4. **Gradual onset is a staircase** (10 steps) of the piecewise-constant law; recurring γ means the departed ids return.
5. **Primary β target is `hub`** (degree concentration, boost 10), chosen after exploratory runs on seeds 5000+/6000+ (disjoint from test and null seeds) showed that `merge`/`densify` are invisible to DoS; all three targets are reported above. Exploration also set `d_edge = 32` for the purity preset (exploratory α detection: 7/12 at d = 4, 8/12 at d = 16, 12/12 at d ≥ 32) and 200 KPM probes for the purity statistics (30 probes leave a W1 noise floor of ≈ 0.007 that swamps β; library default stays 100 moments / 30 probes). ε = 0.2, the window sizes and the thresholds were never changed.
6. **Block-pair χ² uses ground-truth block labels.** It is a generator diagnostic only; the detector statistics in `graph_stats.py` never use labels.
7. Unipartite δ requires a block matrix that is not invariant under block relabelling (guard added).
8. `pyarrow` was installed into `gdb` (needed to unpickle TGB's cached files; you approved it). numpy is 2.4.6 and pandas 3.1.0; `py-tgb`'s `clint` dependency prints a harmless traceback at interpreter exit.

## 7. Open questions for you

1. **β and the DoS statistic (the Gate G1 risk).** DoS W1 detects β[hub] in 88% of seeds on the small preset, barely on `unipartite_small` and not on `wiki_like`; community-structure β is invisible. Options: (a) run G1 as specified, since it measures exactly this (my recommendation); (b) larger β magnitude (ε = 0.4) or larger w for β only; (c) a different β statistic (Fréchet, LAD, or a block-pair/degree statistic); (d) narrow the taxonomy claim. I changed nothing.
2. Are the ε conventions for α and δ (§6.1) acceptable?
3. **Fresh ids from γ** are appended after all initial ids (sources and destinations interleaved), so a fresh source id exceeds every destination id and "destinations after sources" no longer holds for churned streams (and in unipartite TGB data a node that is both source and destination is labelled DST). `node_feat`/`node_ids` also contain rows for ids that appear only later, so their row count reveals ε. Learners must use `node_role` and first-seen masking, never `len(node_feat)` or id magnitude. Do you want separate id ranges per side and lazily revealed node tables?
4. **Learnability for Step 4 / Gate G2.** A featureless TGN does not learn the uniform block SBM (loss stays at 2 ln 2 = 1.386, AP 0.50 over 10–40 epochs, lr 1e-3 to 1e-2); it does learn heavy-tailed popularity (AP 0.66–0.67). The smoke test therefore uses heavy tails. A model that never learned the block structure cannot forget it or suffer from δ, so G2 streams need a learnable link rule (informative node features through the homophily term, informative edge marks, or popularity). Which should the benchmark use?
5. Which preset is the official reference for G1: `small_bipartite` (d = 32) or `wiki_like`?
6. One lint finding remains (`E731` lambda in `tests/test_metrics_audit_step1.py`, the auditor's file); I left that file as written.
