# Paper 1 — Final Blueprint (v1.0, locked 2026-10-28)

**Working title:** *Structural Evolution Learning with a Taxonomy of Relational Drift: Type-Aware Drift Detection and Structure-Preserving Replay for Continual Learning on Continuous-Time Dynamic Graphs*

**Method name:** CSEE — Continual Structural Evolution Engine
**Benchmark name:** GraphDriftBench (v1)
**Target venue:** IEEE TKDE (primary) · IEEE TNNLS (alternative) · Information Sciences / Neural Networks (backup)
**Primary task:** Continuous-time dynamic link prediction (CTDG) on TGB `tgbl-wiki` and `tgbl-review`, with semi-synthetic drift injection.

> **Scope lock.** Paper 1 contains ONLY: (i) the 4-factor drift taxonomy, (ii) the 4-test online detector, (iii) drift-triggered, type-conditioned, k-core structure-aware replay, (iv) GraphDriftBench (synthetic dynamic SBM + semi-synthetic TGB). The structural world model $W(t)$, forecasting $W(t+\Delta t)$, predictive/CVaR memory allocation → **Paper 2**. Healthcare (MIMIC-IV / PhysioNet) → **Paper 3**. Do not let these creep back into Paper 1.

> **Citation rule (supervisor):** no preprints (arXiv etc.) in the reference list. Every reference must have a DOI or a peer-reviewed proceedings record. Items marked **[verify]** below were not in the two LeapSpace surveys and must be confirmed in Scopus/DOI before citing.

---

## 0. One-paragraph abstract (draft — numbers to be filled ONLY from results)

Real-world interaction graphs drift continuously: their features change, their topology rewires, their entities churn, and the rules governing who interacts with whom change. Continual graph learning (CGL) methods mitigate catastrophic forgetting with replay or regularisation, but they assume artificial task boundaries and treat every change identically. Graph change-point detectors, conversely, raise alarms but do not adapt a model. We propose CSEE, which closes this gap for continuous-time dynamic graphs. First, we derive a four-type taxonomy of relational drift from the factorisation $P(V)\,P(A\mid V)\,P(X\mid A,V)\,P(Y\mid A,X,V)$, so that entity turnover ($\gamma$), topological drift ($\beta$), feature drift ($\alpha$) and relational-semantic drift ($\delta$) each correspond to a shift in exactly one factor. Second, we design a linear-time online detector that runs one calibrated test per factor inside a CUSUM scheme with family-wise error control, returning both *when* and *which type* of drift occurred. Third, the detected type selects a replay policy over a budgeted memory of sparsified k-core ego-networks, so the learner refreshes memory only when, and only how, the drift requires. We release GraphDriftBench, which pairs a controllable dynamic-SBM generator with type-pure drift injection into TGB streams and a unified protocol joining forgetting metrics with detection metrics. On `tgbl-wiki`, `tgbl-review` and synthetic streams, CSEE [RESULTS — fill from `results/` only].

---

## 1. Problem statement

A CTDG is an event stream $\mathcal{E}=\{(u_i,v_i,t_i,\mathbf{x}_i)\}_{i\ge1}$ with $t_1\le t_2\le\dots$. A model $f_\theta$ predicts future events (link prediction) from history $\mathcal{H}_t=\{e_i: t_i<t\}$. The stream is **non-stationary**: the data-generating law changes at unknown times $\tau_1<\tau_2<\dots$, with unknown type. Goals:

1. **Plasticity:** high link-prediction quality on the current regime.
2. **Stability:** retain quality on past regimes (low forgetting) — past regimes may recur.
3. **Awareness:** detect each change online, with low delay and few false alarms, and identify its type.
4. **Efficiency:** bounded memory $M$ and bounded adaptation compute; no task boundaries given to the learner.

---

## 2. What already exists (positioning)

### 2.1 Four literatures that do not talk to each other

| Line of work | Representative papers (from your surveys) | What they do | What they lack for our problem |
|---|---|---|---|
| CGL benchmarks | CGLB (NeurIPS 2022), BeGin (ACM TIST 2025) | Standardise task/class/domain/time-incremental settings | Taxonomy is about *learning scenarios*, not *how the graph changes*; no drift annotations |
| CGL methods (replay/regularisation) | ER-GNN, TWP (AAAI 2021); SSM (ICDM 2022); SEM (TNNLS 2024); CaT (ICDM 2023); TEM/PDGNN (KDD 2024); TACO (NeurIPS 2024); PUMA (TKDE 2025); DSLR (WWW 2024); FTF-ER (MM 2024); cohesive-subgraph replay (DCN 2026); KTR (KBS 2026); SPGNet (Displays, in press 2026) | Structure-aware memory: subgraphs, condensation, coarsening, topology-aware embeddings | Mostly snapshot **node classification**; given task boundaries; **reactive**; same replay rule for every kind of change |
| Continual learning on streaming/temporal graphs | ContinualGNN / "Streaming GNNs via Continual Learning" (CIKM 2020); PromptGNN (Appl. Intell. 2025); GCAL (PMLR 2025); Cognitive Bifurcation/DCDHippo (KDD 2026); OTGNet (ICLR 2023) **[verify]** | Adapt to evolving graphs with replay, prompts, distillation | No explicit, type-resolved drift detection; no taxonomy-driven policy |
| Drift-triggered updating on graph streams | Malialis et al. (IJCNN 2024) | Loss-based concept-drift detector triggers prototype update in graph-stream classification | One generic (loss) detector, no structural/type resolution, graph-level classification, not CTDG link prediction |
| Graph change-point detection | LAD (KDD 2020); SCPD (LNCS 2023); Fréchet-statistics CPD (TCSS 2024); LEM-CPD (Inf. Sci. 2023); SBM-CPD (JSTAT 2016); GraphScope (KDD 2007); SpotLight (KDD 2018); CPD survey (TKDE 2025) | Detect *when* the graph changed (spectral, statistical, latent) | No type classification; no learner is adapted after the alarm |
| Graph shift vocabularies | Distributionally-robust HGNN with relation-aware grouping (IPM, in press 2026); structural-semantics UGDA (KBS 2025) | Name structural / feature / relational-semantic shift | Static OOD/domain adaptation, not streaming; not operationalised as online tests |
| Temporal graph learning & benchmark | TGB (NeurIPS 2023); TGAT, TGN, JODIE, DyRep, GraphMixer, DyGFormer **[verify / see TGN note]** | Strong CTDG link predictors and protocol | Assume a single stationary train/test split; no forgetting or drift evaluation |

**Gap (as stated in both of your surveys):** no framework jointly provides (G1) a graph-native drift taxonomy, (G2) detection-triggered continual adaptation, and (G3) a benchmark with annotated drift types; relational-semantic drift is "recognised conceptually but underexplored empirically" (Survey 1, gap 8).

### 2.2 Feature-level comparison (fill ✓/✗ only after reading each full text)

| Method | Graph-native drift taxonomy | Online detection | Type of drift identified | Response depends on type | Structure-aware memory | CTDG link prediction | No task boundaries |
|---|---|---|---|---|---|---|---|
| ER-GNN / TWP / SSM / CaT / TEM / TACO / PUMA | ✗ | ✗ | ✗ | ✗ | ✓ (most) | ✗ | ✗ |
| ContinualGNN (2020) | ✗ | partial (pattern change) | ✗ | ✗ | partial | partial | ✓ |
| PromptGNN / GCAL / DCDHippo | ✗ | ✗ / partial | ✗ | ✗ | partial | check | check |
| Malialis 2024 | ✗ | ✓ (loss) | ✗ | ✗ | ✗ (prototypes) | ✗ | ✓ |
| LAD / SCPD / Fréchet / LEM-CPD | ✗ | ✓ | ✗ | n/a (no learner) | n/a | n/a | ✓ |
| **CSEE (ours)** | ✓ | ✓ | ✓ (α/β/γ/δ + mixtures) | ✓ | ✓ (k-core ego-nets) | ✓ | ✓ |

---

## 3. Objectives (Paper 1 only)

| ID | Objective | Deliverable | Success criterion |
|---|---|---|---|
| **O1** | Formalise a 4-factor relational drift taxonomy for CTDGs | Definitions 1–4, Proposition 1 (separability under stated assumptions) | Each injected pure drift moves only its own test statistic (orthogonality audit, §6.4) |
| **O2** | Design a type-resolved online drift detector | 4 statistics + CUSUM + FWER control; complexity linear in window edges | Detection F1 and type macro-F1 above every single-statistic detector |
| **O3** | Design drift-triggered, type-conditioned, structure-aware replay | k-core ego-net memory, type → policy map, stability–plasticity loss | Lower AF than equal-budget replay baselines without lower AA |
| **O4** | Integrate O2 + O3 into a backbone-agnostic continual learner (CSEE) | Plug-in around TGN and DyGFormer | Gains hold on ≥ 2 backbones |
| **O5** | Build GraphDriftBench v1 | Dynamic-SBM generator, semi-synthetic TGB injection, unified evaluation protocol, loaders | Public code, configs, seeds; one command reproduces each table |
| **O6** | Rigorous empirical validation | Main tables, ablations, sensitivity, efficiency, statistical tests | Pre-registered hypotheses H1–H5 (§8) tested and reported whatever the outcome |

---

## 4. The taxonomy (O1)

### 4.1 Setting

Partition the stream into consecutive windows $\mathcal{W}_k$ (fixed number $w$ of events, default $w$ chosen per dataset). Window $k$ induces $G_k=(V_k,E_k,X_k)$: active nodes, multigraph of events, and features (edge features $\mathbf{x}_e$; node features when available). For link prediction, the target $Y_k$ is the set of events of window $k+1$ (the next interactions). Write the window law as

$$
P_k(V,A,X,Y)=\underbrace{P_k(V)}_{\gamma}\;\underbrace{P_k(A\mid V)}_{\beta}\;\underbrace{P_k(X\mid A,V)}_{\alpha}\;\underbrace{P_k(Y\mid A,X,V)}_{\delta}.
$$

### 4.2 Definitions (a drift between reference regime $R$ and current window $k$)

- **Definition 1 — Type γ (entity turnover):** $P_k(V)\neq P_R(V)$; the support or activity distribution of nodes changes (arrivals, departures, identity churn) while the other three factors, restricted to persistent nodes, are unchanged.
- **Definition 2 — Type β (topological drift):** $P_k(A\mid V)\neq P_R(A\mid V)$ on persistent nodes $P=V_k\cap V_R$; e.g. densification, community merge/split, degree concentration.
- **Definition 3 — Type α (feature drift):** $P_k(X\mid A,V)\neq P_R(X\mid A,V)$; features change given the structure.
- **Definition 4 — Type δ (relational-semantic drift):** $P_k(Y\mid A,X,V)\neq P_R(Y\mid A,X,V)$; the rule that maps the current state to future interactions changes (who links to whom, given the same structure and features).
- **Mixed drift:** any non-empty subset of $\{\alpha,\beta,\gamma,\delta\}$. The taxonomy is a *basis*, not a partition of real events; reporting sets is part of the method.

**Why this is "orthogonal".** The four factors are distinct terms of one chain-rule factorisation, so a pure drift changes exactly one term. This is what we claim — not that real drifts are pure.

**Key observation for link prediction.** In CTDG link prediction the labels $Y$ arrive for free (future events become observed), so δ is detectable online through the prequential (test-then-train) loss — unlike node classification, where δ needs delayed human labels.

### 4.3 Proposition 1 (separability — to be proved in the paper)

Assume (A1) arrivals/departures are independent of the structure and features of persistent nodes (missing-at-random turnover); (A2) statistics $s_\beta, s_\alpha$ are computed on the persistent-node subgraph; (A3) $s_\alpha$ is stratified by structural strata (degree buckets) so it targets $P(X\mid A)$ rather than $P(X)$. Then, under a pure drift of type $c$, the null distributions of the statistics of the other types $c'\neq c$ are unchanged (for $c'\in\{\alpha,\beta,\gamma\}$), and the δ residual statistic (§5.1) is unchanged under pure α/β/γ drift to first order. **Status: conjecture until the proof is written; the orthogonality audit (§6.4) measures how far real data violate A1–A3.**

---

## 5. Method: CSEE (O2–O4)

### 5.1 Online type-resolved detector (O2)

Reference set $R$ = the last $r$ windows since the last alarm (reset after each alarm). For each window $k$ compute four statistics:

| Type | Statistic $s_c(k)$ | Computed on | Cost per window |
|---|---|---|---|
| γ | $s_\gamma = 1-\mathrm{J}(V_k^{\text{act}},V_R^{\text{act}})$ plus new-node rate $\lvert V_k\setminus V_R\rvert/\lvert V_k\rvert$ (max of standardised) | node sets | $O(\lvert V_k\rvert)$ |
| β | $s_\beta = W_1\big(\hat\rho_k,\bar\rho_R\big)$, Wasserstein-1 between spectral densities (density of states) of the normalised Laplacian, estimated by Kernel Polynomial Method with $M_c$ Chebyshev moments and $n_p$ Rademacher probes (as in SCPD) | persistent-node induced subgraph $G_k[P]$ | $O(M_c\, n_p\, \lvert E_k\rvert)$ |
| α | $s_\alpha = \sum_b \pi_b\,\widehat{\mathrm{MMD}}^2_{\ell}(X_k^{(b)},X_R^{(b)})$: linear-time MMD with an RBF kernel (median heuristic), stratified over degree-pair buckets $b$ | edge features of persistent-node events | $O(\lvert E_k\rvert)$ |
| δ | $s_\delta = \ell_k-\hat\ell_k$: prequential loss residual, where $\ell_k$ is the test-then-train loss (BCE, or $1-$MRR) and $\hat\ell_k=\mathbf{w}^\top[z_\alpha,z_\beta,z_\gamma,1]$ is the loss explained by the other three shifts, with $\mathbf{w}$ fitted on reference windows | model predictions | free (already computed) |

**Standardisation.** $z_c(k)=(s_c(k)-\mu_c^R)/\sigma_c^R$, with $\mu_c^R,\sigma_c^R$ from a moving-block bootstrap of the reference windows.

**CUSUM.** $S_c(k)=\max\{0,\,S_c(k-1)+z_c(k)-\kappa_c\}$; alarm for type $c$ when $S_c(k)>h_c$. Thresholds $h_c$ are calibrated on bootstrap replicates of the reference to a target in-control average run length $\mathrm{ARL}_0$, with Bonferroni over the four tests (family-wise false-alarm control).

**Output.** At window $k$: alarm flag and type set $F_k=\{c: S_c(k)>h_c\}$. Type δ is reported only when its *residual* statistic fires, so ordinary loss increases caused by α/β/γ are not mislabelled as δ.

**Proposition 2 (delay — classical result applied).** For each standardised statistic with post-change mean shift $\mu_c>\kappa_c$, CUSUM gives expected detection delay $\approx (h_c+O(1))/(\mu_c-\kappa_c)$ windows and in-control ARL growing exponentially in $h_c$ (Page 1954; Lorden 1971). We state the assumptions (approximate independence of windowed statistics after block bootstrap) and verify the delay empirically.

**Complexity.** Linear in window edges: $O\big((M_c n_p + d_x)\lvert E_k\rvert + \lvert V_k\rvert\big)$. No eigendecomposition.

**Alternative statistics (ablation only):** log-Euclidean Fréchet distance between Laplacians (TCSS 2024) and LAD-style singular-value embedding in place of DoS for β; Page–Hinkley / ADWIN on raw loss in place of the residual δ test.

### 5.2 Structure-aware memory (k-core ego-networks) (O3)

- **Unit of memory:** a sparsified ego-network $\mathcal{U}_c=(c,\,\mathcal{E}_c)$: the centre node $c$ and at most $b$ events among its 1-hop (optionally 2-hop) neighbourhood from recent windows, sampled by recency and edge-feature diversity (k-centre on $\mathbf{x}_e$). Memory budget $M$ events ⇒ at most $\lfloor M/b\rfloor$ units. Storage per unit is $O(b)$ — constant in graph size.
- **Centre selection:** core number $\mathrm{core}(c)$ in the window graph (Batagelj–Zaversnik, $O(\lvert E_k\rvert)$) combined with a coverage term:
  $\mathrm{score}(c)=\frac{\mathrm{core}(c)}{\max\mathrm{core}}+\eta\,\Delta\mathrm{Cov}(c\mid\mathcal{B})$, where $\Delta\mathrm{Cov}$ is the facility-location gain over node embeddings already in memory (greedy, submodular).
- **Partitions:** $\mathcal{B}=\mathcal{B}_{\text{cur}}\cup\mathcal{B}_{\text{arch}}$: current-regime units and archived (pre-drift) units.
- **k-truss:** deferred to an ablation / future extension (truss maintenance costs about $O(m^{1.5})$).

### 5.3 Type-conditioned refresh policy (the core novelty) (O3)

Memory is refreshed **only on an alarm**, and the detected type decides *what* is stored and *how strongly* it is replayed:

| Detected type | Memory action | Replay weight $\lambda_r$ | Distillation $\lambda_{kd}$ | Rationale |
|---|---|---|---|---|
| none | no refresh (stream training only) | $\lambda_0$ | 0 | save compute |
| γ | archive units centred on *departing* nodes + bridging high-core persistent nodes; warm-start memory states of new nodes from neighbour mean | ↑ | ↑ | knowledge of departing entities is what gets forgotten |
| β | archive pre-drift high-core units (protected fraction $\rho_\beta$); fill $\mathcal{B}_{\text{cur}}$ from post-drift k-cores | ↑ | ↑ | old topology changes message passing |
| α | keep units, re-select events inside units by feature k-centre; reset running feature-normalisation statistics | = | ↑ (embedding-level) | structure still valid, feature space moved |
| δ | move pre-drift units to $\mathcal{B}_{\text{arch}}$ as a *stale regime*; down-weight their replay; monitor archive loss for regime recurrence | ↓ ($\lambda_\delta$) | ↓ | the old rule is now wrong for the current regime; protect plasticity but keep it for recurrence |
| mixed | union of actions; weights = max over fired types (δ: min) | — | — | conservative |

### 5.4 Learning objective and loop (O4)

$$
\mathcal{L}(\theta)=\underbrace{\mathcal{L}_{\text{link}}(\mathcal{W}_k;\theta)}_{\text{current}}
+\lambda_r(F_k)\,\underbrace{\mathcal{L}_{\text{link}}(\mathcal{B};\theta)}_{\text{replay}}
+\lambda_{kd}(F_k)\,\underbrace{\frac{1}{\lvert\mathcal{B}\rvert}\sum_{c\in\mathcal{B}}\big\lVert h_\theta(c)-h_{\theta^-}(c)\big\rVert_2^2}_{\text{embedding distillation}}
$$

with $\mathcal{L}_{\text{link}}$ the backbone's BCE with negative sampling and $\theta^-$ a frozen copy taken at the alarm.

**Loop (test-then-train, no task boundaries):**
1. Receive window $k$ → predict → record prequential loss/MRR.
2. Detector update → $F_k$.
3. If $F_k\neq\emptyset$: apply policy (§5.3), snapshot $\theta^-$, run $E_{\text{alarm}}$ adaptation steps on mixed batches; reset reference $R$.
4. Train on window $k$ (+ replay with $\lambda_0$ if configured); update backbone memory states.

**Backbones:** TGN (primary; PyG `TGNMemory` / TGB example) and DyGFormer (via DyGLib) to show CSEE is backbone-agnostic.

---

## 6. GraphDriftBench v1 (O5)

### 6.1 Synthetic: continuous-time dynamic SBM (exact ground truth)

- $N$ nodes, $K$ blocks; node features $\mathbf{x}_v\sim\mathcal{N}(\boldsymbol\mu_{z_v},\sigma^2 I)$; edge features $\mathbf{x}_e$ = independent marks (so δ can be made pure).
- Events from Poisson processes; intensity for pair $(u,v)$: $\lambda_{uv}=B_{z_uz_v}\cdot\exp(\beta_h\,\mathrm{sim}(\mathbf{x}_u,\mathbf{x}_v))$ (+ optional recency/Hawkes term).
- **Pure injections at time $\tau$, magnitude $\varepsilon$:**
  - α: shift the edge-mark distribution ($\boldsymbol\mu_e\to\boldsymbol\mu_e+\varepsilon\,\mathbf{d}$) — $B$, node sets and link rule unchanged.
  - β: $B\to(1-\varepsilon)B+\varepsilon B'$ (community merge/split, densification), with $\beta_h$ unchanged.
  - γ: replace a fraction $\varepsilon$ of nodes with fresh IDs drawn from the same block/feature law.
  - δ: $B\to\Pi B\Pi^\top$ for a block permutation $\Pi$ among blocks with equal size and density (the graph law is preserved up to isomorphism, features unchanged, but which feature-groups link changes), and/or flip $\beta_h\to-\beta_h$ (homophily→heterophily).
- Also: mixed drifts, gradual (ramp) vs abrupt, and recurring regimes (A→B→A).
- Grid: $\varepsilon\in\{0.05,0.1,0.2,0.4\}$, 4 types + 3 mixed + recurring, 5 seeds.

### 6.2 Real streams (TGB)

| Dataset | Domain | Nodes | Events | Edge features | Role |
|---|---|---|---|---|---|
| `tgbl-wiki` | Wikipedia user–page edits | ≈9.2K | ≈157K | 172-d text (LIWC) | primary, fast iteration |
| `tgbl-review` | Amazon user–product reviews (e-commerce interactions) | ≈352K | ≈4.87M | 1-d rating | primary, scale |
| MOOC, LastFM (DyGLib collection) **[optional]** | education / music | small | small | — | robustness, if reviewers ask for > 2 real datasets |

(Confirm counts at load time from the TGB loader and report those values.) On an RTX 3070 (8 GB), `tgbl-coin`/`tgbl-comment` are out of scope.

### 6.3 Semi-synthetic injection into TGB (ground-truth time and type)

Injected at random times $\tau$ (3 schedules × 5 seeds), the injection is hidden from the learner:

- **γ — identity churn:** after $\tau$, a fraction $\varepsilon$ of active source nodes are consistently re-mapped to fresh IDs. Structure law is preserved up to isomorphism.
- **δ — relational-semantic flip:** after $\tau$, destinations are consistently permuted inside degree-matched buckets (a bijection between items of similar popularity). Degree distribution and features are preserved; the user→item rule changes.
- **α — feature shift:** `tgbl-wiki`: add $\varepsilon\sigma\mathbf{d}$ along a random direction / random rotation of edge features; `tgbl-review`: remap ratings for a fraction $\varepsilon$ of events.
- **β — structural burst / rewiring:** inject a dense community burst among a random subset of persistent nodes (SpotLight-style), or rewire a fraction $\varepsilon$ of events to a sharpened popularity law.
- Natural (un-injected) streams are also run: they give CL metrics and a qualitative case study of detected alarms, but no detection F1.

### 6.4 Orthogonality audit (new, required)

For every pure injection, report the $4\times4$ matrix of mean standardised statistic shifts $\mathbb{E}[z_{c'}\mid\text{inject } c]$. A diagonal-dominant matrix supports Proposition 1; off-diagonal leakage (expected for β on real data) is reported honestly and measured against $\varepsilon$.

---

## 7. Evaluation protocol

### 7.1 Streaming continual protocol

- Split each stream chronologically into $T=10$ segments $S_1..S_T$ (segment boundaries are for **evaluation only**; the learner never sees them).
- Within each segment, the last 15% of events are a held-out segment test set (never trained on).
- After processing segment $j$, evaluate MRR on the held-out test of every segment $i\le j$ → accuracy matrix $A_{j,i}$. Negatives: TGB negative-sampling strategy (historical + random) via the TGB negative generator; the same negatives for every method.

### 7.2 Metrics (corrected definitions)

- **Average Accuracy:** $\mathrm{AA}=\frac1T\sum_{i=1}^{T}A_{T,i}$
- **Average Forgetting:** $\mathrm{AF}=\frac{1}{T-1}\sum_{i=1}^{T-1}\Big(\max_{l\in\{i,\dots,T-1\}}A_{l,i}-A_{T,i}\Big)$ (≥ 0; lower is better)
- **Backward Transfer:** $\mathrm{BWT}=\frac{1}{T-1}\sum_{i=1}^{T-1}\big(A_{T,i}-A_{i,i}\big)$ (negative = forgetting)
- **Prequential MRR:** mean test-then-train MRR over the whole stream (plasticity).
- **TGB official test MRR** on the official final split, with the TGB evaluator, for comparability with the TGB leaderboard.
- **Detection:** precision, recall, F1 with tolerance $\Delta_{\text{tol}}$ windows; mean detection delay (events and windows); false alarms per $10^5$ events; **type identification** macro-F1 and the confusion matrix (including mixed).
- **Efficiency:** memory (MB and #events stored), #replay gradient steps, throughput (events/s), adaptation time per alarm, peak GPU memory.

### 7.3 Statistics

5 seeds; mean ± std; paired Wilcoxon signed-rank (CSEE vs best baseline, across seeds × streams); Friedman test + Nemenyi critical-difference diagram across all methods and streams. Equal memory budget $M$ for all replay methods ($M\in\{1\%,2\%,5\%\}$ of stream events).

---

## 8. Hypotheses (pre-registered) and what "better results" means

**Important:** we will *not* compare against numbers printed in other papers — their protocols, tasks and splits differ (most CGL baselines are node classification with task boundaries). Every baseline is re-run in our protocol, on the same backbone, budget, negatives and seeds. "Better" means better under this shared protocol. Each hypothesis is reported whether it holds or not.

| ID | Hypothesis | Test | Target |
|---|---|---|---|
| **H1** | The 4-test detector identifies drift type better than any single-statistic detector | type macro-F1, synthetic + semi-synthetic | CSEE > LAD, SCPD, Fréchet, LEM-CPD, loss-PH, with significance |
| **H2** | Type-conditioned k-core replay reduces forgetting at equal budget | AF, BWT vs ER-reservoir, ER-GNN, SSM, TEM-adapted | lower AF, AA not lower (non-inferiority margin 1 MRR point) |
| **H3** | Triggered refresh saves compute | #replay steps, time vs always-on replay | ≥ 50% fewer replay steps with ≤ 1 MRR point AA loss (target, not promise) |
| **H4** | k-core ego-net units beat random events at equal budget, mostly under β/γ | ablation A3 | AF gain largest on β/γ injections |
| **H5** | Down-weighting the obsolete regime after δ improves post-drift plasticity | prequential MRR in the 5 windows after δ | CSEE > uniform-replay variant |

Expected honest outcomes to state in the paper: on natural streams with mild drift, gains may be small; LEM-CPD may match CSEE on *when*-detection for β (it is not designed for *which type*); β injections on real data will show some leakage into α/δ.

---

## 9. Baselines (final list, all re-run in our protocol)

**A. Bounds and generic CL (same backbone):** Fine-tune (lower bound) · Joint retraining on all past data (upper bound) · EWC (PNAS 2017) · LwF (TPAMI 2018) · ER with reservoir sampling.

**B. Graph CL, adapted to CTDG link prediction** (their memory-selection rule applied to events/ego-nets; adaptation described in the appendix): ER-GNN · TWP · SSM · TEM/PDGNN · CaT (if condensation can be adapted to edge streams; otherwise discuss only).

**C. Continual learning on temporal/streaming graphs:** ContinualGNN (CIKM 2020) · PromptGNN (2025, if code is available) · OTGNet **[verify]** · drift-triggered ER using a single loss detector = re-implementation of the Malialis-2024 idea in our setting.

**D. Detectors (detection-only table and as swap-in triggers for CSEE):** LAD · SCPD · Fréchet-statistics CPD · LEM-CPD · Page–Hinkley and ADWIN on loss.

**E. Temporal backbones without CL (context row):** JODIE, DyRep, TGAT, TGN, GraphMixer, DyGFormer via DyGLib/TGB.

> **TGN citation note:** TGN is widely cited via arXiv only. Under the no-preprint rule, either cite its ICML 2020 Graph Representation Learning workshop record if one with a stable identifier exists, or cite TGB (which documents and benchmarks TGN) and describe TGN in text. Decide with your supervisor.

---

## 10. Ablations and analyses

| ID | Ablation | Question |
|---|---|---|
| A1 | Remove detector (always-on refresh every $n$ windows) | Is triggering useful? |
| A2 | Type-agnostic policy (same refresh for every alarm) | Does the taxonomy matter for the response? (key) |
| A3 | Random-event memory / random-node ego-nets instead of k-core | Does structure-aware selection matter? |
| A4 | Drop each test (−α, −β, −γ, −δ residual) | Contribution of each statistic |
| A5 | DoS → Fréchet / LAD for β; residual-δ → raw-loss PH | Choice of statistics |
| A6 | No distillation; no archive partition | Loss components |
| A7 | k-core → k-truss (on `tgbl-wiki` only) | Value of higher-order cohesion vs cost |
| S1 | Sensitivity: $w$, $r$, $h_c$/ARL$_0$, $b$, $M$, $\eta$, $\lambda$'s | Robustness |
| S2 | Drift magnitude $\varepsilon$ and gradual vs abrupt | Detection limits |
| S3 | Recurring regimes (A→B→A) | Is archived knowledge reused? |
| E1 | Runtime/memory vs dataset size | Scalability claim |

---

## 11. Revised novelty claims (defensible wording)

1. **A factorised, testable taxonomy of relational drift for CTDGs.** Four drift types defined as shifts in four factors of one factorisation, each with an online test and an empirical orthogonality audit. Prior CGL taxonomies (CGLB, BeGin) describe learning scenarios, not graph change; prior shift vocabularies are static/OOD.
2. **Type-resolved online detection.** Detection that returns *which* factor changed, with family-wise false-alarm control and linear cost, including a residual test that separates relational-semantic drift from loss increases caused by other drifts. Prior graph CPD detects *when* only.
3. **Type-conditioned, detection-triggered structure-aware replay.** The detected type decides what is archived and how strongly it is replayed. Prior drift-triggered graph-stream learning (Malialis 2024) uses one generic detector and one response; prior graph replay is not drift-aware.
4. **GraphDriftBench.** First benchmark joining CTDG link prediction, continual-learning metrics and ground-truth drift *type* annotations via type-pure synthetic and semi-synthetic injection.

Avoid: "first to couple drift detection with continual learning" (not true), "O(1) space" (it is $O(b)$ per unit), "mathematically orthogonal" without the factorisation qualifier, and any performance adjective before results exist.

---

## 12. Manuscript outline (TKDE format)

1. Introduction (motivating example of the four drift types on `tgbl-review`; contributions C1–C4)
2. Related work (§2 table → prose: CGL; streaming/temporal CL; graph CPD; temporal graph learning & benchmarks)
3. Problem formulation and taxonomy (Defs 1–4, Proposition 1)
4. CSEE (detector, memory, policy, objective, complexity, Proposition 2)
5. GraphDriftBench (generator, injection, protocol)
6. Experiments (setup; RQ1 detection; RQ2 forgetting/plasticity; RQ3 efficiency; RQ4 ablations; RQ5 orthogonality audit; case study on natural stream)
7. Discussion, limitations (assumptions A1–A3, β leakage, link prediction only), Paper 2 preview
8. Conclusion · Appendix: proofs, baseline adaptations, hyperparameters, full tables

---

## 13. Build order (engineering)

| Step | Module | Done when |
|---|---|---|
| **1** | Synthetic dynamic-SBM generator + 4 pure injectors + mixed/recurring | Unit tests pass incl. purity checks (e.g. γ and δ injections leave degree distribution KS p > 0.05 and DoS distance within null band) |
| **2** | Evaluation protocol: segments, accuracy matrix, AA/AF/BWT, detection metrics | Metrics match hand-computed toy matrices; AF ≠ BWT test |
| **3** | Detector: 4 statistics, bootstrap calibration, CUSUM, FWER | On synthetic: false-alarm rate ≤ target on stationary streams; delay curves vs ε |
| **4** | Memory: k-core, ego-net sparsifier, buffer, type-policy, CSEE trainer with TGN | Runs end-to-end on synthetic + `tgbl-wiki` within 8 GB VRAM |
| **5** | Baselines, semi-synthetic TGB injection, full experiments, ablations, tables | Every table regenerated from `results/*.csv` by one script |

### Validation gates (added 2026-10-28) — go/no-go before expensive work

| Gate | When | Question | Pass criterion | If it fails |
|---|---|---|---|---|
| **G1 — Separability** | after Step 1, before Step 3 is finalised | Can the four raw statistics tell the four pure drift types apart on synthetic streams? | 4×4 audit matrix is diagonal-dominant; simple type classifier (argmax of standardised z) macro-F1 ≥ 0.80 at ε = 0.2; δ residual does not fire on pure α/β/γ (≤ 15% of seeds) | Redesign the weak statistic (most likely δ residual or β); narrow the taxonomy claim (e.g. 3 identifiable types + δ "residual"); do not build replay yet |
| **G2 — Type-conditioning matters** | after a minimal Step 4 on synthetic + `tgbl-wiki` | Does the type-conditioned policy beat the same detector + memory with a type-agnostic policy at equal budget and equal replay steps? | lower AF (paired over 5 seeds) on at least 2 of 4 drift types, AA not lower | Reframe the contribution around detection + structure-aware memory; ablation A2 becomes a reported negative result |

Only after G1 and G2 pass: full baselines, `tgbl-review`, DyGFormer, all ablations.

Indicative timeline: Steps 1–2 weeks 1–3 · Step 3 weeks 4–6 · Step 4 weeks 7–10 · Step 5 weeks 11–18 · writing weeks 16–22.

---

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| β injection on real data leaks into other statistics | Report the audit; rely on synthetic for purity claims |
| δ residual model mis-specified → false δ alarms | Fit $\mathbf{w}$ robustly (Huber); ablate with raw-loss test |
| CGL baselines do not transfer to link prediction | Document the adaptation; include them in the appendix if they are not competitive for structural reasons |
| `tgbl-review` too slow on 8 GB GPU | Use neighbour sampling, smaller memory dim, fewer probes; report throughput honestly |
| Reviewer: "taxonomy is just a factorisation" | The contribution is operationalisation: tests, separability proposition, audit, and a policy that uses the type (ablation A2 proves it matters) |
| Results not better on natural streams | Report them anyway; main claims rest on controlled injections + efficiency |

---

## 15. Reference list (DOIs from your surveys; no preprints)

- Ko et al., BeGin, ACM TIST 2025. https://doi.org/10.1145/3702648
- Zhang, Song, Tao, CGLB, NeurIPS 2022 (Datasets & Benchmarks). Scopus 85147734063
- Zhang, Song, Tao, Sparsified Subgraph Memory (SSM), ICDM 2022. https://doi.org/10.1109/ICDM54844.2022.00177
- Zhang, Song, Tao, Ricci-curvature sparsification (SEM), IEEE TNNLS 2024. https://doi.org/10.1109/TNNLS.2023.3303454
- Zhou, Cao, ER-GNN, AAAI 2021. https://doi.org/10.1609/aaai.v35i5.16602
- Liu, Yang, Wang, TWP, AAAI 2021. https://doi.org/10.1609/aaai.v35i10.17049
- Liu, Qiu, Huang, CaT, ICDM 2023. https://doi.org/10.1109/ICDM58522.2023.00141
- Zhang et al., TEM/PDGNN, KDD 2024. https://doi.org/10.1145/3637528.3671732
- Han, Feng, Ning, TACO, NeurIPS 2024. Scopus 105000522297
- Liu et al., PUMA, IEEE TKDE 2025. https://doi.org/10.1109/TKDE.2024.3485691
- Choi et al., DSLR, WWW 2024. https://doi.org/10.1145/3589334.3645561
- Pang et al., FTF-ER, ACM MM 2024. https://doi.org/10.1145/3664647.3681457
- Xie et al., Cohesive-subgraph replay, Digital Communications and Networks 2026. https://doi.org/10.1016/j.dcan.2025.08.010
- Hu et al., KTR, Knowledge-Based Systems 2026. https://doi.org/10.1016/j.knosys.2026.116672
- Sun, Chen, Wang, SPGNet, Displays (in press, 2026). https://doi.org/10.1016/j.displa.2026.103641
- Wang et al., Streaming GNNs via Continual Learning (ContinualGNN), CIKM 2020. https://doi.org/10.1145/3340531.3411963
- Xu, Tong, Ji, PromptGNN, Applied Intelligence 2025. https://doi.org/10.1007/s10489-025-06888-2
- Qiao et al., GCAL, PMLR 2025. Scopus 105023639663
- Liang et al., Cognitive Bifurcation (DCDHippo), KDD 2026. https://doi.org/10.1145/3770855.3818129
- Malialis et al., Incremental learning with concept-drift detection for graph-stream classification, IJCNN 2024. https://doi.org/10.1109/IJCNN60899.2024.10650724
- Huang et al., LAD, KDD 2020. https://doi.org/10.1145/3394486.3403077
- Huang et al., SCPD (density of states), LNCS 2023. https://doi.org/10.1007/978-3-031-33374-3_2
- Luo, Krishnamurthy, Fréchet-statistics CPD, IEEE TCSS 2024. https://doi.org/10.1109/TCSS.2023.3297233
- Gong et al., LEM-CPD, Information Sciences 2023. https://doi.org/10.1016/j.ins.2023.119376
- de Ridder et al., SBM change points, JSTAT 2016. https://doi.org/10.1088/1742-5468/2016/11/113302
- Sun et al., GraphScope, KDD 2007. https://doi.org/10.1145/1281192.1281266
- Eswaran et al., SpotLight, KDD 2018. https://doi.org/10.1145/3219819.3220040
- Zhou et al., Survey of CPD in dynamic graphs, IEEE TKDE 2025. https://doi.org/10.1109/TKDE.2024.3523857
- Wang et al., Distributionally-robust HGNN (relation-aware grouping), IPM (in press, 2026). https://doi.org/10.1016/j.ipm.2026.105199
- Huang et al., Temporal Graph Benchmark (TGB), NeurIPS 2023. Scopus 85185584830
- Zhang et al., TGSL, CIKM 2023. https://doi.org/10.1145/3583780.3615081
- Carta et al., Catastrophic forgetting in deep graph networks, Frontiers in AI 2022. https://doi.org/10.3389/FRAI.2022.824655
- Kirkpatrick et al., EWC, PNAS 2017. https://doi.org/10.1073/pnas.1611835114
- Li, Hoiem, LwF, IEEE TPAMI 2018. https://doi.org/10.1109/TPAMI.2017.2773081
- Page, Continuous inspection schemes, Biometrika 1954. **[verify DOI]**
- Lorden, Procedures for reacting to a change in distribution, Ann. Math. Statist. 1971. **[verify DOI]**
- Batagelj, Zaversnik, An O(m) algorithm for cores decomposition of networks (2003/2011). **[verify venue/DOI]**
- Gretton et al., A kernel two-sample test, JMLR 2012. **[verify]**
- TGAT (ICLR 2020), JODIE (KDD 2019), DyRep (ICLR 2019), GraphMixer (ICLR 2023), DyGFormer (NeurIPS 2023), OTGNet (ICLR 2023), DyGLib/DyGFormer paper. **[verify; use the conference record, never arXiv]**
