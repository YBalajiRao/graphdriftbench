# Roadmap — Paper 2 and Paper 3 (plan only; do not implement until Paper 1 is submitted)

## Paper 2 — Predictive Structural World Models for Proactive Continual Graph Learning

**Working title:** *From Reactive to Proactive: A Structural World Model for Forecast-Driven Drift Detection and Risk-Aware Memory Allocation on Dynamic Graphs*

**Builds on Paper 1:** reuses GraphDriftBench, the taxonomy, the evaluation protocol and the CSEE replay engine as the reactive baseline.

### Objectives
- **P2-O1 — World model.** Define $W(t)=\langle \mathcal{Z}_V(t), \mathcal{S}(t), \Theta(t)\rangle$: latent node states, global structural summaries (spectral density, core profile, community statistics), and transition dynamics.
- **P2-O2 — Forecasting.** Predict $W(t+\Delta t)$: next-window structural summaries, likely links, node arrivals/departures.
- **P2-O3 — Forecast-surprise detection.** Drift = significant forecast error, decomposed by factor (α/β/γ/δ), unifying detection and forecasting (positioned against LEM-CPD, which predicts the next state for detection but does not adapt a learner).
- **P2-O4 — Proactive, risk-aware memory allocation.** Allocate memory *before* drift using forecast uncertainty; optimise a CVaR objective over forecast scenarios of forgetting, i.e. minimise the expected forgetting in the worst $q\%$ of forecasted futures under budget $M$.
- **P2-O5 — Evaluation.** Lead time (alarm before the true change), forecast quality (Hits@10, MRR, AUC on future windows), and the forgetting/plasticity gains of proactive vs reactive (Paper 1) allocation.

### Novelty (planned)
1. A structural world model whose forecast error *is* the type-resolved drift signal.
2. A proactive memory policy: memory is pre-allocated for the forecasted regime, not refreshed after damage.
3. CVaR-based, risk-aware replay allocation for graph continual learning.
4. GraphDriftBench v2: lead-time and forecast metrics added to v1.

### Baselines (planned)
Paper 1's CSEE (reactive) · LEM-CPD · TGSL (CIKM 2023) · TGN/DyGFormer forecasting heads · always-on replay · uniform allocation.

### Datasets
The same as Paper 1 (synthetic + `tgbl-wiki` + `tgbl-review`) plus drift schedules with precursors (gradual onsets), so that forecasting has signal.

### Key risk
Abrupt drift without precursors cannot be forecast. Restrict proactive claims to gradual/recurring drift and report abrupt drift separately.

---

## Paper 3 — Transfer to clinical dynamic graphs (healthcare)

- **Goal:** apply CSEE (+ Paper 2 world model if ready) to dynamic patient–encounter / care-pathway graphs, where drift comes from protocol changes, new treatments and population shift.
- **Data:** MIMIC-IV (requires PhysioNet credentialing + CITI human-subjects training; **start the application early**, it takes weeks) or PhysioNet physiological networks.
- **New issues:** graph construction from EHR, ethics/data use agreement, label delay (δ is not free in clinical labels), fairness across patient groups.
- **Planned contribution:** clinical drift taxonomy case study + delayed-label variant of the δ test.
