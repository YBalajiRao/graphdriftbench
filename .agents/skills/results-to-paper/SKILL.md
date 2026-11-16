---
name: results-to-paper
description: Use when making tables, figures or any text that states a number.
---

# Results to paper

- Tables only from `results/*.csv` via `scripts/make_tables.py`: LaTeX booktabs, mean ± std over seeds, bold best, significance markers from paired Wilcoxon (CSEE vs best baseline) and Friedman (+ Nemenyi) as in Blueprint §7.3.
- Figures only via `scripts/make_figures.py`, reading `results/*.csv`.
- Any number in prose must be traceable to a CSV row; never round by hand or type a result from memory. Placeholders stay `[RESULTS]` until the CSV exists.
- Report hypotheses H1-H5 (Blueprint §8) whether they hold or not; state non-inferiority margins as specified.
- Never copy baseline numbers from other papers.
- Run the `experiment-auditor` agent before generating final tables.
