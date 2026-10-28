## small_bipartite — primary_first_post_window (mode=dev)

streams 15 (failed 0); macro-F1 argmax 0.914 (95% bootstrap CI 0.62-1.00) (thresholded z>3.0: 0.950); per-type F1 {'alpha': 0.8, 'beta': 1.0, 'gamma': 1.0, 'delta': 0.86}

Audit matrix: mean z of statistic (columns) given injected type (rows)

| | alpha | beta | gamma | delta |
|---|---|---|---|---|
| alpha | 3.02 | 0.78 | 0.13 | -0.66 |
| beta | 0.80 | 7.07 | 0.76 | -3.39 |
| gamma | -0.24 | 44.76 | 106.33 | 6.08 |
| delta | -0.22 | 0.15 | 0.27 | 14.65 |

95% bootstrap CI low / high

| | alpha | beta | gamma | delta |
|---|---|---|---|---|
| alpha | 0.97 | -1.27 | 0.00 | -2.45 |
| beta | 0.61 | 6.42 | 0.13 | -4.66 |
| gamma | -1.15 | 28.02 | 106.17 | -56.05 |
| delta | -1.51 | -0.68 | 0.00 | 9.52 |

| | alpha | beta | gamma | delta |
|---|---|---|---|---|
| alpha | 4.08 | 3.49 | 0.40 | 1.11 |
| beta | 1.08 | 7.76 | 1.89 | -2.69 |
| gamma | 1.07 | 62.39 | 106.66 | 72.60 |
| delta | 0.47 | 1.02 | 0.40 | 21.12 |

Confusion (argmax; rows = true)

| | alpha | beta | gamma | delta | none | tie |
|---|---|---|---|---|---|---|
| alpha | 2 | 0 | 0 | 1 | 0 | 0 |
| beta | 0 | 3 | 0 | 0 | 0 | 0 |
| gamma | 0 | 0 | 3 | 0 | 0 | 0 |
| delta | 0 | 0 | 0 | 3 | 0 | 0 |

Confusion (thresholded; 'none' = missed detection)

| | alpha | beta | gamma | delta | none | tie |
|---|---|---|---|---|---|---|
| alpha | 2 | 0 | 0 | 0 | 1 | 0 |
| beta | 0 | 3 | 0 | 0 | 0 | 0 |
| gamma | 0 | 0 | 3 | 0 | 0 | 0 |
| delta | 0 | 0 | 0 | 3 | 0 | 0 |

Criteria — (a) diagonal-dominant rows {'alpha': True, 'beta': True, 'gamma': True, 'delta': True} -> True; (b) macro-F1 >= 0.8: True; (c) delta fire on pure a/b/g 0.111 (by type {'alpha': 0.0, 'beta': 0.0, 'gamma': 0.3333333333333333}) <= 0.15: True; **pass_all = True**

Detection (z>3.0): precision 1.000, recall 0.917, missed 1, stationary false-alarm streams 0, first-detection window counts {0: 11, 1: 0, 2: 0}, missed within 3 windows 1, mean delay (windows) 0.00, false alarms per 1e5 events 0.00; delta-statistic AUROC (delta vs all others incl. stationary) 0.917

## small_bipartite — sensitivity_mean_first3_windows (mode=dev)

streams 15 (failed 0); macro-F1 argmax 1.000 (95% bootstrap CI 0.75-1.00) (thresholded z>3.0: 0.950); per-type F1 {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0, 'delta': 1.0}

Audit matrix: mean z of statistic (columns) given injected type (rows)

| | alpha | beta | gamma | delta |
|---|---|---|---|---|
| alpha | 4.08 | 0.12 | 0.09 | -0.84 |
| beta | 1.02 | 7.14 | 0.50 | -4.48 |
| gamma | -0.28 | 43.56 | 105.86 | 6.34 |
| delta | 0.07 | -0.06 | 0.13 | 9.98 |

95% bootstrap CI low / high

| | alpha | beta | gamma | delta |
|---|---|---|---|---|
| alpha | 1.75 | -0.84 | 0.00 | -2.11 |
| beta | 0.05 | 5.71 | 0.30 | -6.10 |
| gamma | -0.96 | 29.82 | 105.58 | -54.39 |
| delta | -1.41 | -0.70 | 0.00 | 6.07 |

| | alpha | beta | gamma | delta |
|---|---|---|---|---|
| alpha | 6.56 | 0.64 | 0.13 | 0.83 |
| beta | 2.29 | 8.89 | 0.73 | -3.34 |
| gamma | 0.69 | 57.99 | 106.14 | 71.64 |
| delta | 2.25 | 0.72 | 0.27 | 15.18 |

Confusion (argmax; rows = true)

| | alpha | beta | gamma | delta | none | tie |
|---|---|---|---|---|---|---|
| alpha | 3 | 0 | 0 | 0 | 0 | 0 |
| beta | 0 | 3 | 0 | 0 | 0 | 0 |
| gamma | 0 | 0 | 3 | 0 | 0 | 0 |
| delta | 0 | 0 | 0 | 3 | 0 | 0 |

Confusion (thresholded; 'none' = missed detection)

| | alpha | beta | gamma | delta | none | tie |
|---|---|---|---|---|---|---|
| alpha | 2 | 0 | 0 | 0 | 1 | 0 |
| beta | 0 | 3 | 0 | 0 | 0 | 0 |
| gamma | 0 | 0 | 3 | 0 | 0 | 0 |
| delta | 0 | 0 | 0 | 3 | 0 | 0 |

Criteria — (a) diagonal-dominant rows {'alpha': True, 'beta': True, 'gamma': True, 'delta': True} -> True; (b) macro-F1 >= 0.8: True; (c) delta fire on pure a/b/g 0.111 (by type {'alpha': 0.0, 'beta': 0.0, 'gamma': 0.3333333333333333}) <= 0.15: True; **pass_all = True**

Detection (z>3.0): precision 1.000, recall 0.917, missed 1, stationary false-alarm streams 0, first-detection window counts {0: 11, 1: 0, 2: 0}, missed within 3 windows 1, mean delay (windows) 0.00, false alarms per 1e5 events 0.00; delta-statistic AUROC (delta vs all others incl. stationary) 0.917
