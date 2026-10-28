## small_bipartite — primary_first_post_window (mode=official)

streams 120 (failed 0); macro-F1 argmax 0.938 (95% bootstrap CI 0.88-0.99) (thresholded z>3.0: 0.868); per-type F1 {'alpha': 0.97, 'beta': 0.95, 'gamma': 0.92, 'delta': 0.9}

Audit matrix: mean z of statistic (columns) given injected type (rows)

| | alpha | beta | gamma | delta |
|---|---|---|---|---|
| alpha | 4.42 | 0.73 | 0.45 | -0.16 |
| beta | -0.44 | 7.84 | 0.49 | -3.60 |
| gamma | 0.06 | 46.15 | 105.47 | 1.88 |
| delta | -0.10 | 0.41 | 0.29 | 10.37 |

95% bootstrap CI low / high

| | alpha | beta | gamma | delta |
|---|---|---|---|---|
| alpha | 3.49 | -0.12 | 0.25 | -1.34 |
| beta | -1.00 | 5.97 | 0.26 | -5.27 |
| gamma | -0.46 | 38.94 | 105.00 | -26.33 |
| delta | -0.69 | -0.38 | 0.10 | 7.63 |

| | alpha | beta | gamma | delta |
|---|---|---|---|---|
| alpha | 5.40 | 1.75 | 0.66 | 0.82 |
| beta | 0.11 | 10.16 | 0.80 | -2.19 |
| gamma | 0.62 | 55.88 | 105.81 | 33.84 |
| delta | 0.50 | 1.40 | 0.53 | 13.59 |

Confusion (argmax; rows = true)

| | alpha | beta | gamma | delta | none | tie |
|---|---|---|---|---|---|---|
| alpha | 19 | 1 | 0 | 0 | 0 | 0 |
| beta | 0 | 20 | 0 | 0 | 0 | 0 |
| gamma | 0 | 0 | 17 | 3 | 0 | 0 |
| delta | 0 | 1 | 0 | 19 | 0 | 0 |

Confusion (thresholded; 'none' = missed detection)

| | alpha | beta | gamma | delta | none | tie |
|---|---|---|---|---|---|---|
| alpha | 12 | 1 | 0 | 0 | 7 | 0 |
| beta | 0 | 18 | 0 | 0 | 2 | 0 |
| gamma | 0 | 0 | 17 | 3 | 0 | 0 |
| delta | 0 | 0 | 0 | 18 | 2 | 0 |

Criteria — (a) diagonal-dominant rows {'alpha': True, 'beta': True, 'gamma': True, 'delta': True} -> True; (b) macro-F1 >= 0.8: True; (c) delta fire on pure a/b/g 0.183 (by type {'alpha': 0.1, 'beta': 0.05, 'gamma': 0.4}) <= 0.15: False; **pass_all = False**

Detection (z>3.0): precision 0.986, recall 0.863, missed 11, stationary false-alarm streams 1, first-detection window counts {0: 69, 1: 5, 2: 4}, missed within 3 windows 2, mean delay (windows) 0.17, false alarms per 1e5 events 0.83; delta-statistic AUROC (delta vs all others incl. stationary) 0.889

## small_bipartite — sensitivity_mean_first3_windows (mode=official)

streams 120 (failed 0); macro-F1 argmax 0.950 (95% bootstrap CI 0.90-0.99) (thresholded z>3.0: 0.899); per-type F1 {'alpha': 1.0, 'beta': 0.98, 'gamma': 0.92, 'delta': 0.9}

Audit matrix: mean z of statistic (columns) given injected type (rows)

| | alpha | beta | gamma | delta |
|---|---|---|---|---|
| alpha | 4.69 | 0.38 | 0.40 | 0.10 |
| beta | -0.24 | 7.51 | 0.62 | -3.85 |
| gamma | -0.16 | 46.16 | 105.58 | 1.74 |
| delta | -0.01 | 0.36 | 0.30 | 7.13 |

95% bootstrap CI low / high

| | alpha | beta | gamma | delta |
|---|---|---|---|---|
| alpha | 3.75 | -0.18 | 0.28 | -0.81 |
| beta | -0.64 | 5.98 | 0.46 | -5.37 |
| gamma | -0.68 | 38.86 | 105.15 | -26.20 |
| delta | -0.42 | -0.15 | 0.15 | 5.20 |

| | alpha | beta | gamma | delta |
|---|---|---|---|---|
| alpha | 5.70 | 1.10 | 0.56 | 0.94 |
| beta | 0.15 | 9.28 | 0.80 | -2.53 |
| gamma | 0.41 | 56.49 | 105.86 | 33.53 |
| delta | 0.35 | 0.92 | 0.50 | 9.27 |

Confusion (argmax; rows = true)

| | alpha | beta | gamma | delta | none | tie |
|---|---|---|---|---|---|---|
| alpha | 20 | 0 | 0 | 0 | 0 | 0 |
| beta | 0 | 20 | 0 | 0 | 0 | 0 |
| gamma | 0 | 0 | 17 | 3 | 0 | 0 |
| delta | 0 | 1 | 0 | 19 | 0 | 0 |

Confusion (thresholded; 'none' = missed detection)

| | alpha | beta | gamma | delta | none | tie |
|---|---|---|---|---|---|---|
| alpha | 14 | 0 | 0 | 0 | 6 | 0 |
| beta | 0 | 19 | 0 | 0 | 1 | 0 |
| gamma | 0 | 0 | 17 | 3 | 0 | 0 |
| delta | 0 | 0 | 0 | 18 | 2 | 0 |

Criteria — (a) diagonal-dominant rows {'alpha': True, 'beta': True, 'gamma': True, 'delta': True} -> True; (b) macro-F1 >= 0.8: True; (c) delta fire on pure a/b/g 0.167 (by type {'alpha': 0.1, 'beta': 0.0, 'gamma': 0.4}) <= 0.15: False; **pass_all = False**

Detection (z>3.0): precision 1.000, recall 0.887, missed 9, stationary false-alarm streams 0, first-detection window counts {0: 69, 1: 5, 2: 4}, missed within 3 windows 2, mean delay (windows) 0.17, false alarms per 1e5 events 0.83; delta-statistic AUROC (delta vs all others incl. stationary) 0.882
