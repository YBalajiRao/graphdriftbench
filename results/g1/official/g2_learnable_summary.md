## g2_learnable — primary_first_post_window (mode=official)

streams 120 (failed 0); macro-F1 argmax 1.000 (95% bootstrap CI 1.00-1.00) (thresholded z>3.0: 0.972); per-type F1 {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0, 'delta': 1.0}

Audit matrix: mean z of statistic (columns) given injected type (rows)

| | alpha | beta | gamma | delta |
|---|---|---|---|---|
| alpha | 4.96 | 0.03 | 0.49 | -0.37 |
| beta | 0.18 | 9.92 | 1.39 | -1.36 |
| gamma | 0.57 | 40.23 | 100.44 | 7.22 |
| delta | 0.05 | 0.65 | 0.87 | 29.86 |

95% bootstrap CI low / high

| | alpha | beta | gamma | delta |
|---|---|---|---|---|
| alpha | 4.14 | -0.42 | 0.21 | -1.02 |
| beta | -0.61 | 8.18 | 0.91 | -3.37 |
| gamma | -0.14 | 35.33 | 96.61 | -14.74 |
| delta | -0.54 | -0.03 | 0.49 | 23.76 |

| | alpha | beta | gamma | delta |
|---|---|---|---|---|
| alpha | 5.93 | 0.51 | 0.80 | 0.33 |
| beta | 0.85 | 11.73 | 1.88 | 0.74 |
| gamma | 1.37 | 46.01 | 103.84 | 29.21 |
| delta | 0.56 | 1.45 | 1.39 | 36.12 |

Confusion (argmax; rows = true)

| | alpha | beta | gamma | delta | none | tie |
|---|---|---|---|---|---|---|
| alpha | 20 | 0 | 0 | 0 | 0 | 0 |
| beta | 0 | 20 | 0 | 0 | 0 | 0 |
| gamma | 0 | 0 | 20 | 0 | 0 | 0 |
| delta | 0 | 0 | 0 | 20 | 0 | 0 |

Confusion (thresholded; 'none' = missed detection)

| | alpha | beta | gamma | delta | none | tie |
|---|---|---|---|---|---|---|
| alpha | 16 | 0 | 0 | 0 | 4 | 0 |
| beta | 0 | 20 | 0 | 0 | 0 | 0 |
| gamma | 0 | 0 | 20 | 0 | 0 | 0 |
| delta | 0 | 0 | 0 | 20 | 0 | 0 |

Criteria — (a) diagonal-dominant rows {'alpha': True, 'beta': True, 'gamma': True, 'delta': True} -> True; (b) macro-F1 >= 0.8: True; (c) delta fire on pure a/b/g 0.233 (by type {'alpha': 0.0, 'beta': 0.1, 'gamma': 0.6}) <= 0.15: False; **pass_all = False**

Detection (z>3.0): precision 0.962, recall 0.950, missed 4, stationary false-alarm streams 3, first-detection window counts {0: 76, 1: 2, 2: 1}, missed within 3 windows 1, mean delay (windows) 0.05, false alarms per 1e5 events 5.00; delta-statistic AUROC (delta vs all others incl. stationary) 0.891

## g2_learnable — sensitivity_mean_first3_windows (mode=official)

streams 120 (failed 0); macro-F1 argmax 1.000 (95% bootstrap CI 1.00-1.00) (thresholded z>3.0: 0.987); per-type F1 {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0, 'delta': 1.0}

Audit matrix: mean z of statistic (columns) given injected type (rows)

| | alpha | beta | gamma | delta |
|---|---|---|---|---|
| alpha | 5.14 | 0.13 | 0.38 | -0.26 |
| beta | 0.29 | 9.32 | 1.25 | -0.74 |
| gamma | 0.26 | 40.29 | 100.66 | 6.14 |
| delta | 0.15 | 0.52 | 1.02 | 17.47 |

95% bootstrap CI low / high

| | alpha | beta | gamma | delta |
|---|---|---|---|---|
| alpha | 4.39 | -0.12 | 0.24 | -0.81 |
| beta | -0.07 | 8.20 | 0.97 | -2.38 |
| gamma | -0.17 | 35.38 | 96.98 | -15.64 |
| delta | -0.22 | -0.02 | 0.76 | 14.00 |

| | alpha | beta | gamma | delta |
|---|---|---|---|---|
| alpha | 6.15 | 0.40 | 0.53 | 0.32 |
| beta | 0.61 | 10.44 | 1.53 | 0.93 |
| gamma | 0.72 | 46.23 | 103.91 | 28.12 |
| delta | 0.52 | 1.10 | 1.37 | 20.92 |

Confusion (argmax; rows = true)

| | alpha | beta | gamma | delta | none | tie |
|---|---|---|---|---|---|---|
| alpha | 20 | 0 | 0 | 0 | 0 | 0 |
| beta | 0 | 20 | 0 | 0 | 0 | 0 |
| gamma | 0 | 0 | 20 | 0 | 0 | 0 |
| delta | 0 | 0 | 0 | 20 | 0 | 0 |

Confusion (thresholded; 'none' = missed detection)

| | alpha | beta | gamma | delta | none | tie |
|---|---|---|---|---|---|---|
| alpha | 18 | 0 | 0 | 0 | 2 | 0 |
| beta | 0 | 20 | 0 | 0 | 0 | 0 |
| gamma | 0 | 0 | 20 | 0 | 0 | 0 |
| delta | 0 | 0 | 0 | 20 | 0 | 0 |

Criteria — (a) diagonal-dominant rows {'alpha': True, 'beta': True, 'gamma': True, 'delta': True} -> True; (b) macro-F1 >= 0.8: True; (c) delta fire on pure a/b/g 0.250 (by type {'alpha': 0.0, 'beta': 0.15, 'gamma': 0.6}) <= 0.15: False; **pass_all = False**

Detection (z>3.0): precision 1.000, recall 0.975, missed 2, stationary false-alarm streams 0, first-detection window counts {0: 76, 1: 2, 2: 1}, missed within 3 windows 1, mean delay (windows) 0.05, false alarms per 1e5 events 5.00; delta-statistic AUROC (delta vs all others incl. stationary) 0.862
