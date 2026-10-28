## wiki_like — primary_first_post_window (mode=official)

streams 100 (failed 0); macro-F1 argmax 0.641 (95% bootstrap CI 0.59-0.68) (thresholded z>3.0: 0.568); per-type F1 {'alpha': 1.0, 'beta': 0.0, 'gamma': 0.65, 'delta': 0.92}

Audit matrix: mean z of statistic (columns) given injected type (rows)

| | alpha | beta | gamma | delta |
|---|---|---|---|---|
| alpha | 14.91 | -0.05 | -0.05 | -1.66 |
| beta | 0.04 | 0.24 | 5.59 | -0.61 |
| gamma | 0.22 | 22.40 | 130.03 | 7.67 |
| delta | 0.26 | -0.03 | 0.47 | 3.24 |

95% bootstrap CI low / high

| | alpha | beta | gamma | delta |
|---|---|---|---|---|
| alpha | 13.07 | -0.57 | -0.48 | -3.85 |
| beta | -0.48 | -0.18 | 4.70 | -1.32 |
| gamma | -0.31 | 19.06 | 111.26 | -7.97 |
| delta | -0.14 | -0.59 | -0.08 | 2.29 |

| | alpha | beta | gamma | delta |
|---|---|---|---|---|
| alpha | 17.03 | 0.54 | 0.35 | 0.63 |
| beta | 0.55 | 0.72 | 6.58 | 0.16 |
| gamma | 0.73 | 25.67 | 151.25 | 22.74 |
| delta | 0.62 | 0.48 | 1.03 | 4.32 |

Confusion (argmax; rows = true)

| | alpha | beta | gamma | delta | none | tie |
|---|---|---|---|---|---|---|
| alpha | 20 | 0 | 0 | 0 | 0 | 0 |
| beta | 0 | 0 | 20 | 0 | 0 | 0 |
| gamma | 0 | 0 | 20 | 0 | 0 | 0 |
| delta | 0 | 1 | 2 | 17 | 0 | 0 |

Confusion (thresholded; 'none' = missed detection)

| | alpha | beta | gamma | delta | none | tie |
|---|---|---|---|---|---|---|
| alpha | 20 | 0 | 0 | 0 | 0 | 0 |
| beta | 0 | 0 | 17 | 0 | 3 | 0 |
| gamma | 0 | 0 | 20 | 0 | 0 | 0 |
| delta | 0 | 0 | 0 | 8 | 12 | 0 |

Criteria — (a) diagonal-dominant rows {'alpha': True, 'beta': False, 'gamma': True, 'delta': True} -> False; (b) macro-F1 >= 0.8: False; (c) delta fire on pure a/b/g 0.283 (by type {'alpha': 0.2, 'beta': 0.05, 'gamma': 0.6}) <= 0.15: False; **pass_all = False**

Detection (z>3.0): precision 0.970, recall 0.812, missed 15, stationary false-alarm streams 2, first-detection window counts {0: 65, 1: 3, 2: 1}, missed within 3 windows 11, mean delay (windows) 0.07, false alarms per 1e5 events 2.33; delta-statistic AUROC (delta vs all others incl. stationary) 0.769

## wiki_like — sensitivity_mean_first3_windows (mode=official)

streams 100 (failed 0); macro-F1 argmax 0.638 (95% bootstrap CI 0.59-0.68) (thresholded z>3.0: 0.535); per-type F1 {'alpha': 1.0, 'beta': 0.0, 'gamma': 0.63, 'delta': 0.92}

Audit matrix: mean z of statistic (columns) given injected type (rows)

| | alpha | beta | gamma | delta |
|---|---|---|---|---|
| alpha | 14.78 | -0.01 | 0.31 | -1.83 |
| beta | 0.40 | 0.49 | 5.39 | -1.25 |
| gamma | 0.37 | 22.74 | 130.22 | 7.90 |
| delta | -0.08 | -0.16 | 0.56 | 2.36 |

95% bootstrap CI low / high

| | alpha | beta | gamma | delta |
|---|---|---|---|---|
| alpha | 12.92 | -0.31 | 0.04 | -4.02 |
| beta | 0.10 | 0.12 | 4.68 | -1.93 |
| gamma | -0.03 | 19.88 | 111.82 | -7.70 |
| delta | -0.39 | -0.40 | 0.25 | 1.64 |

| | alpha | beta | gamma | delta |
|---|---|---|---|---|
| alpha | 16.86 | 0.31 | 0.60 | 0.43 |
| beta | 0.74 | 0.97 | 6.18 | -0.44 |
| gamma | 0.82 | 25.98 | 151.02 | 22.90 |
| delta | 0.19 | 0.07 | 0.83 | 3.13 |

Confusion (argmax; rows = true)

| | alpha | beta | gamma | delta | none | tie |
|---|---|---|---|---|---|---|
| alpha | 20 | 0 | 0 | 0 | 0 | 0 |
| beta | 0 | 0 | 20 | 0 | 0 | 0 |
| gamma | 0 | 0 | 20 | 0 | 0 | 0 |
| delta | 0 | 0 | 3 | 17 | 0 | 0 |

Confusion (thresholded; 'none' = missed detection)

| | alpha | beta | gamma | delta | none | tie |
|---|---|---|---|---|---|---|
| alpha | 20 | 0 | 0 | 0 | 0 | 0 |
| beta | 0 | 0 | 19 | 0 | 1 | 0 |
| gamma | 0 | 0 | 20 | 0 | 0 | 0 |
| delta | 0 | 0 | 0 | 6 | 14 | 0 |

Criteria — (a) diagonal-dominant rows {'alpha': True, 'beta': False, 'gamma': True, 'delta': True} -> False; (b) macro-F1 >= 0.8: False; (c) delta fire on pure a/b/g 0.283 (by type {'alpha': 0.15, 'beta': 0.05, 'gamma': 0.65}) <= 0.15: False; **pass_all = False**

Detection (z>3.0): precision 0.985, recall 0.812, missed 15, stationary false-alarm streams 1, first-detection window counts {0: 65, 1: 3, 2: 1}, missed within 3 windows 11, mean delay (windows) 0.07, false alarms per 1e5 events 2.33; delta-statistic AUROC (delta vs all others incl. stationary) 0.766
