Preset `unipartite_small`, w=2000, eps=0.2, 20 seeds, null band q95 of 200 streams.
Cells: seeds counted / total. Null cells count seeds INSIDE the band (pass >= 17/20); shift cells count seeds OUTSIDE (pass >= 18/20); info cells have no requirement.

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
