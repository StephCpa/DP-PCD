# E3 one-run audit

## (a) Validity and power on exact (eps, 1e-6)-DP canaries (20 trials each)

| true eps | canaries m | mean 95% lower bound | max lower bound | ceiling (perfect attacker) |
|---|---|---|---|---|
| 1 | 128 | 0.00 | 0.09 | 3.74 |
| 1 | 1024 | 0.11 | 0.22 | 5.83 |
| 1 | 8192 | 0.32 | 0.48 | 7.89 |
| 2 | 128 | 0.07 | 0.36 | 3.74 |
| 2 | 1024 | 0.45 | 0.88 | 5.83 |
| 2 | 8192 | 0.82 | 1.11 | 7.89 |
| 4 | 128 | 0.42 | 0.78 | 3.74 |
| 4 | 1024 | 1.18 | 1.91 | 5.83 |
| 4 | 8192 | 1.66 | 1.99 | 7.89 |
| 8 | 128 | 1.20 | 1.74 | 3.74 |
| 8 | 1024 | 2.24 | 2.64 | 5.83 |
| 8 | 8192 | 3.19 | 3.44 | 7.89 |

## (b) Leak demonstration: M1 with public vs private (DIAL-style) token selection (5 trials each)

| claimed eps | selection | canaries m | mean lower bound | trials failing the audit |
|---|---|---|---|---|
| 1 | public | 128 | 0.00 | 0/5 |
| 1 | public | 1024 | 0.04 | 0/5 |
| 1 | public | 8192 | 0.00 | 0/5 |
| 1 | private | 128 | 1.79 | 5/5 |
| 1 | private | 1024 | 3.66 | 5/5 |
| 1 | private | 8192 | 5.40 | 5/5 |
| 4 | public | 128 | 0.00 | 0/5 |
| 4 | public | 1024 | 0.00 | 0/5 |
| 4 | public | 8192 | 0.05 | 0/5 |
| 4 | private | 128 | 1.80 | 0/5 |
| 4 | private | 1024 | 3.66 | 0/5 |
| 4 | private | 8192 | 5.40 | 5/5 |
