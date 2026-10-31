# E0 budget table and accounting checks (delta = 1e-6)

| eps | mu (GDP, M1) | mu via RDP (M2/M3) | best order | SNR n=4,096 | SNR n=16,384 | tokens b2=0.01 GDP / RDP | tokens b2=0.1 GDP / RDP |
|---|---|---|---|---|---|---|---|
| 1 | 0.237 | 0.221 | 22 | 7.6 | 15.1 | 5.6 / 4.9 | 0.56 / 0.49 |
| 2 | 0.448 | 0.420 | 12 | 14.3 | 28.7 | 20.1 / 17.6 | 2.01 / 1.76 |
| 4 | 0.838 | 0.788 | 7 | 26.8 | 53.6 | 70.2 / 62.2 | 7.02 / 6.22 |
| 8 | 1.532 | 1.450 | 4 | 49.0 | 98.0 | 234.6 / 210.4 | 23.46 / 21.04 |

Only order 2 tracked: eps >= 12.43 at delta=1e-6 even with zero spend (classic conversion: 13.82).

## M1 noise per release (sigma/c)

| config | eps | R | sigma/c replace (plan) | sigma/c add/remove | per-step SNR replace | add/remove |
|---|---|---|---|---|---|---|
| note min (n=512, 100x32, 4 rollouts, 64 tok) | 1 | 1600 | 338.0 | 169.0 | 0.27 | 0.54 |
| note min (n=512, 100x32, 4 rollouts, 64 tok) | 4 | 1600 | 95.5 | 47.7 | 0.95 | 1.90 |
| note min (n=512, 100x32, 4 rollouts, 64 tok) | 8 | 1600 | 52.2 | 26.1 | 1.73 | 3.47 |
| Phase 2 default, k=16 tok/rollout | 1 | 75 | 73.2 | 36.6 | 0.87 | 1.75 |
| Phase 2 default, k=16 tok/rollout | 4 | 75 | 20.7 | 10.3 | 3.10 | 6.19 |
| Phase 2 default, k=16 tok/rollout | 8 | 75 | 11.3 | 5.7 | 5.66 | 11.32 |
| Phase 2 default, k=64 tok/rollout | 1 | 300 | 146.3 | 73.2 | 0.87 | 1.75 |
| Phase 2 default, k=64 tok/rollout | 4 | 300 | 41.3 | 20.7 | 3.10 | 6.19 |
| Phase 2 default, k=64 tok/rollout | 8 | 300 | 22.6 | 11.3 | 5.66 | 11.32 |
| Phase 2 default, all 1,024 tok | 1 | 4800 | 585.4 | 292.7 | 0.87 | 1.75 |
| Phase 2 default, all 1,024 tok | 4 | 4800 | 165.4 | 82.7 | 3.10 | 6.19 |
| Phase 2 default, all 1,024 tok | 8 | 4800 | 90.5 | 45.2 | 5.66 | 11.32 |

## DP-SGD noise multiplier, n=16,384, 300 steps, batch 64 (PLD)

| eps | z, add/remove (B1, residual B2) | z, replace-one (plain B2) |
|---|---|---|
| 1 | 0.844 | 0.863 |
| 4 | 0.579 | 0.584 |
| 8 | 0.461 | 0.464 |

## One-run audit ceiling (perfect attacker, 95%)

| canaries m | max certifiable eps |
|---|---|
| 64 | 3.04 |
| 128 | 3.74 |
| 256 | 4.44 |
| 512 | 5.14 |
| 1024 | 5.83 |
| 4096 | 7.21 |
| 16384 | 8.56 |

| claim | attacker accuracy | canaries needed to exceed the claim |
|---|---|---|
| 4 | 1.0 | 166 |
| 4 | 0.99 | 871 |
| 4 | 0.95 | > 1e6 |
| 4 | 0.9 | > 1e6 |
| 8 | 1.0 | 9153 |
| 8 | 0.99 | > 1e6 |
| 8 | 0.95 | > 1e6 |
| 8 | 0.9 | > 1e6 |
