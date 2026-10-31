# E5 miniature CPU pilot (toy task, NOT the plan's LLM experiment)

n = 4096 records + canaries, d = 790 parameters, held-out exact match (%) on 2,000 fresh problems. Mean ± sd over training seeds (5 at eps = 4, 3 otherwise).

| arm | eps = 1 | eps = 4 | eps = 8 | non-private |
|---|---|---|---|---|
| R0 | | | | 42.9 |
| R1 | | | | 56.7 ± 0.4 |
| R2 | | | | 58.4 ± 0.8 |
| R3 | | | | 57.1 ± 1.3 |
| B1 | 47.9 ± 0.4 | 49.6 ± 0.5 | 50.9 ± 0.3 | |
| B2 | 44.2 ± 0.2 | 46.8 ± 0.7 | 47.6 ± 0.6 | |
| M1 | 42.3 ± 0.8 | 43.9 ± 0.3 | 45.2 ± 0.7 | |
| M2a | 36.7 ± 1.0 | 45.0 ± 1.0 | 51.9 ± 0.5 | |
| M2b | 38.2 ± 2.3 | 42.9 ± 1.1 | 47.7 ± 0.5 | |
| M3 | 38.5 ± 1.9 | 43.2 ± 1.7 | 47.1 ± 0.4 | |

Primary toy endpoint (eps = 4): best M arm = M2a, best B arm = B1; paired difference -4.6 points, 95% CI [-6.2, -3.0] over 5 seeds.

## Canary audit (256 canaries, ~128 included; one-run 95% lower bound on eps, and AUROC)

| arm | eps | mean lower bound | max lower bound | mean AUROC |
|---|---|---|---|---|
| R1 | inf | 0.00 | 0.00 | 0.551 |
| R2 | inf | 0.00 | 0.00 | 0.517 |
| R3 | inf | 0.00 | 0.00 | 0.529 |
| B1 | 4 | 0.00 | 0.00 | 0.535 |
| B1 | 8 | 0.00 | 0.00 | 0.534 |
| B2 | 4 | 0.00 | 0.00 | 0.537 |
| B2 | 8 | 0.00 | 0.00 | 0.534 |
| M1 | 4 | 0.00 | 0.00 | 0.494 |
| M1 | 8 | 0.00 | 0.00 | 0.518 |
| M2a | 4 | 0.00 | 0.00 | 0.514 |
| M2a | 8 | 0.00 | 0.00 | 0.517 |
| M2b | 4 | 0.01 | 0.07 | 0.544 |
| M2b | 8 | 0.00 | 0.00 | 0.520 |
| M3 | 4 | 0.00 | 0.00 | 0.533 |
| M3 | 8 | 0.00 | 0.00 | 0.527 |

## Tuned configurations (proxy task, eps = 4)

| arm | chosen config | lam_hat (public) | configs tried |
|---|---|---|---|
| R1 | {'lr': 0.003} | - | 6 |
| R2 | {'lr': 0.01} | - | 6 |
| R3 | {'lr': 0.01} | - | 6 |
| B1 | {'lr': 0.003} | - | 6 |
| B2 | {'lr': 0.003} | - | 6 |
| M1 | {'lr': 0.001, 'clip_c': 0.5} | - | 8 |
| M2a | {'lr': 0.003, 'n_syn': 1} | 0.390 | 7 |
| M2b | {'lr': 0.003} | 0.156 | 6 |
| M3 | {'lr': 0.003, 'g': 2.0} | 0.144 | 7 |

## G1 shared-component check (train on half A)

| arm | half-A problems (r seen) | half-B problems (r unseen) |
|---|---|---|
| R0 | 41.4 | 42.3 |
| R2 | 59.4 | 55.0 |
| R3 | 60.4 | 54.4 |

## Noise geometry: parameter dimension d (eps = 4, retuned per d, 3 seeds)

| arm | d = 790 | d = 10,390 | d = 82,070 |
|---|---|---|---|
| R0 | 42.9 | 39.8 | 41.3 |
| R2 | 58.4 | 42.2 | 41.9 |
| B1 | 49.6 | 40.3 | 41.0 |
| B2 | 46.8 | 40.0 | 40.7 |
| M1 | 43.9 | 40.0 | 35.8 |
| M2b | 42.9 | 40.2 | 41.2 |
| M3 | 43.2 | 39.8 | 36.2 |

## Records n (eps = 4, 3 seeds)

| arm | n = 1,024 | n = 4,096 | n = 16,384 |
|---|---|---|---|
| R2 | 45.6 | 58.4 | 66.4 |
| B1 | 44.9 | 49.6 | 57.1 |
| B2 | 44.0 | 46.8 | 56.4 |
| M1 | 43.1 | 43.9 | 46.7 |
| M2b | 41.7 | 42.9 | 45.0 |
| M3 | 41.9 | 43.2 | 44.3 |

## Mechanism diagnostics at eps = 4

| arm | noise / budget | mean lambda | clip fraction | M3 ledger used |
|---|---|---|---|---|
| B1 | z = 0.708 | - | 1.00 | - |
| B2 | z = 0.732 | - | 1.00 | - |
| M1 | sigma/c = 6.8 (R = 32) | - | 0.86 | - |
| M2a | b = 0.544 at order 7, 4 tokens/record | 0.383 | - | - |
| M2b | b = 0.136 at order 7, 16 tokens/record | 0.153 | - | - |
| M3 | b = 0.136 at order 7, 16 tokens/record | 0.141 | - | 1.00 |

## Selection-swap audit

Changed selections after swapping one r: {'M2b': 0, 'M3': 0, 'M1_private_selection': 1}

