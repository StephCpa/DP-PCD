# G0: divergence profile, public-nucleus variant, mixture-teacher accuracy (31 Oct 2026)

Inference only. Qwen3-1.7B (bf16, thinking off) on one A6000. Privileged context r = the GSM8K gold worked solution, placed in the prompt.
Units: μ_t² = 2·D_α/α, using the two-sided Rényi divergence at the RDP order for each ε (α = 22 / 7 / 4 for ε = 1 / 4 / 8; δ = 1e-6).
The per-record budget at ε = 4 is μ²_rdp = 0.788² ≈ 0.621.

Files: `g0.py` + `analyze.py` (first pass, full vocab) and `g0b.py` + `analyze_b.py` (nucleus variant + mixture accuracy). The JSON reports hold the outputs. Per-token npz profiles are kept off-repo.

## 1. Pre-registered G0 (full vocabulary): fails the 3× filter gate
- Headroom on GSM8K test[:500], greedy decoding: q0 77.8%, teacher with r 94.4%.
- Profile: GSM8K train (shuffle seed 0) 500 records × 4 rollouts at T = 1, 502k tokens.
- At ε = 4, the median μ_t² is 0.10.
- At ε = 4, the hot share (μ_t² > 0.01 / 0.03 / 0.1) is 57% / 54% / 50%.
- Filter gain over fixed b2 (0.01 / 0.03 / 0.1) is 1.69× / 1.76× / 1.87× at ε = 4 and 2.3–2.7× at ε = 8. That is below 3× everywhere.
- In `g0_report.json`, the ε = 1 row used α = 10 instead of 22, so it is approximate. `g0b_report.json` recomputes it at α = 22.

## 2. New variant: "M3 with public nucleus support" (not a rescue of the gate)
Both q and q0 are restricted to q0's top-p nucleus and renormalised. The nucleus is a public support. Rollouts are sampled from the truncated q0.

| support | ε | median μ² | mean μ² | hot >0.01 / 0.03 / 0.1 | filter gain b2 = 0.01 / 0.03 / 0.1 | teacher mass outside nucleus (mean; median) |
|---|---|---|---|---|---|---|
| full | 1 | 0.272 | 0.351 | 98 / 96 / 85% | 1.02 / 1.02 / 1.07 | 0 |
| full | 4 | 0.101 | 0.538 | 57 / 54 / 50% | 1.69 / 1.76 / 1.87 | 0 |
| full | 8 | 1.4e-4 | 0.579 | 40 / 37 / 34% | 2.34 / 2.50 / 2.73 | 0 |
| p0.9 | 4 | 0 | 0.061 | 11.2 / 10.3 / 8.9% | 8.81 / 9.28 / 10.11 | 0.029; 0 |
| p0.95 | 4 | 0 | 0.081 | 14.2 / 13.2 / 11.5% | 6.92 / 7.24 / 7.86 | 0.022; 0 |
| p0.99 | 4 | 0 | 0.124 | 19.8 / 18.6 / 16.5% | 4.93 / 5.14 / 5.54 | 0.013; 0 |

For the nucleus supports, the ε = 1 and ε = 8 rows are nearly identical to ε = 4; see `g0b_report.json`. The median is 0 because on most tokens q0's nucleus is a single token.

## 3. Mixture-teacher accuracy at the ε = 4 budget (GSM8K test[:500])
Mechanism: DP-Fusion mixture λq + (1−λ)q0, with λ found by bisection so that the two-sided μ_t² at α = 7 stays ≤ the bound.
- `fixed_b2_budget`: b2 = 0.621 / 251 ≈ 0.0025, where 251 is the mean rollout length. This arm has no length cap; long answers overspend, which only favours it.
- `fixed_b2_0.01`: about 4× over budget at the mean length.
- `filter`: per-token bound = remaining record budget; it charges the realised cost and falls back to q0 once the budget is exhausted.

| decoding | q0 | fixed b2 = 0.0025 | fixed b2 = 0.01 | filter (budget 0.621) |
|---|---|---|---|---|
| full, greedy | 78.0 | 78.6 | 78.6 | 78.2 |
| full, sample T = 1 | 77.2 | 79.0 | 77.4 | 79.0 |
| p0.95, greedy | 78.0 | 78.6 | 78.8 | 77.6 |
| p0.95, sample T = 1 | 77.4 | 78.8 | 78.8 | 76.4 |

The unconstrained teacher scores 94.4. Gate: at least +5 points over q0. The best cell is +1.8 points (n = 500, so ±~1.8 pt SE). The gate fails for every variant.

## Decision
- Nucleus hot share falls well below 1/3: 11–20%, with 5–10× filter gain. That condition passes.
- The budget-feasible mixture teacher is ≈ q0 (+0–2 points), so the second condition fails.
- Drop M3, including the nucleus variant, and go to B1 vs M2a at n = 4,096. Note that M2a's sampling teacher at ε = 4 is about q0-strength, so M2a starts weak.
