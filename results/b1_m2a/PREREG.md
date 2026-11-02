# DP-PCD — B1 (DP-SFT) vs M2a (DP-Fusion synthetic SFT, debiased) — EXPLORATORY
Written 2026-10-31 (PDT) BEFORE any training was launched. Code: run.py / drive.py in this dir.

**Status: EXPLORATORY.** This departs from the pre-registered G0 rule (G0 mixture-teacher condition failed:
best budget-feasible mixture +1.8 pts vs +5 required -> plan says stop signal-layer arms, run Phase 1 only).
It is reported as a separate exploratory experiment, not as part of the confirmatory plan.

## Setup
- Model Qwen3-1.7B (thinking off), LoRA r=16, alpha=32, on q,k,v,o,gate,up,down. Task GSM8K.
- Data (train shuffled with seed 0): private set = first n=4096 records (x=question, r=gold solution, calculator
  <<..>> annotations stripped); dev = next 500 (public; used ONLY for lr selection); proxy = next 500 (public;
  used ONLY to estimate lambda_hat). Final metric: greedy accuracy on full GSM8K test (1319), SE = binomial / paired bootstrap.
- Privacy: eps=4, delta=1e-6, add/remove one record.

## Arms
- **B1 DP-SFT**: SFT on (x -> r), DP-SGD with Poisson sampling q=256/4096, 4 epochs (64 steps), per-example clip C=1,
  noise multiplier from PLD accountant (dp_accounting), asserted eps(z) <= 4. AdamW on LoRA params.
- **M2a DP-Fusion synthetic SFT (debiased)**: for each private record, release n_syn=2 synthetic solutions sampled
  token-by-token from mix = lam*q_r + (1-lam)*q0 (full vocab), where lam is the largest value with two-sided
  Renyi(alpha=7) cost mu2_t = 2*D_7/7 <= b = mu^2/(n_syn*MAXNEW), mu = mu_rdp(4,1e-6) = 0.788, MAXNEW = 320.
  This fixed bound is valid regardless of realized length (no mean-length fitting). Per-record ledger kept in
  float64 (costs recomputed in float64); assertion ledger <= mu^2 with zero tolerance, run aborts otherwise.
  Student (non-private post-processing) minimizes -log(lam_hat*pi(y) + (1-lam_hat)*q0(y)) per token,
  q0 = frozen base. lam_hat = mean realized lam from running the same mechanism on the public proxy split at eps=4.

## Procedure (order fixed)
1. B1 lr sweep, seed 0, grid {3e-4, 1e-3, 3e-3, 1e-2}, selected on dev accuracy. If best lr is on the grid edge,
   extend by x(1/3)/x3 on that side and repeat until the best is strictly interior (max 3 extensions; if still
   on edge, report as not-interior).
2. B1 at best lr, seeds 0,1,2 -> test.
3. M2a: lam_hat on proxy; synthetic release seed 0; student lr sweep {1e-4, 3e-4, 1e-3} (same interior rule) on dev,
   seed 0; then seeds 1,2 (new synthetic release + training seed each) -> test.
4. Base q0 test accuracy reported for reference.

## Stopping rule (fixed in advance)
Primary: Delta = mean over 3 seeds of (M2a test acc - tuned B1 test acc), paired 95% CI by bootstrap over test
questions (seed-averaged per question). **If the CI lower bound is not > 0 (M2a does not beat tuned B1 at eps=4),
stop all signal-layer arms and write up the negative result.** No eps=1/8, no other n, no MedMCQA (deferred) in this run.
