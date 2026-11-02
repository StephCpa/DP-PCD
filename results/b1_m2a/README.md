# B1 (DP-SFT) vs M2a (DP-Fusion synthetic SFT, debiased) — EXPLORATORY, eps = 4, n = 4096

Pre-registered in `PREREG.md` (written before launch). This experiment departs from the G0 rule, which said to stop the signal-layer arms after the mixture-teacher gate failed, so it is reported as exploratory. Setup: Qwen3-1.7B with LoRA r16, GSM8K, delta = 1e-6, greedy decoding, full test set (1319). The run was on a6000 GPU 6 from 2026-10-31 to 11-02.

**Decision: STOP the signal-layer arms (pre-registered rule).** M2a − B1 = +1.4 pts, paired 95% CI [−0.1, +2.9], which does not exclude 0.

## B1 lr sweep (seed 0, dev 500; started on {3e-4…1e-2} and extended downward until the best lr was interior)
| lr | 1.1e-5 | **3.3e-5** | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 |
|---|---|---|---|---|---|---|---|
| dev acc | 78.4 | **81.0** | 67.2 | 61.6 | 58.8 | 43.2 | 0.0 |

DP-SGD settings: z = 1.032, eps = 3.9995 (PLD), q = 256/4096, 64 steps, C = 1.

## M2a student lr sweep (seed 0, dev 500)
| lr | 3.3e-5 | **1e-4** | 3e-4 | 1e-3 |
|---|---|---|---|---|
| dev acc | 81.8 | **82.2** | 79.4 | 31.8 |

Mechanism settings: b = 9.71e-4 (μ² units) per token, n_syn = 2, MAXNEW = 320. λ̂ = 0.643 (proxy). The realized mean λ on the private set was 0.641 in every seed. Synthetic-solution accuracy was 75.5–75.9%. The largest float64 ledger value was 0.542 against a budget of 0.6217, so nothing was overspent.

## Test (1319), 3 seeds
| arm | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| base q0 | | | | 77.7 ± 1.1 |
| B1 (lr 3.3e-5) | 76.0 | 76.6 | 75.4 | 76.0 |
| M2a (lr 1e-4) | 78.5 | 76.7 | 76.8 | 77.3 |

Paired bootstrap over questions (seed-averaged): Δ(M2a − B1) = +1.36 pts, 95% CI [−0.13, +2.88].

## Notes
- Neither arm beats the base model on test. B1 is 1.7 pts below base, and M2a is within noise of base (−0.4). M2a's small edge over B1 comes from B1 hurting accuracy, not from M2a transferring the privileged context.
- On dev, both tuned arms are above base dev (79.8%), but this does not carry over to test. Part of that gap is the winner's curse from lr selection.
- `res/m2a_lr0.0001_s0.oom.log` records one CUDA out-of-memory error (about 44 GB at peak) in the first attempt at that run. The code was unchanged and the rerun completed. The synthetic-release JSONs (about 10 MB each) are not included.
