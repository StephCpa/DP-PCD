# DP-PCD: DP privileged-context distillation experiments

Code, data and results for the plan "DP Privileged-Context Distillation: Experimental Plan" (31 Oct 2026) and the feasibility note `dp-opsd-feasibility.md`.

**Read [RESULTS.md](RESULTS.md) first.** It lists what ran, what is blocked, and the findings that change the plan.

## What ran here

E0–E5 ran in a session with 4 CPUs, no GPU, and no access to `huggingface.co`, so Qwen3-1.7B could not be loaded there. Everything in the plan that does not need the LLM was run in full, and a toy pilot stands in for the training phases; it is labelled as such everywhere. G0 was then run on Qwen3-1.7B on a GPU (inference only). Training (Phases 1–2) has not been run.

| Experiment | Plan item | Script | Results |
|---|---|---|---|
| E0 | Phase 0 accountant tests and budget table | `experiments/e0_budget.py` | `results/e0_budget/` |
| E1 | Phase 0 efficiency estimate (on synthetic profiles) | `experiments/e1_efficiency.py` | `results/e1_efficiency/` |
| E2 | Phase 3 filter witness (exact enumeration) | `experiments/e2_filter_witness.py` | `results/e2_filter_witness/` |
| E3 | Phase 3 one-run audit validation and leak demonstration | `experiments/e3_audit.py` | `results/e3_audit/` |
| E4 | Analytic categorical simulator (direction 4, ablation predictions) | `experiments/e4_simulator.py` | `results/e4_simulator/` |
| E5 | Toy CPU pilot of all ten arms with canary audits | `experiments/e5_pilot.py` | `results/e5_pilot/` |
| G0 | Phase 0 on Qwen3-1.7B (GSM8K): divergence profile, public-nucleus variant, mixture-teacher accuracy; inference only, run on a GPU outside this container | `results/g0/g0.py`, `results/g0/g0b.py` | `results/g0/` |

## Library (`dpcd/`)

- `accounting.py`: GDP, RDP-to-(ε, δ) conversion, Gaussian calibration, DP-SGD noise via the PLD accountant (add/remove and replace-one), and the budget table.
- `divergence.py`: Rényi and χ² divergences for categorical distributions, and the λ search for DP-Fusion mixing.
- `mechanisms.py`: model-agnostic M1 release, M2 fixed-bound sampling, and the M3 Rényi filter (per-record ledger, parallel-safe).
- `witness.py`: exact privacy loss of token-sampling mechanisms on toy LMs.
- `audit.py`: one-run canary audit (Steinke, Nasr and Jagielski, 2023), the audit ceiling, and the selection-swap check.
- `efficiency.py`: usable private tokens per record from a divergence profile. It takes `(q, q0, sampled token)` arrays, so it runs directly on LLM rollouts.
- `simulator.py`: the analytic categorical simulator.
- `pilot.py`: the toy end-to-end pilot (log-linear autoregressive student).

## Reproduce

```bash
python -m venv venv
```

```bash
venv/bin/pip install -r requirements.txt
```

```bash
venv/bin/python -m pytest -q tests
```

```bash
for e in e0_budget e1_efficiency e2_filter_witness e3_audit e4_simulator e5_pilot make_figures; do venv/bin/python experiments/$e.py; done
```

Run times on 4 CPUs: E0 3 min, E1 20 s, E2 1 min, E3 25 s, E4 about 22 min, E5 about 31 min (plus 40 s for `e5_pilot.py lamfix`, which re-measures λ̂ per ε for M2/M3 at ε = 1 and 8 when updating an existing e5.json; a fresh run measures it directly).
