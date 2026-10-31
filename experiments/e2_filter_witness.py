"""E2 - Phase 3 filter witness (zero-GPU part of the M3 validity check).

For random toy LMs with extreme privileged contexts we enumerate every output
sequence and compute the exact privacy loss of the whole run, in both
directions, against its add/remove-r neighbour. The claim being tested:

    realized-divergence filter at order alpha with budget B
    => D_alpha(P || P0) <= B and D_alpha(P0 || P) <= B,
       hence (eps, delta)-DP with eps from RDP conversion at alpha.

Negative controls must fail: a filter whose last token overshoots the budget,
a filter that bounds the wrong Renyi order, and a one-sided filter.

Writes results/e2_filter_witness/{witness.json, tables.md}.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from dpcd import accounting as A
from dpcd import witness as W

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "e2_filter_witness")
os.makedirs(OUT, exist_ok=True)
DELTA = 1e-6


def main():
    rows = []
    configs = []
    for V, T in [(3, 8), (4, 7)]:
        for strength in [3.0, 6.0, 12.0]:
            for frac in [0.3, 1.0]:
                for seed in range(6):
                    configs.append((V, T, strength, frac, seed))
    alpha = 7.0  # optimal order at eps=4, delta=1e-6 (E0)
    B = 0.6
    b_max = 0.25
    eps_claim = A.rdp_to_eps([B], DELTA, orders=[alpha])[0]
    for (V, T, strength, frac, seed) in configs:
        lm = W.hashed_toy_lm(V, T, seed=1000 * seed + V, strength=strength, frac_private=frac)
        for policy in ["filter", "fixed", "overspend_filter", "wrong_order_filter", "one_sided_filter"]:
            kw = dict(budget=B, b_max=b_max)
            if policy == "fixed":
                kw = dict(b=b_max, n_private=int(B // b_max))
            lp, lp0, ledger = W.enumerate_mechanism(lm, policy, alpha, **kw)
            fwd, bwd = W.exact_renyi(lp, lp0, alpha)
            d_fwd, d_bwd = W.exact_delta(lp, lp0, eps_claim)
            rows.append(dict(V=V, T=T, strength=strength, frac=frac, seed=seed, policy=policy,
                             ledger=ledger, renyi_fwd=fwd, renyi_bwd=bwd,
                             ratio=max(fwd, bwd) / B, delta_fwd=d_fwd, delta_bwd=d_bwd,
                             mass_check=float(np.exp(lp).sum())))
    summary = {}
    for policy in ["filter", "fixed", "overspend_filter", "wrong_order_filter", "one_sided_filter"]:
        rs = [r for r in rows if r["policy"] == policy]
        summary[policy] = dict(
            n=len(rs),
            max_ratio=max(r["ratio"] for r in rs),
            median_ratio=float(np.median([r["ratio"] for r in rs])),
            violations=sum(r["ratio"] > 1 + 1e-9 for r in rs),
            max_ledger=max(r["ledger"] for r in rs),
            max_delta=max(max(r["delta_fwd"], r["delta_bwd"]) for r in rs),
            max_bwd_over_fwd=max(r["renyi_bwd"] / max(r["renyi_fwd"], 1e-12) for r in rs),
        )
    res = dict(alpha=alpha, budget=B, b_max=b_max, eps_claim=eps_claim, delta=DELTA, summary=summary, rows=rows)
    with open(os.path.join(OUT, "witness.json"), "w") as f:
        json.dump(res, f, indent=1)
    L = [f"# E2 filter witness: exact privacy loss vs ledger\n",
         f"Order alpha = {alpha:g}, per-record RDP budget B = {B}, per-token cap {b_max}; "
         f"claim eps = {eps_claim:.3f} at delta = 1e-6. {len(configs)} toy LMs (V^T up to 4^7 = 16,384 outputs).\n",
         "| policy | runs | violations (D_alpha > B) | max D_alpha / B | median D_alpha / B | max ledger | max exact delta at claimed eps |",
         "|---|---|---|---|---|---|---|"]
    for p, s in summary.items():
        L.append(f"| {p} | {s['n']} | {s['violations']} | {s['max_ratio']:.3f} | {s['median_ratio']:.3f} | "
                 f"{s['max_ledger']:.3f} | {s['max_delta']:.2e} |")
    with open(os.path.join(OUT, "tables.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
