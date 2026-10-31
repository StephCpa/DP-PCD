"""E1 - Phase 0 efficiency estimate on synthetic divergence profiles.

The real G0 measurement needs Qwen3-1.7B rollouts (blocked here: no model
access, no GPU). This script runs the same estimator on synthetic LLM-like
profiles so the decision rule is mapped out in advance:

  gain of the filter over fixed b_2 = b_2 / E[min(D_t, b_2)],

i.e. it depends only on how many tokens carry less divergence than b_2.

Synthetic profile: V = 512, peaked q0 (LLM-like, median top-1 ~ 0.85). A share
f of tokens are 'hot' (r points to a token q0 ranks 2-50, logit shift
s ~ U(2, 8)); the rest are 'cold' (r changes logits by N(0, tau^2) noise).
The student samples from q0.

Writes results/e1_efficiency/{e1.json, tables.md}.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from dpcd import efficiency as E
from dpcd.mechanisms import sample_rows

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "e1_efficiency")
os.makedirs(OUT, exist_ok=True)


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def synth_profile(n_tok, f_hot, tau_cold, alpha, rng, V=512):
    ranks = np.arange(1, V + 1)
    base = -1.3 * np.log(ranks)  # Zipf-like logits
    z0 = base[None, :] * rng.uniform(1.5, 4.0, (n_tok, 1)) + rng.normal(0, 0.5, (n_tok, V))
    for i in range(n_tok):
        z0[i] = z0[i, rng.permutation(V)]
    q0 = softmax(z0)
    hot = rng.random(n_tok) < f_hot
    order = np.argsort(-q0, axis=1)
    k = order[np.arange(n_tok), rng.integers(1, 50, n_tok)]
    shift = rng.normal(0, tau_cold, (n_tok, V)) * (~hot)[:, None]
    shift[np.arange(n_tok), k] += rng.uniform(2, 8, n_tok) * hot
    q = softmax(z0 + shift)
    a = sample_rows(q0, rng)
    return E.Profile.from_distributions(q, q0, a, alpha), q0.max(1)


def main():
    rng = np.random.default_rng(0)
    eps_list = [1, 4, 8]
    rows = []
    for tau in [0.05, 0.2, 0.5]:
        for f in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
            for eps in eps_list:
                from dpcd.accounting import mu_rdp
                alpha = mu_rdp(eps, 1e-6)[1]
                prof, top1 = synth_profile(3000, f, tau, alpha, rng)
                rep = E.report(prof, eps)
                rows.append({"tau_cold": tau, "f_hot": f, "eps": eps, "alpha": alpha,
                             "median_top1_q0": float(np.median(top1)),
                             "top10_share": E.concentration(prof.d_mu2, 0.1),
                             "median_d_mu2": float(np.median(prof.d_mu2)),
                             "report": rep})
    with open(os.path.join(OUT, "e1.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=float)

    L = ["# E1 efficiency estimate on synthetic profiles\n",
         "Gain = usable private tokens (and signal) of the realized-divergence filter over fixed b_2, "
         "at eps = 4 (the gain does not depend on eps). G0 needs >= 3x.\n",
         "| cold-token noise tau | hot share f | median per-token mu_t^2 | top-10% share of divergence | gain b2=0.01 | gain b2=0.03 | gain b2=0.1 | "
         "fixed-b tokens/record b2=0.01 | filter tokens/record b2=0.01 | Gaussian efficiency c=0.5 / 1 / 2 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["eps"] != 4:
            continue
        rep = r["report"]
        g = rep["gaussian"]
        L.append(f"| {r['tau_cold']} | {r['f_hot']} | {r['median_d_mu2']:.4f} | {r['top10_share']:.2f} | "
                 f"{rep['filter'][0.01]['gain_tokens_vs_fixed']:.1f} | {rep['filter'][0.03]['gain_tokens_vs_fixed']:.1f} | "
                 f"{rep['filter'][0.1]['gain_tokens_vs_fixed']:.1f} | {rep['fixed_b'][0.01]['tokens']:.0f} | "
                 f"{rep['filter'][0.01]['tokens']:.0f} | "
                 f"{g[0.5]['efficiency']:.2f} / {g[1.0]['efficiency']:.2f} / {g[2.0]['efficiency']:.2f} |")
    L.append("")
    L.append(f"median top-1 prob of q0: {np.median([r['median_top1_q0'] for r in rows]):.2f}")
    with open(os.path.join(OUT, "tables.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
