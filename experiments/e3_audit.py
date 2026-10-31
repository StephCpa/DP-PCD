"""E3 - one-run canary audit: validity, power, and the M1 leak demonstration.

(a) Validity and power on canaries with a known guarantee: each canary's
    inclusion is released through a Gaussian mechanism that is exactly mu-GDP,
    i.e. (eps, 1e-6)-DP for eps in {1, 2, 4, 8}. The audit's 95% lower bound must
    stay below eps; how close it gets measures its power.
(b) Leak demonstration from the plan: M1 with DIAL-style private token
    selection. A canary's r creates a large log-ratio at a secret position p_i;
    without r the log-ratio is 0 everywhere. Releases are Gaussian with sigma
    calibrated for the claimed eps (ignoring selection). With public selection
    the audit must pass; with private selection it should fail - if it has the
    canaries to see it.

Writes results/e3_audit/{audit.json, tables.md}.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from dpcd import accounting as A
from dpcd import audit as AU

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "e3_audit")
os.makedirs(OUT, exist_ok=True)
DELTA = 1e-6


def gaussian_canaries(m, mu, rng):
    inc = rng.random(m) < 0.5
    return mu * inc + rng.normal(size=m), inc


def m1_canaries(m, eps_claim, private_selection, rng, T=64, k=8, c=1.0, spike=3.0, ctx_noise=0.05):
    """One release round of M1 on m canary records (each releases k of T tokens).

    The attacker knows each canary's secret position p_i; its score is the
    released value at p_i if p_i was released, else -inf.
    """
    mu = A.gdp_mu(eps_claim, DELTA)
    sigma = A.gaussian_sigma(k, c, mu)  # add/remove-r sensitivity c, k releases per record
    inc = rng.random(m) < 0.5
    p = rng.integers(0, T, m)
    scores = np.empty(m)
    for i in range(m):
        lr = np.zeros(T)
        if inc[i]:
            lr = rng.normal(0, ctx_noise, T)
            lr[p[i]] = spike
        if private_selection:
            sel = np.lexsort((rng.random(T), -np.abs(lr)))[:k]  # top-k |lr|, random ties
        else:
            sel = np.arange(k) * (T // k)  # public rule: fixed positions
        rel = np.clip(lr[sel], -c, c) + rng.normal(0, sigma, k)
        hit = np.where(sel == p[i])[0]
        scores[i] = rel[hit[0]] if len(hit) else -1e9
    return scores, inc


def main():
    rng = np.random.default_rng(7)
    res = {"validity": [], "leak": []}
    for eps in [1, 2, 4, 8]:
        mu = A.gdp_mu(eps, DELTA)
        for m in [128, 1024, 8192]:
            lbs = []
            for _ in range(20):
                s, inc = gaussian_canaries(m, mu, rng)
                lbs.append(AU.best_lower_bound_from_scores(s, inc, m, DELTA)[0])
            res["validity"].append(dict(eps=eps, m=m, mean_lb=float(np.mean(lbs)), max_lb=float(np.max(lbs)),
                                        ceiling=AU.max_certifiable_eps(m, DELTA)))
    for eps_claim in [1, 4]:
        for priv in [False, True]:
            for m in [128, 1024, 8192]:
                lbs = []
                for _ in range(5):
                    s, inc = m1_canaries(m, eps_claim, priv, rng)
                    lbs.append(AU.best_lower_bound_from_scores(s, inc, m, DELTA)[0])
                res["leak"].append(dict(eps_claim=eps_claim, private_selection=priv, m=m,
                                        mean_lb=float(np.mean(lbs)), min_lb=float(np.min(lbs)),
                                        fails=int(sum(l > eps_claim for l in lbs)), trials=len(lbs)))
    with open(os.path.join(OUT, "audit.json"), "w") as f:
        json.dump(res, f, indent=1)
    L = ["# E3 one-run audit\n", "## (a) Validity and power on exact (eps, 1e-6)-DP canaries (20 trials each)\n",
         "| true eps | canaries m | mean 95% lower bound | max lower bound | ceiling (perfect attacker) |",
         "|---|---|---|---|---|"]
    for r in res["validity"]:
        L.append(f"| {r['eps']} | {r['m']} | {r['mean_lb']:.2f} | {r['max_lb']:.2f} | {r['ceiling']:.2f} |")
    L += ["\n## (b) Leak demonstration: M1 with public vs private (DIAL-style) token selection (5 trials each)\n",
          "| claimed eps | selection | canaries m | mean lower bound | trials failing the audit |",
          "|---|---|---|---|---|"]
    for r in res["leak"]:
        L.append(f"| {r['eps_claim']} | {'private' if r['private_selection'] else 'public'} | {r['m']} | "
                 f"{r['mean_lb']:.2f} | {r['fails']}/{r['trials']} |")
    with open(os.path.join(OUT, "tables.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
