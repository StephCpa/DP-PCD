"""E0 - Phase 0 accountant checks and budget table.

Reproduces the plan's budget table with the library code, and adds the
accounting quantities the later phases need:
  * GDP mu vs the mu reachable through RDP conversion (M2/M3 are RDP-accounted),
  * per-record token caps under both,
  * M1 noise per release under add/remove-r (sensitivity c) vs replace-r (2c),
  * DP-SGD noise multipliers for B1/B2 at the Phase 2 defaults (PLD),
  * the ceiling of the one-run canary audit and the canaries a claim needs.

Writes results/e0_budget/{budget.json, tables.md}.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dp_accounting
from dp_accounting import pld

from dpcd import accounting as A
from dpcd import audit

DELTA = 1e-6
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "e0_budget")
os.makedirs(OUT, exist_ok=True)


def main():
    res = {"delta": DELTA}
    eps_list = [1, 2, 4, 8]

    # 1. budget table ------------------------------------------------------
    rows = A.budget_table(eps_list, DELTA)
    res["budget"] = [
        {"eps": r.eps, "mu_gdp": r.mu_gdp, "mu_rdp": r.mu_rdp, "rdp_order": r.rdp_order,
         "snr_replace": r.snr, "snr_addremove": {n: 2 * s for n, s in r.snr.items()},
         "tokens_gdp": r.tokens_gdp, "tokens_rdp": r.tokens_rdp}
        for r in rows
    ]
    res["order2_only_eps_floor"] = A.eps_from_single_order(0.0, 2.0, DELTA)
    res["order2_only_eps_floor_classic"] = A.eps_from_single_order(0.0, 2.0, DELTA, improved=False)

    # 2. M1 sigma per release -------------------------------------------------
    m1 = []
    # minimum experiment from the feasibility note, then the Phase 2 defaults
    configs = [
        ("note min (n=512, 100x32, 4 rollouts, 64 tok)", 512, 100, 32, 4, 64),
        ("Phase 2 default, k=16 tok/rollout", 16384, 300, 64, 4, 16),
        ("Phase 2 default, k=64 tok/rollout", 16384, 300, 64, 4, 64),
        ("Phase 2 default, all 1,024 tok", 16384, 300, 64, 4, 1024),
    ]
    for name, n, steps, b, ro, k in configs:
        epochs = steps * b / n
        R = epochs * ro * k
        for e in [1, 4, 8]:
            mu = A.gdp_mu(e, DELTA)
            m1.append({"config": name, "n": n, "epochs": epochs, "R": R, "eps": e,
                       "sigma_over_c_replace": A.gaussian_sigma(R, 2.0, mu),
                       "sigma_over_c_addremove": A.gaussian_sigma(R, 1.0, mu),
                       "per_step_snr_replace": A.per_step_snr(mu, b, epochs, 2.0),
                       "per_step_snr_addremove": A.per_step_snr(mu, b, epochs, 1.0)})
    res["m1"] = m1

    # 3. DP-SGD noise multipliers (B1, B2) at Phase 2 defaults ---------------
    n, steps, b = 16384, 300, 64
    q = b / n
    dpsgd = []
    for e in [1, 4, 8]:
        z_ar = A.dpsgd_noise_multiplier(e, DELTA, q, steps)
        # plain B2 under add/remove-r: x stays, so a record's clipped gradient is
        # replaced (g_r -> g_0), which is the REPLACE_ONE relation
        def eps_rep(z):
            acc = pld.PLDAccountant(neighboring_relation=dp_accounting.NeighboringRelation.REPLACE_ONE,
                                    value_discretization_interval=1e-4)
            acc.compose(dp_accounting.PoissonSampledDpEvent(q, dp_accounting.GaussianDpEvent(z)), steps)
            return acc.get_epsilon(DELTA)
        lo, hi = 0.3, 60.0
        for _ in range(40):
            mid = math.sqrt(lo * hi)
            lo, hi = (mid, hi) if eps_rep(mid) > e else (lo, mid)
        dpsgd.append({"eps": e, "sample_rate": q, "steps": steps,
                      "z_add_remove": z_ar, "z_replace_one": hi})
    res["dpsgd"] = dpsgd

    # 4. one-run audit power -------------------------------------------------
    res["audit_ceiling"] = {m: audit.max_certifiable_eps(m, DELTA) for m in [64, 128, 256, 512, 1024, 4096, 16384]}
    need = {}
    for claim in [4, 8]:
        for acc_ in [1.0, 0.99, 0.95, 0.9]:
            need[f"eps{claim}_acc{acc_}"] = audit.canaries_needed(claim, DELTA, accuracy=acc_)
    res["canaries_needed"] = need

    with open(os.path.join(OUT, "budget.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    write_md(res)


def write_md(res):
    L = []
    L.append("# E0 budget table and accounting checks (delta = 1e-6)\n")
    L.append("| eps | mu (GDP, M1) | mu via RDP (M2/M3) | best order | SNR n=4,096 | SNR n=16,384 | tokens b2=0.01 GDP / RDP | tokens b2=0.1 GDP / RDP |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in res["budget"]:
        L.append(f"| {r['eps']} | {r['mu_gdp']:.3f} | {r['mu_rdp']:.3f} | {r['rdp_order']:.0f} | "
                 f"{r['snr_replace'][4096]:.1f} | {r['snr_replace'][16384]:.1f} | "
                 f"{r['tokens_gdp'][0.01]:.1f} / {r['tokens_rdp'][0.01]:.1f} | "
                 f"{r['tokens_gdp'][0.1]:.2f} / {r['tokens_rdp'][0.1]:.2f} |")
    L.append(f"\nOnly order 2 tracked: eps >= {res['order2_only_eps_floor']:.2f} at delta=1e-6 even with zero spend "
             f"(classic conversion: {res['order2_only_eps_floor_classic']:.2f}).\n")
    L.append("## M1 noise per release (sigma/c)\n")
    L.append("| config | eps | R | sigma/c replace (plan) | sigma/c add/remove | per-step SNR replace | add/remove |")
    L.append("|---|---|---|---|---|---|---|")
    for m in res["m1"]:
        L.append(f"| {m['config']} | {m['eps']} | {m['R']:.0f} | {m['sigma_over_c_replace']:.1f} | "
                 f"{m['sigma_over_c_addremove']:.1f} | {m['per_step_snr_replace']:.2f} | {m['per_step_snr_addremove']:.2f} |")
    L.append("\n## DP-SGD noise multiplier, n=16,384, 300 steps, batch 64 (PLD)\n")
    L.append("| eps | z, add/remove (B1, residual B2) | z, replace-one (plain B2) |")
    L.append("|---|---|---|")
    for d in res["dpsgd"]:
        L.append(f"| {d['eps']} | {d['z_add_remove']:.3f} | {d['z_replace_one']:.3f} |")
    L.append("\n## One-run audit ceiling (perfect attacker, 95%)\n")
    L.append("| canaries m | max certifiable eps |")
    L.append("|---|---|")
    for m, e in res["audit_ceiling"].items():
        L.append(f"| {m} | {e:.2f} |")
    L.append("\n| claim | attacker accuracy | canaries needed to exceed the claim |")
    L.append("|---|---|---|")
    for k, v in res["canaries_needed"].items():
        claim, acc = k.split("_")
        L.append(f"| {claim[3:]} | {acc[3:]} | {v if v else '> 1e6'} |")
    with open(os.path.join(OUT, "tables.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
