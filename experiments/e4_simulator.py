"""E4 - analytic categorical simulator: pre-registered predictions and direction 4.

Blocks (all at delta = 1e-6, add/remove-r, 3 evaluation seeds, knobs tuned on a
separate proxy world draw):
  A  main grid: worlds {state, global} x n {512, 4096, 16384} x eps {1, 4, 8}
     x mechanisms {m1_sampled, m1_full, m2_q0, m2_unif, m3_filter} plus the
     non-private references; answers 'Gaussian vs DP-Fusion equivalence',
     'q0 vs uniform anchor', 'sampled vs full feedback under DP' and '1/sqrt(n)'.
  B  sparse vs dense release (M1): k in {4, 16, 64} of L = 64 states, eps = 4.
  C  few strong vs many weak (M2): k in {4, 16, 64} tokens per record, eps = 4.
  E  filter vs fixed b as the record-specific divergence tau varies.

Writes results/e4_simulator/{e4.json, tables.md}.
"""
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from dpcd import simulator as S

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "e4_simulator")
os.makedirs(OUT, exist_ok=True)

DP_MECHS = ["m1_sampled", "m1_full", "m2_q0", "m2_unif", "m3_filter"]
NP_MECHS = ["np_full", "np_sampled", "np_imitate"]


def job(args):
    block, world_kw, mech, n, L, eps, k = args
    t = time.time()
    r = S.tune_and_eval(world_kw, mech, n, L, eps, k=k)
    r.update(block=block, world=world_kw, secs=time.time() - t)
    return r


def main():
    jobs = []
    for kind, n in itertools.product(["state", "global"], [512, 4096, 16384]):
        wk = {"kind": kind}
        for mech in NP_MECHS:
            jobs.append(("A", wk, mech, n, 16, np.inf, None))
        for mech, eps in itertools.product(DP_MECHS, [1, 4, 8]):
            jobs.append(("A", wk, mech, n, 16, eps, None))
    for k in [4, 16, 64]:
        jobs.append(("B", {"kind": "global"}, "m1_sampled", 4096, 64, 4, k))
        jobs.append(("B", {"kind": "state"}, "m1_sampled", 4096, 64, 4, k))
        jobs.append(("C", {"kind": "global"}, "m2_q0", 4096, 64, 4, k))
        jobs.append(("C", {"kind": "state"}, "m2_q0", 4096, 64, 4, k))
    for tau in [0.0, 0.25, 0.5]:
        wk = {"kind": "state", "tau": tau, "f_hot": 0.25}
        for mech in ["m2_q0", "m3_filter"]:
            jobs.append(("E", wk, mech, 4096, 16, 4, None))
        jobs.append(("E", wk, "np_full", 4096, 16, np.inf, None))
    # heavy jobs first for better packing
    jobs.sort(key=lambda j: -(j[3] * (3 if j[2] in ("m2_q0", "m3_filter") else 1)))
    t0 = time.time()
    with Pool(4) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(job, jobs)):
            res.append(r)
            print(f"[{i+1}/{len(jobs)} {time.time()-t0:.0f}s] {r['block']} {r['world']} {r['mech']} n={r['n']} "
                  f"k={r['k']} eps={r['eps']} H={r['H_mean']:.3f}+-{r['H_sd']:.3f}", flush=True)
    with open(os.path.join(OUT, "e4.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    write_tables(res)


def write_tables(res):
    def get(block, world, mech, n=None, eps=None, k=None, **wk):
        for r in res:
            if r["block"] != block or r["mech"] != mech or r["world"]["kind"] != world:
                continue
            if n is not None and r["n"] != n:
                continue
            if eps is not None and r["eps"] != eps:
                continue
            if k is not None and r["k"] != k:
                continue
            if any(r["world"].get(a) != b for a, b in wk.items()):
                continue
            return r
        return None

    fmt = lambda r: "-" if r is None else f"{r['H_mean']:.3f} ± {r['H_sd']:.3f}"
    L = ["# E4 analytic categorical simulator\n",
         "Utility H = share of headroom recovered (1 = the population teacher, 0 = the public prior q0, "
         "negative = worse than q0). Mean ± sd over 3 seeds; knobs tuned on a proxy world.\n"]
    for world in ["state", "global"]:
        L.append(f"\n## Block A, {world} world (L = 16 private tokens per record)\n")
        L.append("| mechanism | eps | n = 512 | n = 4,096 | n = 16,384 |")
        L.append("|---|---|---|---|---|")
        for mech in NP_MECHS:
            L.append(f"| {mech} | inf | " + " | ".join(fmt(get('A', world, mech, n)) for n in [512, 4096, 16384]) + " |")
        for mech in DP_MECHS:
            for eps in [1, 4, 8]:
                L.append(f"| {mech} | {eps} | " + " | ".join(fmt(get('A', world, mech, n, eps)) for n in [512, 4096, 16384]) + " |")
    L.append("\n## Block B, sparse vs dense M1 release at eps = 4 (n = 4,096, L = 64)\n")
    L.append("| world | k = 4 | k = 16 | k = 64 |")
    L.append("|---|---|---|---|")
    for world in ["global", "state"]:
        L.append(f"| {world} | " + " | ".join(fmt(get('B', world, 'm1_sampled', k=k)) for k in [4, 16, 64]) + " |")
    L.append("\n## Block C, few strong vs many weak M2 tokens at equal total budget, eps = 4 (n = 4,096)\n")
    L.append("| world | 4 tokens/record | 16 tokens/record | 64 tokens/record |")
    L.append("|---|---|---|---|")
    for world in ["global", "state"]:
        L.append(f"| {world} | " + " | ".join(fmt(get('C', world, 'm2_q0', k=k)) for k in [4, 16, 64]) + " |")
    L.append("\n## Block E, realized-divergence filter vs fixed b as record-specific divergence tau varies (eps = 4, n = 4,096, 25% hot states)\n")
    L.append("| tau | np_full | m2_q0 (fixed b) | m3_filter | m3 chosen cap multiplier g |")
    L.append("|---|---|---|---|---|")
    for tau in [0.0, 0.25, 0.5]:
        a = get('E', 'state', 'np_full', tau=tau)
        b = get('E', 'state', 'm2_q0', tau=tau)
        c = get('E', 'state', 'm3_filter', tau=tau)
        L.append(f"| {tau} | {fmt(a)} | {fmt(b)} | {fmt(c)} | {c['knob'] if c else '-'} |")
    with open(os.path.join(OUT, "tables.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "tables":
        write_tables(json.load(open(os.path.join(OUT, "e4.json"))))
    else:
        main()
