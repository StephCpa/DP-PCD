"""E5 - miniature CPU pilot of all ten arms (toy stand-in for plan Phases 1-3).

Protocol (mirrors the plan at toy scale):
  * main task: n = 4,096 records (+256 canaries, each included w.p. 1/2),
    2 epochs, 64 problems per update, 4 rollouts, T = 4 answer tokens.
  * tuning: every arm gets 6 configurations at eps = 4 on a separate proxy task
    (2 seeds each); learning-rate grids are extended until the best is interior.
  * evaluation: eps in {1, 4, 8}; 5 training seeds at eps = 4, 3 otherwise.
  * metrics: held-out greedy exact match vs the shared rule (fresh x), canary
    one-run audit lower bound and AUROC, mechanism diagnostics.
  * G1 check: R2/R3 trained on half A, evaluated on half-A vs half-B problems.
  * selection-swap audit for the chunk selection (M2b/M3) and an M1 DIAL-style
    private selection as a contrast.
  * noise-geometry sweep: parameter dimension d via irrelevant features, and n.

Writes results/e5_pilot/{e5.json, tables.md}.
"""
import itertools
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from dpcd import audit as AU
from dpcd import pilot as P

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "e5_pilot")
os.makedirs(OUT, exist_ok=True)
N = 4096
M_CAN = 256
MAIN_TASK, PROXY_TASK = 0, 100

LR6 = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
GRIDS = {
    "R1": [dict(lr=l) for l in LR6], "R2": [dict(lr=l) for l in LR6], "R3": [dict(lr=l) for l in LR6],
    "B1": [dict(lr=l) for l in LR6], "B2": [dict(lr=l) for l in LR6],
    "M1": [dict(lr=l, clip_c=c) for l in [0.003, 0.01, 0.03] for c in [0.5, 2.0]],
    "M2a": [dict(lr=l, n_syn=s) for l in [0.003, 0.01, 0.03] for s in [1, 4]],
    "M2b": [dict(lr=l) for l in LR6],
    "M3": [dict(lr=l, g=g) for l in [0.003, 0.01, 0.03] for g in [2.0, 4.0]],
}
ARMS = ["R1", "R2", "R3", "B1", "B2", "M1", "M2a", "M2b", "M3"]
NONPRIV = {"R0", "R1", "R2", "R3"}


def data(task_seed, cfg, n, data_seed, with_canaries=True):
    task = P.make_task(cfg, task_seed)
    rng = np.random.default_rng(data_seed)
    x, r = P.make_records(task, n, rng)
    xe = P.sample_x(cfg, 2000, np.random.default_rng(data_seed + 1))
    ye = P.rule_answer(task, xe)
    can = None
    if with_canaries:
        crng = np.random.default_rng(data_seed + 2)
        xc = P.sample_x(cfg, M_CAN, crng)
        rc = crng.integers(0, cfg.K, (M_CAN, cfg.T))
        inc = crng.random(M_CAN) < 0.5
        x = np.concatenate([x, xc[inc]])
        r = np.concatenate([r, rc[inc]])
        can = (xc, rc, inc)
    return task, x, r, xe, ye, can


def run_one(args):
    arm, kw, eps, task_seed, data_seed, train_seed, cfg_kw, n, proxy = args
    cfg = P.TaskCfg(**cfg_kw)
    task, x, r, xe, ye, can = data(task_seed, cfg, n, data_seed, with_canaries=not proxy)
    rc = P.RunCfg(arm=arm, eps=eps, seed=train_seed, **kw)
    t = time.time()
    theta, log = P.train(task, rc, x, r, proxy=proxy)
    out = dict(arm=arm, kw=kw, eps=eps, n=n, cfg=cfg_kw, seed=train_seed, proxy=proxy, secs=time.time() - t,
               **P.evaluate(task, theta, xe, ye),
               lam=[float(l) for l in log["lambda"]],
               clip_frac=float(np.mean(log["clip_frac"])) if log["clip_frac"] else None,
               pp={k: float(v) for k, v in log["pp"].items()},
               ledger_mean_frac=log.get("ledger_mean_frac"))
    if can is not None:
        xc, rcn, inc = can
        s = P.canary_scores(task, theta, xc, rcn)
        out["canary_lb"] = AU.best_lower_bound_from_scores(s, inc, M_CAN, P.DELTA)[0]
        order = np.argsort(s)
        ranks = np.empty(len(s))
        ranks[order] = np.arange(1, len(s) + 1)
        npos = inc.sum()
        out["canary_auroc"] = float((ranks[inc].sum() - npos * (npos + 1) / 2) / (npos * (len(s) - npos)))
    return out


def tune(pool, arms, cfg_kw, n, eps=4.0):
    """Return {arm: (best kw, lam_hat)} using 6 configs per arm on the proxy task."""
    best = {}
    pending = {a: list(GRIDS[a]) for a in arms}
    tried = {a: [] for a in arms}
    for _round in range(3):
        jobs = [(a, kw, eps, PROXY_TASK, 500, s, cfg_kw, n, True) for a in arms for kw in pending[a] for s in (0, 1)]
        if not jobs:
            break
        res = pool.map(run_one, jobs)
        for a in arms:
            for kw in pending[a]:
                rs = [x for x in res if x["arm"] == a and x["kw"] == kw]
                tried[a].append((float(np.mean([x["exact"] for x in rs])), kw,
                                 float(np.mean([x["lam"][-1] for x in rs])) if rs[0]["lam"] else None))
        new_pending = {}
        for a in arms:
            sc, kw, lam = max(tried[a], key=lambda z: z[0])
            best[a] = (kw, lam, sc)
            lrs = sorted({k["lr"] for _, k, _ in tried[a]})
            ext = []
            if kw["lr"] == lrs[0]:
                ext.append(dict(kw, lr=lrs[0] / 3))
            if kw["lr"] == lrs[-1]:
                ext.append(dict(kw, lr=lrs[-1] * 3))
            new_pending[a] = ext
        pending = new_pending
    return best, tried


def main():
    t0 = time.time()
    res = {"tuning": {}, "eval": [], "dsweep": [], "nsweep": [], "g1": [], "swap": {}}
    base_cfg = {}
    with Pool(4) as pool:
        best, tried = tune(pool, ARMS, base_cfg, N)
        res["tuning"] = {a: {"best_kw": best[a][0], "lam_hat": best[a][1], "proxy_exact": best[a][2],
                             "tried": [(s, k) for s, k, _ in tried[a]]} for a in ARMS}
        print(f"tuned in {time.time()-t0:.0f}s", {a: best[a][0] for a in ARMS}, flush=True)

        def kw_of(a, b=best):
            kw = dict(b[a][0])
            if a in ("M2a", "M2b", "M3"):
                kw["lam_hat"] = b[a][1]
            return kw

        jobs = [("R0", {}, math.inf, MAIN_TASK, 1, 0, base_cfg, N, False)]
        for a in ["R1", "R2", "R3"]:
            jobs += [(a, kw_of(a), math.inf, MAIN_TASK, 1, s, base_cfg, N, False) for s in range(3)]
        for a in ["B1", "B2", "M1", "M2a", "M2b", "M3"]:
            for eps in [1.0, 4.0, 8.0]:
                seeds = range(5) if eps == 4.0 else range(3)
                jobs += [(a, kw_of(a), eps, MAIN_TASK, 1, s, base_cfg, N, False) for s in seeds]
        res["eval"] = pool.map(run_one, jobs)
        print(f"main grid done {time.time()-t0:.0f}s", flush=True)

        # G1: train on half A (n/2), evaluate on half-A problems and on fresh problems
        res["g1"] = g1_check(pool, kw_of)

        # noise-geometry: parameter dimension (retuned per d) and n
        for p_pad in [240, 2032]:
            cfg_kw = {"p_pad": p_pad}
            arms = ["B1", "B2", "M1", "M2b", "M3"]
            bd, _ = tune(pool, arms, cfg_kw, N)
            js = []
            for a in arms:
                kw = dict(bd[a][0])
                if a in ("M2b", "M3"):
                    kw["lam_hat"] = bd[a][1]
                js += [(a, kw, 4.0, MAIN_TASK, 1, s, cfg_kw, N, False) for s in range(3)]
            res["dsweep"] += pool.map(run_one, js)
            # non-private reference at this d (retuned)
            bnp, _ = tune(pool, ["R2"], cfg_kw, N, eps=math.inf)
            res["dsweep"] += pool.map(run_one, [("R2", bnp["R2"][0], math.inf, MAIN_TASK, 1, s, cfg_kw, N, False)
                                                 for s in range(3)])
            res["dsweep"] += pool.map(run_one, [("R0", {}, math.inf, MAIN_TASK, 1, 0, cfg_kw, N, False)])
            print(f"d sweep p_pad={p_pad} done {time.time()-t0:.0f}s", flush=True)
        for n in [1024, 16384]:
            js = [(a, kw_of(a), 4.0, MAIN_TASK, 1, s, base_cfg, n, False)
                  for a in ["B1", "B2", "M1", "M2b", "M3"] for s in range(3)]
            js += [("R2", kw_of("R2"), math.inf, MAIN_TASK, 1, s, base_cfg, n, False) for s in range(3)]
            res["nsweep"] += pool.map(run_one, js)
            print(f"n sweep n={n} done {time.time()-t0:.0f}s", flush=True)
    res["swap"] = swap_audit()
    with open(os.path.join(OUT, "e5.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    write_tables(res)


def g1_check(pool, kw_of):
    jobs = []
    for a in ["R0", "R2", "R3"]:
        for s in range(3 if a != "R0" else 1):
            jobs.append((a, kw_of(a) if a != "R0" else {}, s))
    return pool.map(_g1_one, jobs)


def _g1_one(args):
    arm, kw, seed = args
    cfg = P.TaskCfg()
    task = P.make_task(cfg, MAIN_TASK)
    x, r = P.make_records(task, N, np.random.default_rng(1))
    xa, ra = x[: N // 2], r[: N // 2]
    xb = x[N // 2:]
    theta, _ = P.train(task, P.RunCfg(arm=arm, eps=math.inf, seed=seed, **kw), xa, ra)
    return dict(arm=arm, seed=seed,
                in_split=P.evaluate(task, theta, xa, P.rule_answer(task, xa))["exact"],
                half_b=P.evaluate(task, theta, xb, P.rule_answer(task, xb))["exact"])


def swap_audit():
    """Selections must not change when one record's r is swapped (same student, same RNG)."""
    cfg = P.TaskCfg()
    task = P.make_task(cfg, MAIN_TASK)
    x, r = P.make_records(task, 256, np.random.default_rng(3))
    r2 = r.copy()
    r2[0] = (r[0] + 1) % cfg.K  # swap record 0's privileged context
    rc = P.RunCfg(arm="M2b", eps=4.0, lam_hat=0.2)
    pp = P.privacy_params(rc, 256, cfg.T)
    out = {}
    for arm in ["M2b", "M3"]:
        rc = replace(rc, arm=arm)
        sels = []
        for rr in (r, r2):
            rng = np.random.default_rng(9)
            rows_x = np.repeat(x[:64], 4, 0)
            rows_r = np.repeat(rr[:64], 4, 0)
            rec = np.repeat(np.arange(64), 4)
            toks = P.rollout(task.theta0, rows_x, cfg.K, cfg.T, np.random.default_rng(10))
            G, ctoks, _, _ = P._chunk_step(task.theta0, task.theta0, rows_x, rows_r, toks, rec, np.arange(64), rc, pp,
                                          np.zeros(256), rng, cfg.K, cfg.T, cfg.beta, 0.2)
            sels.append((np.abs(G).sum(-1) > 0))  # positions selected for the private loss
        out[arm] = int((sels[0] != sels[1]).sum())
    # contrast: DIAL-style private selection for M1 (position of largest |log-ratio|)
    picks = []
    for rr in (r, r2):
        toks = P.rollout(task.theta0, x[:64], cfg.K, cfg.T, np.random.default_rng(10))
        z0, zq = P.teacher_logits(task.theta0, x[:64], toks, rr[:64], cfg.beta, cfg.K)
        ii = (np.arange(64)[:, None], np.arange(cfg.T)[None, :], toks)
        lr_ = P.log_softmax(zq)[ii] - P.log_softmax(z0)[ii]
        picks.append(np.abs(lr_).argmax(1))
    out["M1_private_selection"] = int((picks[0] != picks[1]).sum())
    return out


def write_tables(res):
    ev = res["eval"]

    def cell(arm, eps, key="exact", rows=ev):
        xs = [r[key] for r in rows if r["arm"] == arm and r["eps"] == eps and r.get(key) is not None]
        if not xs:
            return "-"
        if len(xs) == 1:
            return f"{100*xs[0]:.1f}" if key == "exact" else f"{xs[0]:.2f}"
        m, sd = np.mean(xs), np.std(xs, ddof=1)
        return f"{100*m:.1f} ± {100*sd:.1f}" if key == "exact" else f"{m:.2f} ± {sd:.2f}"

    L = ["# E5 miniature CPU pilot (toy task, NOT the plan's LLM experiment)\n",
         f"n = {N} records + canaries, d = {P.TaskCfg().d} parameters, held-out exact match (%) on 2,000 fresh problems. "
         "Mean ± sd over training seeds (5 at eps = 4, 3 otherwise).\n",
         "| arm | eps = 1 | eps = 4 | eps = 8 | non-private |", "|---|---|---|---|---|"]
    for a in ["R0", "R1", "R2", "R3"]:
        L.append(f"| {a} | | | | {cell(a, math.inf)} |")
    for a in ["B1", "B2", "M1", "M2a", "M2b", "M3"]:
        L.append(f"| {a} | {cell(a, 1.0)} | {cell(a, 4.0)} | {cell(a, 8.0)} | |")
    # paired comparison at eps=4
    def paired(a, b, eps=4.0):
        xa = {r["seed"]: r["exact"] for r in ev if r["arm"] == a and r["eps"] == eps}
        xb = {r["seed"]: r["exact"] for r in ev if r["arm"] == b and r["eps"] == eps}
        d = np.array([xa[s] - xb[s] for s in xa if s in xb])
        if len(d) < 2:
            return None
        from scipy import stats
        h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
        return 100 * d.mean(), 100 * (d.mean() - h), 100 * (d.mean() + h)
    bm = {}
    for a in ["B1", "B2", "M1", "M2a", "M2b", "M3"]:
        xs = [r["exact"] for r in ev if r["arm"] == a and r["eps"] == 4.0]
        bm[a] = np.mean(xs)
    best_b = max(["B1", "B2"], key=lambda a: bm[a])
    best_m = max(["M1", "M2a", "M2b", "M3"], key=lambda a: bm[a])
    pr = paired(best_m, best_b)
    L.append(f"\nPrimary toy endpoint (eps = 4): best M arm = {best_m}, best B arm = {best_b}; "
             f"paired difference {pr[0]:+.1f} points, 95% CI [{pr[1]:+.1f}, {pr[2]:+.1f}] over 5 seeds.\n")
    L.append("## Canary audit (256 canaries, ~128 included; one-run 95% lower bound on eps, and AUROC)\n")
    L.append("| arm | eps | mean lower bound | max lower bound | mean AUROC |")
    L.append("|---|---|---|---|---|")
    for a in ["R1", "R2", "R3", "B1", "B2", "M1", "M2a", "M2b", "M3"]:
        for eps in ([math.inf] if a in NONPRIV else [4.0, 8.0]):
            rs = [r for r in ev if r["arm"] == a and r["eps"] == eps]
            if not rs:
                continue
            lbs = [r["canary_lb"] for r in rs]
            L.append(f"| {a} | {eps:g} | {np.mean(lbs):.2f} | {np.max(lbs):.2f} | "
                     f"{np.mean([r['canary_auroc'] for r in rs]):.3f} |")
    L.append("\n## Tuned configurations (proxy task, eps = 4)\n")
    L.append("| arm | chosen config | lam_hat (public) | configs tried |")
    L.append("|---|---|---|---|")
    for a, t in res["tuning"].items():
        lh = f"{t['lam_hat']:.3f}" if t["lam_hat"] is not None else "-"
        L.append(f"| {a} | {t['best_kw']} | {lh} | {len(t['tried'])} |")
    L.append("\n## G1 shared-component check (train on half A)\n")
    L.append("| arm | half-A problems (r seen) | half-B problems (r unseen) |")
    L.append("|---|---|---|")
    for a in ["R0", "R2", "R3"]:
        rs = [r for r in res["g1"] if r["arm"] == a]
        L.append(f"| {a} | {100*np.mean([r['in_split'] for r in rs]):.1f} | {100*np.mean([r['half_b'] for r in rs]):.1f} |")
    L.append("\n## Noise geometry: parameter dimension d (eps = 4, retuned per d, 3 seeds)\n")
    rows = res["eval"] + res["dsweep"]
    L.append("| arm | d = 790 | d = 10,390 | d = 82,070 |")
    L.append("|---|---|---|---|")
    for a in ["R0", "R2", "B1", "B2", "M1", "M2b", "M3"]:
        cells = []
        for pad in [0, 240, 2032]:
            eps = math.inf if a in NONPRIV else 4.0
            rs = [r["exact"] for r in rows if r["arm"] == a and r["eps"] == eps and r["cfg"].get("p_pad", 0) == pad
                  and r["n"] == N]
            cells.append(f"{100*np.mean(rs):.1f}" if rs else "-")
        L.append(f"| {a} | " + " | ".join(cells) + " |")
    L.append("\n## Records n (eps = 4, 3 seeds)\n")
    rows = res["eval"] + res["nsweep"]
    L.append("| arm | n = 1,024 | n = 4,096 | n = 16,384 |")
    L.append("|---|---|---|---|")
    for a in ["R2", "B1", "B2", "M1", "M2b", "M3"]:
        cells = []
        for n in [1024, 4096, 16384]:
            eps = math.inf if a in NONPRIV else 4.0
            rs = [r["exact"] for r in rows if r["arm"] == a and r["eps"] == eps and r["n"] == n
                  and not r["cfg"].get("p_pad")]
            cells.append(f"{100*np.mean(rs):.1f}" if rs else "-")
        L.append(f"| {a} | " + " | ".join(cells) + " |")
    L.append("\n## Mechanism diagnostics at eps = 4\n")
    L.append("| arm | noise / budget | mean lambda | clip fraction | M3 ledger used |")
    L.append("|---|---|---|---|---|")
    for a in ["B1", "B2", "M1", "M2a", "M2b", "M3"]:
        rs = [r for r in ev if r["arm"] == a and r["eps"] == 4.0]
        r0 = rs[0]
        nb = (f"z = {r0['pp']['z']:.3f}" if "z" in r0["pp"] else
              f"sigma/c = {r0['pp']['sigma']/r0['kw'].get('clip_c', 1):.1f} (R = {r0['pp']['R']:.0f})" if "sigma" in r0["pp"] else
              f"b = {r0['pp']['b']:.3f} at order {r0['pp']['alpha']:.0f}, {r0['pp']['n_tok']:.0f} tokens/record")
        lam = f"{np.mean([r['lam'][-1] for r in rs]):.3f}" if r0["lam"] else "-"
        cf = f"{np.mean([r['clip_frac'] for r in rs]):.2f}" if r0["clip_frac"] is not None else "-"
        lg = f"{np.mean([r['ledger_mean_frac'] for r in rs]):.2f}" if r0.get("ledger_mean_frac") is not None else "-"
        L.append(f"| {a} | {nb} | {lam} | {cf} | {lg} |")
    L.append(f"\n## Selection-swap audit\n\nChanged selections after swapping one r: {res['swap']}\n")
    with open(os.path.join(OUT, "tables.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "tables":
        write_tables(json.load(open(os.path.join(OUT, "e5.json"))))
    else:
        main()
