"""Sequential driver following PREREG.md. Resumable (skips existing result files)."""
import os, sys, json, subprocess, numpy as np
D = os.path.expanduser("~/dppcd_b1m2a"); R = f"{D}/res"; os.makedirs(R, exist_ok=True)
def run(args, out, ev):
    if os.path.exists(out): return json.load(open(out))
    lg = out.replace(".json", ".log")
    with open(lg, "w") as f:
        subprocess.run([sys.executable, f"{D}/run.py", *args, out], env={**os.environ, "EVAL": ev}, stdout=f, stderr=subprocess.STDOUT, check=True)
    return json.load(open(out))
def sweep(fn, grid, tag):
    grid = sorted(grid); res = {}
    for ext in range(4):
        for lr in grid:
            if lr not in res: res[lr] = fn(lr)["dev"]["acc"]
        best = max(grid, key=lambda l: res[l]); i = grid.index(best)
        print(tag, {f"{k:g}": v for k, v in sorted(res.items())}, "best", best, flush=True)
        if 0 < i < len(grid)-1: return best, True, res
        if ext == 3: return best, False, res
        grid = sorted(grid + [best/3 if i == 0 else best*3])
S = {}
lr_b1, interior, sw = sweep(lambda lr: run(["b1", str(lr), "0"], f"{R}/b1_lr{lr:g}_s0.json", "dev"), [3e-4, 1e-3, 3e-3, 1e-2], "B1 sweep")
S["b1_sweep"] = {"dev": {f"{k:g}": v for k, v in sw.items()}, "best": lr_b1, "interior": interior}
S["base"] = run(["base"], f"{R}/base.json", "both")
S["b1"] = [run(["b1", str(lr_b1), str(s)], f"{R}/b1_best_s{s}.json", "test") for s in (0, 1, 2)]
px = run(["gen", "proxy", "100"], f"{R}/syn_proxy.json", "dev"); lam_hat = px["lam_mean"]; S["lam_hat"] = lam_hat
syn = {s: f"{R}/syn_priv_s{s}.json" for s in (0, 1, 2)}
run(["gen", "priv", "0"], syn[0], "dev")
lr_m, interior_m, swm = sweep(lambda lr: run(["m2a", str(lr), "0", syn[0], str(lam_hat)], f"{R}/m2a_lr{lr:g}_s0.json", "dev"), [1e-4, 3e-4, 1e-3], "M2a sweep")
S["m2a_sweep"] = {"dev": {f"{k:g}": v for k, v in swm.items()}, "best": lr_m, "interior": interior_m}
for s in (1, 2): run(["gen", "priv", str(s)], syn[s], "dev")
S["m2a"] = [run(["m2a", str(lr_m), str(s), syn[s], str(lam_hat)], f"{R}/m2a_best_s{s}.json", "test") for s in (0, 1, 2)]
cb = np.mean([r["test"]["correct"] for r in S["b1"]], 0); cm = np.mean([r["test"]["correct"] for r in S["m2a"]], 0)
d = cm - cb; rng = np.random.default_rng(0); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
S["primary"] = {"b1_test": float(cb.mean()), "m2a_test": float(cm.mean()), "delta": float(d.mean()),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
S["primary"]["decision"] = "CONTINUE (M2a beats B1)" if S["primary"]["ci95"][0] > 0 else "STOP signal-layer arms (prereg rule)"
for k in ("base",): S[k] = {sp: {"acc": S[k][sp]["acc"], "se": S[k][sp]["se"]} for sp in ("dev", "test")}
for k in ("b1", "m2a"): S[k] = [{x: r[x] for x in r if x not in ("test", "dev")} | {"test_acc": r["test"]["acc"], "test_se": r["test"]["se"]} for r in S[k]]
json.dump(S, open(f"{D}/summary.json", "w"), indent=1); print("SUMMARY", json.dumps(S["primary"]), flush=True)
