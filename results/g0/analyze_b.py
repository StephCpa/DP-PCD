import sys, json, glob, os, numpy as np
sys.path.insert(0, sys.argv[2]); from dpcd import efficiency as E, accounting as A
out = {}
for f in sorted(glob.glob(sys.argv[1] + "/profile_*.npz")):
    sup = os.path.basename(f)[8:-4]; z = np.load(f); row = {"meanlen": float(z["meanlen"]), "q0_rollout_acc": float(z["acc"]),
        "outside_mass_mean": float(z["outside"].mean()), "outside_mass_median": float(np.median(z["outside"])), "n_tok": int(len(z["lr"]))}
    for eps, al in ((1, 22.0), (4, 7.0), (8, 4.0)):
        d = z[f"d_mu2_a{al:g}"]; rep = E.report(E.Profile(d_mu2=d, lr=z["lr"], alpha=al), eps)
        row[eps] = {"alpha": al, "median_mu2": float(np.median(d)), "mean_mu2": float(d.mean()),
                    "hot": {b: float(np.mean(d > b)) for b in (0.01, 0.03, 0.1)},
                    "filter_gain": {b: rep["filter"][b]["gain_tokens_vs_fixed"] for b in (0.01, 0.03, 0.1)},
                    "fixed_tokens": {b: rep["fixed_b"][b]["tokens"] for b in (0.01, 0.03, 0.1)},
                    "m1_eff": {c: rep["gaussian"][c]["efficiency"] for c in (0.5, 1.0, 2.0)}}
    out[sup] = row
json.dump(out, open(sys.argv[1] + "/g0b_report.json", "w"), indent=1, default=float)
for s, r in out.items():
    print(s, "len", round(r["meanlen"]), "acc", round(r["q0_rollout_acc"], 3), "out mean/med", round(r["outside_mass_mean"], 4), round(r["outside_mass_median"], 5))
    for e in (1, 4, 8): x = r[e]; print("  eps", e, "med", f'{x["median_mu2"]:.3g}', "hot", [round(v, 3) for v in x["hot"].values()], "gain", [round(v, 2) for v in x["filter_gain"].values()])
