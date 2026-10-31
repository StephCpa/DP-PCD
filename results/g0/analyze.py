import sys, json, numpy as np
sys.path.insert(0, sys.argv[2])
from dpcd import efficiency as E, accounting as A
z = np.load(sys.argv[1] + "/profile.npz"); out = {}
for eps in (1, 4, 8):
    mu, order = A.mu_rdp(eps, 1e-6)
    keys = sorted([k for k in z if k.startswith("d_mu2_a")], key=lambda k: abs(float(k[7:]) - order))
    d = z[keys[0]]; al = float(keys[0][7:])
    r = E.report(E.Profile(d_mu2=d, lr=z["lr"], alpha=al), eps)
    r.update(alpha_used=al, median_mu2=float(np.median(d)), mean_mu2=float(d.mean()),
             hot_share={b: float(np.mean(d > b)) for b in (0.01, 0.03, 0.1)},
             top10_concentration=E.concentration(d, 0.1))
    out[eps] = r
print(json.dumps(out, indent=1, default=float)); json.dump(out, open(sys.argv[1] + "/g0_report.json", "w"), indent=1, default=float)
