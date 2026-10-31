"""Analytic categorical simulator (feasibility note, direction 4; plan ablations).

A world has S public states (contexts) and V tokens. Each state has a public
prior q0_s. The privileged teacher of record i at state s is

    q_{i,s} = softmax(log q0_s + Delta_s + xi_{i,s}),   xi ~ N(0, tau^2 I),

where Delta is the *shared* privileged component (what DP can transfer) and xi
the record-specific one. 'state' worlds have a hot token on a share of states;
'global' worlds share one shift across all states (the fully shared case of P2).

Each record visits L states. A mechanism turns the records into releases; the
student pi_s = softmax(log q0_s + theta_{g(s)}) is then fitted to the releases by
penalised maximum likelihood (post-processing). Utility is the share of
headroom recovered, H = 1 - KL(qbar||pi)/KL(qbar||q0), averaged over states,
where qbar_s is the population teacher.

Mechanisms (privacy unit: one record's r, add/remove-r; eps at delta=1e-6):
  np_full      non-private, full teacher distribution (cross-entropy)
  np_sampled   non-private, exact log-ratio at a token sampled from q0
  np_imitate   non-private, one token sampled from the teacher (ActionOPD-like)
  m1_sampled   clipped Gaussian log-ratio at the sampled token (M1)
  m1_full      L2-clipped Gaussian release of the whole log-ratio vector
  m2_q0        DP-Fusion mixture toward q0, fixed per-token bound (M2)
  m2_unif      mixture toward uniform with worst-case (public) lambda (RR-like)
  m3_filter    mixture toward q0 with the realized-divergence filter (M3)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize
from scipy.special import logsumexp

from . import accounting as A
from .divergence import lambda_for_bound, mixture, renyi, two_sided_renyi
from .mechanisms import sample_rows

DELTA = 1e-6


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


@dataclass
class World:
    kind: str
    S: int
    V: int
    logq0: np.ndarray  # (S, V)
    Delta: np.ndarray  # (S, V)
    tau: float
    qbar: np.ndarray  # (S, V) population teacher

    @property
    def groups(self):
        return np.zeros(self.S, int) if self.kind == "global" else np.arange(self.S)

    @property
    def n_groups(self):
        return 1 if self.kind == "global" else self.S


def make_world(kind="state", S=64, V=32, tau=0.5, f_hot=0.5, shift=(1.5, 3.0), temp=1.5, seed=0) -> World:
    rng = np.random.default_rng(seed)
    z0 = rng.normal(0, temp, (S, V))
    logq0 = z0 - logsumexp(z0, -1, keepdims=True)
    Delta = np.zeros((S, V))
    if kind == "global":
        k = rng.integers(V)
        Delta[:, k] = rng.uniform(*shift)
    else:
        hot = rng.random(S) < f_hot
        ks = rng.integers(0, V, S)
        Delta[np.arange(S), ks] = rng.uniform(*shift, S) * hot
    xi = rng.normal(0, tau, (2000, S, V))
    qbar = softmax(logq0[None] + Delta[None] + xi).mean(0)
    return World(kind, S, V, logq0, Delta, tau, qbar)


def sample_records(w: World, n: int, L: int, rng):
    states = rng.integers(0, w.S, (n, L))
    q = softmax(w.logq0[states] + w.Delta[states] + rng.normal(0, w.tau, (n, L, w.V)))
    return states, q


# ---------------------------------------------------------------------------
# Student fitting (sufficient statistics -> penalised ML / least squares)
# ---------------------------------------------------------------------------
def _pi(w: World, theta):
    return softmax(w.logq0 + theta[w.groups])


def _grad_to_theta(w, g_state):
    out = np.zeros((w.n_groups, w.V))
    np.add.at(out, w.groups, g_state)
    return out


def fit(w: World, stats: dict, kappa: float, maxiter: int = 400):
    """Fit theta given sufficient statistics of one of four loss types."""
    kind = stats["loss"]

    def f(th_flat):
        th = th_flat.reshape(w.n_groups, w.V)
        pi = _pi(w, th)
        logpi = np.log(pi)
        if kind == "ce":  # sum_r CE(q_r, pi_s): stats Q[s,a] = sum of q_r, N[s]
            val = -(stats["Q"] * logpi).sum()
            g = stats["N"][:, None] * pi - stats["Q"]
        elif kind == "ls_token":  # sum_r (log pi(a)-log q0(a) - v)^2/2 ; stats C[s,a], SV[s,a], SVV
            u = logpi - w.logq0
            C, SV = stats["C"], stats["SV"]
            val = 0.5 * (C * u * u - 2 * u * SV).sum() + 0.5 * stats["SVV"]
            r = C * u - SV  # d val / d u(s,a)
            g = r - pi * r.sum(-1, keepdims=True)
        elif kind == "ls_vector":  # sum_r ||u_s - v_r||^2/2 ; stats N[s], SVvec[s,:], SVV
            u = logpi - w.logq0
            N, SV = stats["N"], stats["SVvec"]
            val = 0.5 * (N[:, None] * u * u - 2 * u * SV).sum() + 0.5 * stats["SVV"]
            r = N[:, None] * u - SV
            g = r - pi * r.sum(-1, keepdims=True)
        elif kind == "mix":  # -sum_r log(lam_r pi_s(y) + (1-lam_r) a_s(y)) ; per-release arrays
            s, y, lam, anc = stats["s"], stats["y"], stats["lam"], stats["anchor_p"]
            piy = pi[s, y]
            p = lam * piy + (1 - lam) * anc
            val = -np.log(np.maximum(p, 1e-300)).sum()
            wgt = -lam * piy / np.maximum(p, 1e-300)  # d val / d logit_y via pi(y)(e_y - pi)
            G = np.zeros((w.S, w.V))
            np.add.at(G, (s, y), wgt)
            Ws = np.bincount(s, weights=wgt, minlength=w.S)
            g = G - Ws[:, None] * pi
        else:
            raise ValueError(kind)
        val += 0.5 * kappa * (th * th).sum()
        gt = _grad_to_theta(w, g) + kappa * th
        return val, gt.ravel()

    x0 = np.zeros(w.n_groups * w.V)
    r = optimize.minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    return r.x.reshape(w.n_groups, w.V)


def headroom(w: World, theta):
    pi = _pi(w, theta)
    kl = lambda p, q: (p * (np.log(p) - np.log(q))).sum(-1)
    return float(1 - kl(w.qbar, pi).mean() / kl(w.qbar, np.exp(w.logq0)).mean())


# ---------------------------------------------------------------------------
# Mechanisms -> sufficient statistics
# ---------------------------------------------------------------------------
def releases(w: World, mech: str, n: int, L: int, eps: float, rng, k: int | None = None,
             c: float = 1.0, g: float = 1.0):
    """Simulate one run of mechanism ``mech``; returns (stats, info)."""
    states, q = sample_records(w, n, L, rng)
    k = k or L  # number of states per record that are released (public choice: the first k)
    states, q = states[:, :k], q[:, :k]
    s = states.ravel()
    qf = q.reshape(-1, w.V)
    q0f = np.exp(w.logq0[s])
    info = {}
    if eps == np.inf and mech.startswith(("m1", "m2", "m3")):
        raise ValueError("use np_* for eps=inf")
    if mech == "np_full":
        Q = np.zeros((w.S, w.V))
        np.add.at(Q, s, qf)
        return {"loss": "ce", "Q": Q, "N": np.bincount(s, minlength=w.S).astype(float)}, info
    if mech in ("np_sampled", "m1_sampled"):
        a = sample_rows(q0f, rng)
        lr = np.log(qf[np.arange(len(a)), a]) - np.log(q0f[np.arange(len(a)), a])
        if mech == "m1_sampled":
            mu = A.gdp_mu(eps, DELTA)
            sigma = A.gaussian_sigma(k, c, mu)
            v = np.clip(lr, -c, c) + rng.normal(0, sigma, len(lr))
            info.update(sigma=sigma, clip_frac=float(np.mean(np.abs(lr) > c)))
        else:
            v = lr
        C = np.zeros((w.S, w.V))
        SV = np.zeros((w.S, w.V))
        np.add.at(C, (s, a), 1.0)
        np.add.at(SV, (s, a), v)
        return {"loss": "ls_token", "C": C, "SV": SV, "SVV": float((v * v).sum())}, info
    if mech == "m1_full":
        lr = np.log(qf) - np.log(q0f)
        nrm = np.linalg.norm(lr, axis=1, keepdims=True)
        lr_c = lr * np.minimum(1.0, c / np.maximum(nrm, 1e-12))
        mu = A.gdp_mu(eps, DELTA)
        sigma = A.gaussian_sigma(k, c, mu)
        v = lr_c + rng.normal(0, sigma, lr_c.shape)
        SV = np.zeros((w.S, w.V))
        np.add.at(SV, s, v)
        info.update(sigma=sigma, clip_frac=float(np.mean(nrm > c)))
        return {"loss": "ls_vector", "N": np.bincount(s, minlength=w.S).astype(float), "SVvec": SV,
                "SVV": float((v * v).sum())}, info
    if mech == "np_imitate":
        y = sample_rows(qf, rng)
        return {"loss": "mix", "s": s, "y": y, "lam": np.ones(len(y)), "anchor_p": np.zeros(len(y))}, info
    if mech in ("m2_q0", "m3_filter", "m2_unif"):
        mu_r, alpha = A.mu_rdp(eps, DELTA)
        B = A.rdp_budget_at_order(eps, DELTA, alpha)
        if mech == "m2_q0":
            lam = lambda_for_bound(qf, q0f, B / k, alpha, two_sided=True)
            p = mixture(qf, q0f, lam)
            anchor = q0f
        elif mech == "m2_unif":
            u = np.full(w.V, 1.0 / w.V)
            lam_s = _rr_lambda(np.exp(w.logq0), u, B / k, alpha)  # public, per state
            lam = lam_s[s]
            p = lam[:, None] * qf + (1 - lam[:, None]) * u
            anchor = np.broadcast_to(u, qf.shape)
            info.update(mean_lambda=float(lam.mean()))
        else:  # m3: sequential filter per record over its k tokens, cap b_max = g * B / k
            lam = np.zeros(n * k)
            spent = np.zeros(n)
            for t in range(k):
                idx = np.arange(n) * k + t
                bound = np.minimum(g * B / k, np.maximum(B - spent, 0))
                lt = lambda_for_bound(qf[idx], q0f[idx], bound, alpha, two_sided=True)
                pt = mixture(qf[idx], q0f[idx], lt)
                cost = np.where(lt > 0, two_sided_renyi(pt, q0f[idx], alpha), 0.0)
                assert np.all(cost <= bound + 1e-9)
                spent += cost
                lam[idx] = lt
            p = mixture(qf, q0f, lam)
            anchor = q0f
            info.update(spent_mean=float(spent.mean() / B))
        y = sample_rows(p, rng)
        info.update(mean_lambda=float(lam.mean()), alpha=alpha)
        st = {"loss": "mix", "s": s, "y": y, "anchor_p": anchor[np.arange(len(y)), y]}
        if mech == "m2_unif":
            st["lam"] = lam  # public: depends only on q0
        else:
            # lam depends on the private teacher and is NOT released; the student
            # may only use a public lam_hat (see publicize / tune_and_eval)
            st["lam_private_mean"] = float(lam.mean())
        return st, info
    raise ValueError(mech)


def _rr_lambda(q0, u, b, alpha, iters=50):
    """Public lambda per state: worst case over point-mass teachers (convexity => vertices)."""
    S, V = q0.shape
    lo, hi = np.zeros(S), np.ones(S)
    eye = np.eye(V)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        worst = np.zeros(S)
        for a in range(V):
            p = mid[:, None] * eye[a] + (1 - mid[:, None]) * u
            p0 = mid[:, None] * q0 + (1 - mid[:, None]) * u
            worst = np.maximum(worst, two_sided_renyi(p, p0, alpha))
        ok = worst <= b
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    return lo


def publicize(stats: dict, lam_hat: float | None) -> dict:
    """Replace the private per-token lambda by a public constant lam_hat."""
    if stats["loss"] == "mix" and "lam" not in stats:
        stats = dict(stats)
        stats["lam"] = np.full(len(stats["y"]), lam_hat)
    return stats


KAPPAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)
GRIDS = {"m1_sampled": ("c", (0.1, 0.25, 0.5, 1.0, 2.0)), "m1_full": ("c", (0.5, 1.0, 2.0, 4.0)),
         "m3_filter": ("g", (1.0, 2.0, 4.0, 8.0))}


def tune_and_eval(world_kw: dict, mech: str, n: int, L: int, eps: float, seeds=(1, 2, 3), k=None,
                  proxy_seed: int = 10_000):
    """Tune (kappa, mechanism knob) on a proxy world draw, then evaluate on ``seeds``."""
    name, grid = GRIDS.get(mech, (None, (None,)))
    wp = make_world(seed=proxy_seed, **world_kw)
    best = (-np.inf, None, None, None)
    for val in grid:
        kw = {name: val} if name else {}
        st, _ = releases(wp, mech, n, L, eps, np.random.default_rng(proxy_seed), k=k, **kw)
        lam_hat = st.get("lam_private_mean")  # measured on the public proxy world
        st = publicize(st, lam_hat)
        for kap in KAPPAS:
            h = headroom(wp, fit(wp, st, kap))
            if h > best[0]:
                best = (h, kap, val, lam_hat)
    _, kap, val, lam_hat = best
    kw = {name: val} if name else {}
    hs, infos = [], []
    for sd in seeds:
        w = make_world(seed=sd, **world_kw)
        st, info = releases(w, mech, n, L, eps, np.random.default_rng(sd + 777), k=k, **kw)
        hs.append(headroom(w, fit(w, publicize(st, lam_hat), kap)))
        infos.append(info)
    return {"mech": mech, "n": n, "L": L, "k": k or L, "eps": eps, "kappa": kap, "knob": val, "lam_hat": lam_hat,
            "H_mean": float(np.mean(hs)), "H_sd": float(np.std(hs, ddof=1)) if len(hs) > 1 else 0.0,
            "H": hs, "info": infos[0]}
