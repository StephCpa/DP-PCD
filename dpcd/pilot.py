"""Miniature CPU pilot of the plan's ten arms (plan Phases 1-3 at toy scale).

This is NOT the plan's LLM experiment. It runs the same arms, accounting,
schedule rules and audits end to end on a synthetic privileged-context task
small enough for a CPU, to validate the pipeline and to get a first, toy-scale
read on the noise-geometry question behind H3.

Task. Public question x in R^P (p_sig informative features, p_pad irrelevant
ones, plus a bias). The answer is T tokens over K symbols, produced by a shared
autoregressive rule y_t = argmax(gamma * Wstar_t x_sig + Estar[y_{t-1}]); the
private worked solution r_i is that answer with each token replaced by a random
symbol with probability rho (record-specific content).

Student. Log-linear autoregressive policy pi(y_t | x, y_<t) = softmax(W_t x + E[y_{t-1}]),
initialised at a weak 'base' theta0. q0 = the frozen base; the privileged
teacher is the same frozen base with r in context, modelled as
q(.|x, r, h_t) = softmax(logits0 + beta * e_{r_t}).

Arms: R0 base, R1 SFT, R2 OPSD, R3 privileged-sample imitation, B1 DP-SFT,
B2 DP-OPSD, M1 Gaussian log-ratio, M2a DP-Fusion synthetic SFT, M2b DP-Fusion
on-policy chunks, M3 filtered chunks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from . import accounting as A
from .divergence import lambda_for_bound, mixture, two_sided_renyi
from .mechanisms import sample_rows

DELTA = 1e-6


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskCfg:
    K: int = 10
    T: int = 4
    p_sig: int = 16
    p_pad: int = 0
    rho: float = 0.25
    gamma: float = 4.0
    base_scale: float = 0.3
    beta: float = 3.0

    @property
    def P(self):
        return self.p_sig + self.p_pad + 1

    @property
    def d(self):
        return self.T * self.K * self.P + (self.K + 1) * self.K


@dataclass
class Task:
    cfg: TaskCfg
    Wstar: np.ndarray  # (T, K, p_sig)
    Estar: np.ndarray  # (K+1, K)
    theta0: dict


def make_task(cfg: TaskCfg, seed: int) -> Task:
    rng = np.random.default_rng(seed)
    Wstar = rng.normal(0, 1 / math.sqrt(cfg.p_sig), (cfg.T, cfg.K, cfg.p_sig)) * cfg.gamma
    Estar = rng.normal(0, 1.0, (cfg.K + 1, cfg.K))
    W0 = np.zeros((cfg.T, cfg.K, cfg.P))
    W0[:, :, : cfg.p_sig] = cfg.base_scale * Wstar + rng.normal(0, 0.3 / math.sqrt(cfg.p_sig), Wstar.shape)
    E0 = cfg.base_scale * Estar
    return Task(cfg, Wstar, Estar, {"W": W0, "E": E0})


def sample_x(cfg: TaskCfg, n: int, rng) -> np.ndarray:
    x = rng.normal(size=(n, cfg.P))
    x[:, -1] = 1.0
    return x


def rule_answer(task: Task, x) -> np.ndarray:
    c = task.cfg
    y = np.zeros((len(x), c.T), int)
    prev = np.full(len(x), c.K)
    for t in range(c.T):
        z = x[:, : c.p_sig] @ task.Wstar[t].T + task.Estar[prev]
        y[:, t] = z.argmax(1)
        prev = y[:, t]
    return y


def make_records(task: Task, n: int, rng):
    x = sample_x(task.cfg, n, rng)
    y = rule_answer(task, x)
    noise = rng.random(y.shape) < task.cfg.rho
    r = np.where(noise, rng.integers(0, task.cfg.K, y.shape), y)
    return x, r


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def logits_at(theta, x, prev, t):
    return x @ theta["W"][t].T + theta["E"][prev]


def prev_of(tokens, K):
    """prev token per position given tokens (B, T): [start, y_0, ..., y_{T-2}]."""
    return np.concatenate([np.full((len(tokens), 1), K), tokens[:, :-1]], 1)


def seq_logits(theta, x, tokens, K):
    """Logits at every position under teacher forcing on ``tokens``: (B, T, K)."""
    prev = prev_of(tokens, K)
    T = tokens.shape[1]
    return np.stack([logits_at(theta, x, prev[:, t], t) for t in range(T)], 1)


def rollout(theta, x, K, T, rng, greedy=False):
    B = len(x)
    toks = np.zeros((B, T), int)
    prev = np.full(B, K)
    for t in range(T):
        z = logits_at(theta, x, prev, t)
        toks[:, t] = z.argmax(1) if greedy else sample_rows(softmax(z), rng)
        prev = toks[:, t]
    return toks


def teacher_logits(theta0, x, tokens, r, beta, K):
    """Base logits at each state of ``tokens`` and the privileged teacher logits."""
    z0 = seq_logits(theta0, x, tokens, K)
    zq = z0.copy()
    B, T = r.shape
    zq[np.arange(B)[:, None], np.arange(T)[None, :], r] += beta
    return z0, zq


def grads(x, tokens, G, K, rec=None, n_rec=None, P=None):
    """Parameter gradient from logit gradients G (B, T, K) at states of ``tokens``.

    If ``rec`` (record index per row) is given, also returns per-record squared
    norms and a function computing the per-record clipped sum.
    """
    prev = prev_of(tokens, K)
    T = tokens.shape[1]
    dW = np.einsum("btk,bp->tkp", G, x)
    dE = np.zeros((K + 1, K))
    for t in range(T):
        np.add.at(dE, prev[:, t], G[:, t])
    return {"W": dW, "E": dE}


def per_record_clipped_sum(x, tokens, G, K, rec, n_rec, C):
    """Sum over records of clip_C(record gradient); rows of the same record share x."""
    prev = prev_of(tokens, K)
    B, T, _ = G.shape
    # aggregate rows to records
    Gr = np.zeros((n_rec, T, K))
    np.add.at(Gr, rec, G)
    xr = np.zeros((n_rec, x.shape[1]))
    xr[rec] = x
    Er = np.zeros((n_rec, K + 1, K))
    for t in range(T):
        np.add.at(Er, (rec, prev[:, t]), G[:, t])
    nrm2 = (Gr ** 2).sum((1, 2)) * (xr ** 2).sum(1) + (Er ** 2).sum((1, 2))
    scale = np.minimum(1.0, C / np.maximum(np.sqrt(nrm2), 1e-12))
    dW = np.einsum("rtk,rp->tkp", Gr * scale[:, None, None], xr)
    dE = (Er * scale[:, None, None]).sum(0)
    return {"W": dW, "E": dE}, float(np.mean(scale < 1))


class Adam:
    def __init__(self, theta, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in theta.items()}
        self.v = {k: np.zeros_like(v) for k, v in theta.items()}
        self.t = 0

    def step(self, theta, g):
        self.t += 1
        for k in theta:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            theta[k] = theta[k] - self.lr * mh / (np.sqrt(vh) + self.eps)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
@dataclass
class RunCfg:
    arm: str
    eps: float = 4.0
    lr: float = 0.03
    epochs: int = 2
    batch: int = 64
    rollouts: int = 4
    clip_C: float = 1.0  # DP-SGD per-record clip norm
    clip_c: float = 1.0  # M1 log-ratio clip
    n_syn: int = 2  # M2a synthetic sequences per record
    chunk: int = 2  # M2b/M3 chunk length
    g: float = 2.0  # M3 per-token cap multiplier over the fixed-b bound
    lam_hat: float | None = None  # public lambda for M2/M3 student loss
    seed: int = 0


def privacy_params(rc: RunCfg, n: int, T: int):
    """Everything fixed before training: noise, budgets, releases per record."""
    steps = rc.epochs * math.ceil(n / rc.batch)
    out = {"steps": steps}
    if rc.arm in ("B1", "B2"):
        q = rc.batch / n
        out["sample_rate"] = q
        out["z"] = A.dpsgd_noise_multiplier(rc.eps, DELTA, q, steps, replace=(rc.arm == "B2"))
    if rc.arm == "M1":
        R = rc.epochs * rc.rollouts * T
        out["R"] = R
        out["sigma"] = A.gaussian_sigma(R, rc.clip_c, A.gdp_mu(rc.eps, DELTA))  # add/remove-r: sens c
    if rc.arm in ("M2a", "M2b", "M3"):
        mu_r, alpha = A.mu_rdp(rc.eps, DELTA)
        B = A.rdp_budget_at_order(rc.eps, DELTA, alpha)
        n_tok = rc.n_syn * T if rc.arm == "M2a" else rc.epochs * rc.rollouts * rc.chunk
        out.update(alpha=alpha, B=B, n_tok=n_tok, b=B / n_tok)
    return out


def evaluate(task: Task, theta, x_eval, y_eval):
    c = task.cfg
    pred = rollout(theta, x_eval, c.K, c.T, None, greedy=True)
    return {"exact": float((pred == y_eval).all(1).mean()), "token": float((pred == y_eval).mean())}


def canary_scores(task: Task, theta, x_can, r_can):
    c = task.cfg
    lp = log_softmax(seq_logits(theta, x_can, r_can, c.K))
    lp0 = log_softmax(seq_logits(task.theta0, x_can, r_can, c.K))
    idx = (np.arange(len(r_can))[:, None], np.arange(c.T)[None, :], r_can)
    return (lp[idx] - lp0[idx]).sum(1)


def train(task: Task, rc: RunCfg, x, r, eval_sets=None, proxy=False):
    """Train one arm. ``proxy=True`` marks a public proxy run (the student may use the
    realized lambda's running mean as lam_hat, since nothing there is private)."""
    c = task.cfg
    K, T = c.K, c.T
    n = len(x)
    rng = np.random.default_rng(rc.seed)
    theta = {k: v.copy() for k, v in task.theta0.items()}
    th0 = task.theta0
    pp = privacy_params(rc, n, T)
    opt = Adam(theta, rc.lr)
    log = {"clip_frac": [], "lambda": [], "pp": pp}
    if rc.arm == "R0":
        return theta, log

    ledger = np.zeros(n)
    lam_sum, lam_cnt = 0.0, 0
    syn = None
    if rc.arm == "M2a":  # release synthetic solutions once, before training
        syn_x, syn_y, lams = [], [], []
        for _ in range(rc.n_syn):
            toks = np.zeros((n, T), int)
            prev = np.full(n, K)
            for t in range(T):
                z0 = logits_at(th0, x, prev, t)
                zq = z0.copy()
                zq[np.arange(n), r[:, t]] += c.beta
                lam = lambda_for_bound(softmax(zq), softmax(z0), pp["b"], pp["alpha"])
                toks[:, t] = sample_rows(mixture(softmax(zq), softmax(z0), lam), rng)
                prev = toks[:, t]
                lams.append(lam)
            syn_x.append(x)
            syn_y.append(toks)
        syn = (np.concatenate(syn_x), np.concatenate(syn_y))
        lam_all = np.concatenate(lams)
        log["lambda"].append(float(lam_all.mean()))
        lam_hat = float(lam_all.mean()) if proxy else rc.lam_hat

    for step in range(pp["steps"]):
        # ---- batch selection ------------------------------------------------
        if rc.arm in ("B1", "B2"):
            idx = np.where(rng.random(n) < pp["sample_rate"])[0]  # Poisson
            if len(idx) == 0:
                continue
        else:  # deterministic epochs: releases per record fixed in advance
            e, pos = divmod(step * rc.batch, n)
            perm = np.random.default_rng(rc.seed * 1000 + e).permutation(n)
            idx = perm[pos: pos + rc.batch]
        xb, rb = x[idx], r[idx]
        nb = len(idx)

        if rc.arm in ("R1", "B1"):
            toks = rb
            G = softmax(seq_logits(theta, xb, toks, K))
            G[np.arange(nb)[:, None], np.arange(T)[None, :], toks] -= 1.0
            rows_x, rec = xb, np.arange(nb)
        elif rc.arm == "M2a":
            sel = np.concatenate([idx + j * n for j in range(rc.n_syn)])
            xs, ys = syn[0][sel], syn[1][sel]
            G = _mixture_ce_grad(theta, th0, xs, ys, K, lam_hat)
            g = grads(xs, ys, G, K)
            opt.step(theta, _scale(g, 1.0 / len(xs)))
            continue
        else:
            # on-policy rollouts at the student's states
            rows_x = np.repeat(xb, rc.rollouts, 0)
            rows_r = np.repeat(rb, rc.rollouts, 0)
            rec = np.repeat(np.arange(nb), rc.rollouts)
            toks = rollout(theta, rows_x, K, T, rng)
            z = seq_logits(theta, rows_x, toks, K)
            pi = softmax(z)
            logpi = log_softmax(z)
            z0, zq = teacher_logits(th0, rows_x, toks, rows_r, c.beta, K)
            if rc.arm in ("R2", "B2"):
                lq = log_softmax(zq)
                kl = (pi * (logpi - lq)).sum(-1, keepdims=True)
                G = pi * (logpi - lq - kl)
            elif rc.arm == "R3":
                y = sample_rows(softmax(zq).reshape(-1, K), rng).reshape(toks.shape)
                G = pi.copy()
                G[np.arange(len(toks))[:, None], np.arange(T)[None, :], y] -= 1.0
            elif rc.arm == "M1":
                l0 = log_softmax(z0)
                lq = log_softmax(zq)
                kl0 = (pi * (logpi - l0)).sum(-1, keepdims=True)
                G_pub = pi * (logpi - l0 - kl0)
                ii = (np.arange(len(toks))[:, None], np.arange(T)[None, :], toks)
                lr_ = lq[ii] - l0[ii]
                v = np.clip(lr_, -rc.clip_c, rc.clip_c) + rng.normal(0, pp["sigma"], lr_.shape)
                onehot = np.zeros_like(pi)
                onehot[ii] = 1.0
                G = G_pub - v[..., None] * (onehot - pi)
                log["clip_frac"].append(float(np.mean(np.abs(lr_) > rc.clip_c)))
            elif rc.arm in ("M2b", "M3"):
                G, ctoks, cx, lam_used = _chunk_step(theta, th0, rows_x, rows_r, toks, rec, idx, rc, pp,
                                                     ledger, rng, K, T, c.beta,
                                                     (lam_sum / max(lam_cnt, 1)) if proxy else rc.lam_hat)
                lam_sum += lam_used.sum()
                lam_cnt += lam_used.size
                g = grads(cx, ctoks, G, K)
                opt.step(theta, _scale(g, 1.0 / nb))
                continue
            else:
                raise ValueError(rc.arm)

        # ---- aggregate ---------------------------------------------------------
        if rc.arm in ("B1", "B2"):
            gsum, cf = per_record_clipped_sum(rows_x, toks, G, K, rec, nb, rc.clip_C)
            noise_sd = pp["z"] * rc.clip_C
            gsum = {k: v + rng.normal(0, noise_sd, v.shape) for k, v in gsum.items()}
            log["clip_frac"].append(cf)
            g = _scale(gsum, 1.0 / (pp["sample_rate"] * n))  # divide by expected batch size
        else:
            g = _scale(grads(rows_x, toks, G, K), 1.0 / nb)
        opt.step(theta, g)

    if rc.arm in ("M2b", "M3"):
        log["lambda"].append(lam_sum / max(lam_cnt, 1))
        log["ledger_max"] = float(ledger.max()) if rc.arm == "M3" else None
        log["ledger_mean_frac"] = float(ledger.mean() / pp["B"]) if rc.arm == "M3" else None
    return theta, log


def _scale(g, s):
    return {k: v * s for k, v in g.items()}


def _mixture_ce_grad(theta, th0, xs, ys, K, lam_hat):
    """Gradient of -log(lam_hat * pi(y) + (1 - lam_hat) * q0(y)) w.r.t. logits."""
    z = seq_logits(theta, xs, ys, K)
    pi = softmax(z)
    q0 = softmax(seq_logits(th0, xs, ys, K))
    T = ys.shape[1]
    ii = (np.arange(len(ys))[:, None], np.arange(T)[None, :], ys)
    piy, q0y = pi[ii], q0[ii]
    w = lam_hat * piy / (lam_hat * piy + (1 - lam_hat) * q0y)  # posterior that y came from pi
    G = pi * w[..., None]
    G[ii] -= w
    return G


def _chunk_step(theta, th0, rows_x, rows_r, toks, rec, idx, rc, pp, ledger, rng, K, T, beta, lam_hat):
    """M2b / M3: at a public random position of each student rollout, sample a teacher
    chunk from the q/q0 mixture and return the logit gradient of the mixture loss."""
    nrow = len(toks)
    t0 = rng.integers(0, T, nrow)  # public choice
    ctoks = toks.copy()
    mask = np.zeros((nrow, T), bool)
    lam_used = []
    prev = prev_of(toks, K)[np.arange(nrow), t0]
    for j in range(rc.chunk):
        t = t0 + j
        live = t < T
        if not live.any():
            break
        rows = np.where(live)[0]
        tt = t[rows]
        z0 = np.einsum("rkp,rp->rk", th0["W"][tt], rows_x[rows]) + th0["E"][prev[rows]]
        zq = z0.copy()
        zq[np.arange(len(rows)), rows_r[rows, tt]] += beta
        q, q0 = softmax(zq), softmax(z0)
        if rc.arm == "M2b":
            bound = np.full(len(rows), pp["b"])
        else:
            # rows of the same record are released in parallel: split the record's
            # remaining budget evenly across its live rows so the sum cannot overspend
            rid = idx[rec[rows]]
            share = np.bincount(rid, minlength=len(ledger))[rid]
            rem = np.maximum(pp["B"] - ledger[rid], 0.0) / share
            bound = np.minimum(rc.g * pp["b"], rem)
        lam = lambda_for_bound(q, q0, bound, pp["alpha"])
        p = mixture(q, q0, lam)
        if rc.arm == "M3":
            cost = np.where(lam > 0, two_sided_renyi(p, q0, pp["alpha"]), 0.0)
            assert np.all(cost <= bound + 1e-9)
            np.add.at(ledger, idx[rec[rows]], cost)
            assert ledger.max() <= pp["B"] + 1e-9, "filter overspend"
        y = sample_rows(p, rng)
        ctoks[rows, tt] = y
        mask[rows, tt] = True
        prev[rows] = y
        lam_used.append(lam)
    # loss only on chunk tokens, at the chunk's states (teacher-forced on ctoks)
    lh = 1.0 if lam_hat is None else lam_hat
    G = _mixture_ce_grad(theta, th0, rows_x, ctoks, K, lh) * mask[..., None]
    return G, ctoks, rows_x, np.concatenate(lam_used) if lam_used else np.zeros(0)
