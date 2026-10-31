"""Exact privacy-loss witness for token-sampling mechanisms (plan Phase 3, M3).

For a toy language model with vocabulary V and horizon T we enumerate all V^T
output sequences and compute their exact probabilities under the dataset with
r (P) and its add/remove-r neighbour (P0: the teacher always equals q0). From
these we get the exact Renyi divergence of the whole run in both directions and
the exact hockey-stick delta(eps), and compare them with what the ledger claims.

A toy LM is a pair of callables q0(prefixes) and qr(prefixes), mapping an int
array of prefixes (N, t) to next-token distributions (N, V).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import logsumexp

from .divergence import lambda_for_bound, mixture, renyi, two_sided_renyi


@dataclass
class ToyLM:
    V: int
    T: int
    q0: Callable[[np.ndarray], np.ndarray]
    qr: Callable[[np.ndarray], np.ndarray]


def hashed_toy_lm(V: int, T: int, seed: int, strength: float = 4.0, frac_private: float = 0.5,
                  temp0: float = 1.0) -> ToyLM:
    """Random toy LM whose distributions depend on the full prefix via hashing.

    On a random ``frac_private`` share of states the privileged teacher moves
    sharply away from q0 (logit shift of size ``strength`` toward a token q0
    finds unlikely); on the rest q and q0 agree, so realized costs vary a lot.
    """
    rng = np.random.default_rng(seed)
    n_states = 4096
    Z0 = rng.normal(0, temp0, (n_states, V))
    target = np.argmin(Z0, axis=1)  # the token q0 likes least
    active = rng.random(n_states) < frac_private
    shift = np.zeros((n_states, V))
    shift[np.arange(n_states), target] = strength * active
    shift += 0.3 * rng.normal(size=(n_states, V)) * active[:, None]
    mult = (np.arange(1, T + 2) * 2654435761) % (2**31 - 1)

    def key(pref):
        if pref.shape[1] == 0:
            return np.zeros(pref.shape[0], dtype=np.int64)
        h = (pref.astype(np.int64) + 1) * mult[: pref.shape[1]]
        return (h.sum(1) + 7 * pref.shape[1]) % n_states

    def sm(z):
        z = z - z.max(-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(-1, keepdims=True)

    return ToyLM(V, T, lambda p: sm(Z0[key(p)]), lambda p: sm(Z0[key(p)] + shift[key(p)]))


def enumerate_mechanism(lm: ToyLM, policy: str, alpha: float, budget: float = 0.0, b: float = 0.0,
                        b_max: float = np.inf, n_private: int | None = None):
    """Exact log-probabilities of all V^T outputs under P (with r) and P0 (without).

    policy:
      'filter'           M3: lam_t so the two-sided D_alpha <= min(b_max, remaining); charge realized.
      'fixed'            M2: lam_t so the two-sided D_alpha <= b for the first n_private tokens.
      'one_sided_filter' negative control: like 'filter' but bounds/charges D_alpha(p||q0) only.
      'overspend_filter' negative control: last token charged min(cost, remaining) after using b_max.
    Returns (logP, logP0, max_ledger) where max_ledger is the largest charged total on any path.
    """
    V, T = lm.V, lm.T
    pref = np.zeros((1, 0), dtype=np.int64)
    logP = np.zeros(1)
    logP0 = np.zeros(1)
    spent = np.zeros(1)
    for t in range(T):
        q0 = lm.q0(pref)
        qr = lm.qr(pref)
        if policy == "filter":
            bound = np.minimum(b_max, budget - spent)
            lam = lambda_for_bound(qr, q0, np.maximum(bound, 0), alpha, two_sided=True)
            p = mixture(qr, q0, lam)
            cost = np.where(lam > 0, two_sided_renyi(p, q0, alpha), 0.0)
        elif policy == "one_sided_filter":
            bound = np.minimum(b_max, budget - spent)
            lam = lambda_for_bound(qr, q0, np.maximum(bound, 0), alpha, two_sided=False)
            p = mixture(qr, q0, lam)
            cost = np.where(lam > 0, renyi(p, q0, alpha), 0.0)
        elif policy == "overspend_filter":
            rem = budget - spent
            bound = np.where(rem > 0, b_max, 0.0)  # bug: ignores how little budget is left
            lam = lambda_for_bound(qr, q0, bound, alpha, two_sided=True)
            p = mixture(qr, q0, lam)
            cost = np.minimum(np.where(lam > 0, two_sided_renyi(p, q0, alpha), 0.0), np.maximum(rem, 0))
        elif policy == "wrong_order_filter":
            # negative control: bounds and charges order-2 divergence while the
            # claim is made at order ``alpha``
            bound = np.minimum(b_max, budget - spent)
            lam = lambda_for_bound(qr, q0, np.maximum(bound, 0), 2.0, two_sided=True)
            p = mixture(qr, q0, lam)
            cost = np.where(lam > 0, two_sided_renyi(p, q0, 2.0), 0.0)
        elif policy == "fixed":
            use = n_private is None or t < n_private
            lam = lambda_for_bound(qr, q0, b if use else 0.0, alpha, two_sided=True)
            p = mixture(qr, q0, lam)
            cost = np.full(len(p), b if use else 0.0)
        else:
            raise ValueError(policy)
        with np.errstate(divide="ignore"):
            lp = np.log(p)
            lq = np.log(q0)
        logP = (logP[:, None] + lp).reshape(-1)
        logP0 = (logP0[:, None] + lq).reshape(-1)
        spent = np.repeat(spent + cost, V)
        pref = np.concatenate([np.repeat(pref, V, axis=0), np.tile(np.arange(V), len(pref))[:, None]], 1)
    return logP, logP0, float(spent.max())


def exact_renyi(logP, logP0, alpha):
    """D_alpha(P||P0) and D_alpha(P0||P) of the whole run."""
    fwd = logsumexp(alpha * logP + (1 - alpha) * logP0) / (alpha - 1)
    bwd = logsumexp(alpha * logP0 + (1 - alpha) * logP) / (alpha - 1)
    return float(fwd), float(bwd)


def exact_delta(logP, logP0, eps):
    """Hockey-stick delta(eps) in both directions."""
    P, P0 = np.exp(logP), np.exp(logP0)
    return float(np.maximum(P - np.exp(eps) * P0, 0).sum()), float(np.maximum(P0 - np.exp(eps) * P, 0).sum())
