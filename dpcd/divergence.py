"""Divergences between categorical next-token distributions.

Every function works on the last axis, so a batch of states (..., V) is
handled in one call. Inputs are probabilities; rows must sum to 1.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

_TINY = 1e-300


def _log(p):
    with np.errstate(divide="ignore"):
        return np.log(p)


def renyi(p: np.ndarray, q: np.ndarray, alpha: float) -> np.ndarray:
    """D_alpha(p || q) for alpha > 1 (alpha=1 gives KL)."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    lp, lq = _log(p), _log(q)
    if alpha == 1.0:
        terms = np.where(p > 0, p * (lp - lq), 0.0)
        return terms.sum(-1)
    with np.errstate(invalid="ignore"):
        x = np.where(p > 0, alpha * lp + (1 - alpha) * lq, -np.inf)
    return logsumexp(x, axis=-1) / (alpha - 1)


def chi2(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Pearson chi^2(p || q) = sum p^2/q - 1."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p > 0, p * p / np.maximum(q, _TINY), 0.0).sum(-1) - 1.0


def mixture(q: np.ndarray, q0: np.ndarray, lam) -> np.ndarray:
    lam = np.asarray(lam, float)[..., None]
    return lam * q + (1 - lam) * q0


def two_sided_renyi(p, q, alpha):
    return np.maximum(renyi(p, q, alpha), renyi(q, p, alpha))


def lambda_for_bound(q: np.ndarray, q0: np.ndarray, bound, alpha: float, two_sided: bool = True,
                     iters: int = 40) -> np.ndarray:
    """Largest lambda in [0, 1] with D_alpha(mix || q0) <= bound (vectorised bisection).

    Both directions of the Renyi divergence of the mixture are nondecreasing in
    lambda (the inner sums are convex in lambda and minimal at 0), so bisection
    is exact up to ``iters`` halvings. ``bound`` may be an array (per state).
    """
    q = np.atleast_2d(q)
    q0 = np.atleast_2d(q0)
    bound = np.broadcast_to(np.asarray(bound, float), q.shape[:-1])
    f = (lambda lam: two_sided_renyi(mixture(q, q0, lam), q0, alpha)) if two_sided else \
        (lambda lam: renyi(mixture(q, q0, lam), q0, alpha))
    full = f(np.ones(q.shape[:-1]))
    lo = np.zeros(q.shape[:-1])
    hi = np.ones(q.shape[:-1])
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok = f(mid) <= bound
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    lam = np.where(full <= bound, 1.0, lo)
    return np.where(bound <= 0, 0.0, lam)


def log_ratio(q: np.ndarray, q0: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """log q(a) - log q0(a) at the chosen tokens a (last axis indexed by tokens)."""
    idx = np.asarray(tokens)[..., None]
    return (_log(np.take_along_axis(q, idx, -1)) - _log(np.take_along_axis(q0, idx, -1)))[..., 0]
