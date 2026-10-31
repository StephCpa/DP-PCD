"""Signal-layer release mechanisms (arms M1, M2, M3), model-agnostic.

Each function takes the privileged teacher distribution q = q(.|x, r, h) and the
public prior q0 = q(.|x, h) at a batch of states (shape (B, V)) and returns what
is released. The student update is post-processing of these outputs.

Privacy unit: one record's privileged context r, add/remove-r adjacency (the
neighbour's teacher equals q0 at every state).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .divergence import lambda_for_bound, log_ratio, mixture, renyi, two_sided_renyi


# ---------------------------------------------------------------------------
# M1: clipped Gaussian log-ratio at publicly chosen tokens
# ---------------------------------------------------------------------------
def m1_release(q, q0, tokens, c: float, sigma: float, rng: np.random.Generator):
    """v = clip(log q(a) - log q0(a), -c, c) + N(0, sigma^2).

    Under add/remove-r the neighbour releases clip(0)+noise, so the per-release
    sensitivity is c; under replace-r it is 2c. Calibrate sigma accordingly with
    ``accounting.gaussian_sigma``.
    """
    lr = log_ratio(q, q0, tokens)
    clipped = np.clip(lr, -c, c)
    return clipped + rng.normal(0.0, sigma, size=clipped.shape), lr


# ---------------------------------------------------------------------------
# M2: DP-Fusion style mixing with a fixed per-token Renyi bound
# ---------------------------------------------------------------------------
def dpfusion_step(q, q0, bound: float, alpha: float, rng: np.random.Generator, two_sided: bool = True):
    """Sample one token per state from mix = lam*q + (1-lam)*q0, D_alpha <= bound.

    Returns (tokens, lam, realized_cost). The ledger charges ``bound`` per token
    regardless of the realized cost (fixed-b accounting).
    """
    lam = lambda_for_bound(q, q0, bound, alpha, two_sided=two_sided)
    p = mixture(q, q0, lam)
    tok = sample_rows(p, rng)
    cost = two_sided_renyi(p, q0, alpha) if two_sided else renyi(p, q0, alpha)
    return tok, lam, cost


# ---------------------------------------------------------------------------
# M3: realized-divergence individual Renyi filter (Feldman & Zrnic 2021)
# ---------------------------------------------------------------------------
@dataclass
class RenyiFilter:
    """Per-record Renyi filter at a single order ``alpha``.

    Record i may be charged realized costs rho_t (each a function of the
    history and r_i only) until their sum reaches ``budget``; afterwards the
    record's teacher is replaced by q0, which costs 0 under add/remove-r.
    """

    n_records: int
    budget: float
    alpha: float
    spent: np.ndarray = field(init=False)

    def __post_init__(self):
        self.spent = np.zeros(self.n_records)

    def remaining(self, idx):
        return np.maximum(self.budget - self.spent[idx], 0.0)

    def charge(self, idx, cost):
        """Charge costs; ``idx`` may repeat a record (parallel releases), so the
        check is on the aggregated charge per record."""
        idx = np.asarray(idx)
        cost = np.broadcast_to(np.asarray(cost, float), idx.shape)
        total = np.bincount(idx, weights=cost, minlength=self.n_records)
        if np.any(total > self.budget - self.spent + 1e-9):
            raise RuntimeError("filter overspend: a charge exceeded the remaining budget")
        self.spent += total


def filtered_step(q, q0, remaining, b_max: float, alpha: float, rng: np.random.Generator):
    """One M3 token per state: lam chosen so the two-sided cost <= min(b_max, remaining).

    Returns (tokens, lam, realized_cost); the caller charges realized_cost.
    """
    bound = np.minimum(b_max, remaining)
    lam = lambda_for_bound(q, q0, bound, alpha, two_sided=True)
    p = mixture(q, q0, lam)
    cost = two_sided_renyi(p, q0, alpha)
    cost = np.where(lam > 0, cost, 0.0)
    tok = sample_rows(p, rng)
    return tok, lam, cost


def sample_rows(p, rng: np.random.Generator):
    p = np.atleast_2d(p)
    u = rng.random(p.shape[0])[:, None]
    c = np.cumsum(p, -1)
    c[:, -1] = 1.0
    return (u > c).sum(-1)
