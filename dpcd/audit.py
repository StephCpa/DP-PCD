"""Privacy audits (plan Phase 3).

* ``one_run_eps_lower_bound``: one-run canary audit of Steinke, Nasr and
  Jagielski (2023). Each of m canaries is included independently with
  probability 1/2; the auditor makes r guesses and gets W right. Under
  (eps, delta)-DP, W is stochastically dominated by Binomial(r, e^eps/(1+e^eps))
  up to a delta term. We bound that delta term conservatively by 2*m*delta,
  which is below 3e-4 for every configuration in the plan (m <= 1e3, delta = 1e-6).
* ``max_certifiable_eps``: the largest lower bound any attacker could reach with
  m canaries (all guesses made and all correct). It is the audit's ceiling.
* ``selection_swap``: checks that a token/state selection rule is unchanged
  when one record's r is swapped (selection must read only public quantities).
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy import optimize, stats


def _p(eps):
    return 1.0 / (1.0 + math.exp(-eps))


def one_run_pvalue(eps: float, n_guesses: int, n_correct: int, m: int, delta: float,
                   conservative_delta: bool = False) -> float:
    """p-value of 'M is (eps, delta)-DP' given W = n_correct out of r = n_guesses.

    p = P[Bin(r, q) >= v] + 2 m delta max_{1<=i<=m} P[v-i <= Bin(r, q) < v] / i,
    q = e^eps/(1+e^eps), following the p-value routine of Steinke, Nasr and
    Jagielski (2023). The delta term never exceeds 2 m delta, which
    ``conservative_delta=True`` uses instead.
    """
    q = _p(eps)
    v = n_correct
    tail = stats.binom.sf(v - 1, n_guesses, q)
    if conservative_delta:
        return float(min(1.0, tail + 2 * m * delta))
    i = np.arange(1, m + 1)
    pmf = stats.binom.pmf(v - i, n_guesses, q)  # P[Bin = v - i], zero for v - i < 0
    alpha = float(np.max(np.cumsum(pmf) / i))
    return float(min(1.0, tail + 2 * m * delta * alpha))


def one_run_eps_lower_bound(n_guesses: int, n_correct: int, m: int, delta: float = 1e-6,
                            level: float = 0.05) -> float:
    """Largest eps rejected at ``level``: the audit's (1-level) lower bound on eps."""
    if n_guesses == 0 or one_run_pvalue(0.0, n_guesses, n_correct, m, delta) > level:
        return 0.0
    if one_run_pvalue(50.0, n_guesses, n_correct, m, delta) <= level:
        return 50.0
    return float(optimize.brentq(lambda e: one_run_pvalue(e, n_guesses, n_correct, m, delta) - level,
                                 0.0, 50.0, xtol=1e-6))


def best_lower_bound_from_scores(scores: np.ndarray, included: np.ndarray, m: int | None = None,
                                 delta: float = 1e-6, level: float = 0.05,
                                 ks: list[int] | None = None) -> tuple[float, int]:
    """Guess 'in' for the top-k/2 scores and 'out' for the bottom-k/2.

    The number of guesses k is swept; scanning k costs a multiple-testing
    correction, so ``level`` is split evenly over the k values tried (Bonferroni).
    """
    scores = np.asarray(scores, float)
    included = np.asarray(included, bool)
    m = m or len(scores)
    # random tie-breaking so that tied scores carry no information
    rng = np.random.default_rng(12345)
    order = np.lexsort((rng.random(len(scores)), -scores))
    ks = ks or sorted({k for k in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, m] if k <= m})
    modes = ("both", "in", "out")
    lvl = level / (len(ks) * len(modes))
    best = (0.0, 0)
    for k in ks:
        for mode in modes:
            if mode == "both":
                top, bot = order[: k // 2], order[len(order) - (k - k // 2):]
            elif mode == "in":
                top, bot = order[:k], order[:0]
            else:
                top, bot = order[:0], order[len(order) - k:]
            correct = included[top].sum() + (~included[bot]).sum()
            lb = one_run_eps_lower_bound(k, int(correct), m, delta, lvl)
            if lb > best[0]:
                best = (lb, k)
    return best


def max_certifiable_eps(m: int, delta: float = 1e-6, level: float = 0.05) -> float:
    return one_run_eps_lower_bound(m, m, m, delta, level)


def canaries_needed(eps_claim: float, delta: float = 1e-6, level: float = 0.05,
                    accuracy: float = 1.0, m_max: int = 1_000_000) -> int | None:
    """Smallest m such that an attacker who guesses all m canaries with the given
    accuracy can push the lower bound above ``eps_claim``."""
    def lb(m):
        return one_run_eps_lower_bound(m, int(math.floor(accuracy * m)), m, delta, level)
    m = 8
    while m <= m_max and lb(m) <= eps_claim:
        m *= 2
    if m > m_max:
        return None
    lo, hi = m // 2, m
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if lb(mid) > eps_claim:
            hi = mid
        else:
            lo = mid
    return hi


def selection_swap(select: Callable[[object], np.ndarray], dataset_a, dataset_b) -> int:
    """Number of selections that change when one record's r is swapped (must be 0)."""
    a = np.asarray(select(dataset_a))
    b = np.asarray(select(dataset_b))
    return int((a != b).sum())
