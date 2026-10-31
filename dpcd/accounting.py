"""Privacy accounting for DP privileged-context distillation (plan Phase 0).

Conventions
-----------
* ``mu`` is the Gaussian-DP parameter (Dong, Roth & Su 2019). A mechanism is
  mu-GDP if its trade-off curve is no worse than that of N(0,1) vs N(mu,1).
* Releases that are Gaussian with L2 sensitivity ``sens`` and std ``sigma``
  are (sens/sigma)-GDP; R of them compose to sqrt(R)*sens/sigma.
* RDP curves are functions alpha -> rho(alpha). A mu-GDP mechanism has the
  RDP curve rho(alpha) = alpha * mu^2 / 2 (Gaussian mechanism).

All numbers in the plan's budget table are reproduced by ``budget_table``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy import optimize, stats

DEFAULT_ORDERS: tuple[float, ...] = tuple(
    [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
    + list(np.arange(5, 64, 1.0))
    + [64.0, 80.0, 96.0, 128.0, 256.0]
)


# ---------------------------------------------------------------------------
# Gaussian DP
# ---------------------------------------------------------------------------
def gdp_delta(eps: float, mu: float) -> float:
    """delta(eps) of a mu-GDP mechanism (exact, Dong-Roth-Su Cor. 2.13)."""
    if mu <= 0:
        return 0.0
    return float(
        stats.norm.cdf(-eps / mu + mu / 2) - math.exp(eps) * stats.norm.cdf(-eps / mu - mu / 2)
    )


def gdp_mu(eps: float, delta: float) -> float:
    """Largest mu such that mu-GDP implies (eps, delta)-DP."""
    f = lambda m: gdp_delta(eps, m) - delta
    return float(optimize.brentq(f, 1e-6, 100.0, xtol=1e-12))


def gdp_eps(mu: float, delta: float) -> float:
    """Smallest eps such that mu-GDP implies (eps, delta)-DP."""
    if gdp_delta(0.0, mu) <= delta:
        return 0.0
    return float(optimize.brentq(lambda e: gdp_delta(e, mu) - delta, 0.0, 500.0, xtol=1e-12))


def gaussian_sigma(n_releases: int, sensitivity: float, mu: float) -> float:
    """Noise std so that ``n_releases`` Gaussian releases compose to mu-GDP."""
    return sensitivity * math.sqrt(n_releases) / mu


# ---------------------------------------------------------------------------
# SNR bookkeeping from the feasibility note (generalised P2)
# ---------------------------------------------------------------------------
def total_snr(n: int, mu: float, s_over_c: float = 1.0, sens_over_c: float = 2.0) -> float:
    """SNR of the summed update for a fully shared component of size s.

    Every record releases R values v = s + N(0, sigma^2) with
    sigma = sens*sqrt(R)/mu. Summing n*R releases gives signal n*R*s and noise
    std sigma*sqrt(n*R), so SNR = (s/sens) * mu * sqrt(n); with replace-r
    sensitivity sens = 2c this is (s/c) * mu * sqrt(n) / 2. R cancels.
    """
    return s_over_c / sens_over_c * mu * math.sqrt(n)


def per_step_snr(mu: float, batch: int, epochs: float, sens_over_c: float = 2.0) -> float:
    """SNR of one step's batch mean, mu*sqrt(b)/(sens/c * sqrt(E))."""
    return mu * math.sqrt(batch) / (sens_over_c * math.sqrt(epochs))


# ---------------------------------------------------------------------------
# RDP -> (eps, delta)
# ---------------------------------------------------------------------------
def rdp_to_eps(rho: Callable[[float], float] | Sequence[float], delta: float,
               orders: Sequence[float] = DEFAULT_ORDERS, improved: bool = True) -> tuple[float, float]:
    """Convert an RDP curve to (eps, delta)-DP; returns (eps, best order).

    ``improved`` uses Canonne-Kamath-Steinke (2020) Prop. 12 / Asoodeh et al.:
        eps = rho + log((a-1)/a) - (log delta + log a)/(a-1),
    otherwise the classical Mironov (2017) bound eps = rho + log(1/delta)/(a-1).
    """
    vals = rho if not callable(rho) else [rho(a) for a in orders]
    best = (math.inf, float("nan"))
    for a, r in zip(orders, vals):
        if a <= 1:
            continue
        if improved:
            e = r + math.log((a - 1) / a) - (math.log(delta) + math.log(a)) / (a - 1)
        else:
            e = r + math.log(1 / delta) / (a - 1)
        if e < best[0]:
            best = (max(e, 0.0), a)
    return best


def rdp_budget_at_order(eps: float, delta: float, alpha: float, improved: bool = True) -> float:
    """Largest RDP value at a single order alpha that still certifies (eps, delta)."""
    if improved:
        return eps - math.log((alpha - 1) / alpha) + (math.log(delta) + math.log(alpha)) / (alpha - 1)
    return eps - math.log(1 / delta) / (alpha - 1)


def mu_rdp(eps: float, delta: float, orders: Sequence[float] = DEFAULT_ORDERS,
           improved: bool = True) -> tuple[float, float]:
    """Largest mu whose Gaussian RDP curve alpha*mu^2/2 certifies (eps, delta).

    This is the budget available to mechanisms accounted with RDP (M2, M3),
    versus ``gdp_mu`` for mechanisms accounted with exact GDP (M1).
    Returns (mu, optimal order).
    """
    best = (0.0, float("nan"))
    for a in orders:
        if a <= 1:
            continue
        b = rdp_budget_at_order(eps, delta, a, improved)
        if b <= 0:
            continue
        m = math.sqrt(2 * b / a)
        if m > best[0]:
            best = (m, a)
    return best


# ---------------------------------------------------------------------------
# Token caps for fixed per-token Renyi bound (DP-Fusion style, arm M2)
# ---------------------------------------------------------------------------
def max_private_tokens(mu: float, b2: float) -> float:
    """First-order cap N <= mu^2/b2 on private-context tokens per record.

    For small mixing D_a(p_mix||q0) ~ (a/2) * lambda^2 chi^2 = (a/2) b2 for every
    order a, so each token is ~ sqrt(b2)-GDP and N tokens compose to sqrt(N b2).
    """
    return mu * mu / b2


def eps_from_single_order(total_rdp: float, alpha: float, delta: float, improved: bool = True) -> float:
    """eps certified when only one RDP order is tracked."""
    return rdp_to_eps([total_rdp], delta, orders=[alpha], improved=improved)[0]


# ---------------------------------------------------------------------------
# DP-SGD baseline accounting (Poisson-sampled Gaussian, PLD) via dp_accounting
# ---------------------------------------------------------------------------
def dpsgd_epsilon(noise_multiplier: float, sample_rate: float, steps: int, delta: float,
                  accountant: str = "pld", discretization: float = 1e-4, replace: bool = False) -> float:
    """eps of Poisson-sampled DP-SGD. ``replace=True`` uses the REPLACE_ONE relation
    (each of the two records' clipped contributions has norm <= C), which is what
    plain DP-OPSD needs under add/remove-r when x stays in the data."""
    import dp_accounting
    from dp_accounting import pld, rdp

    rel = (dp_accounting.NeighboringRelation.REPLACE_ONE if replace
           else dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE)
    event = dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(sample_rate, dp_accounting.GaussianDpEvent(noise_multiplier)),
        steps,
    )
    if accountant == "pld":
        acc = pld.PLDAccountant(neighboring_relation=rel, value_discretization_interval=discretization)
    else:
        acc = rdp.RdpAccountant(neighboring_relation=rel)
    acc.compose(event)
    return float(acc.get_epsilon(delta))


def dpsgd_noise_multiplier(eps: float, delta: float, sample_rate: float, steps: int,
                           accountant: str = "pld", replace: bool = False) -> float:
    """Smallest noise multiplier giving eps at delta (bisection on log scale)."""
    lo, hi = 0.3, 50.0
    if dpsgd_epsilon(hi, sample_rate, steps, delta, accountant, replace=replace) > eps:
        raise ValueError("eps unreachable")
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        if dpsgd_epsilon(mid, sample_rate, steps, delta, accountant, replace=replace) > eps:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1 + 1e-4:
            break
    return hi


def gaussian_composition_eps_pld(sigma_over_sens: float, n_releases: int, delta: float,
                                 discretization: float = 1e-4) -> float:
    """eps of n Gaussian releases via the PLD accountant (cross-check of GDP)."""
    import dp_accounting
    from dp_accounting import pld

    acc = pld.PLDAccountant(value_discretization_interval=discretization)
    acc.compose(dp_accounting.GaussianDpEvent(sigma_over_sens), n_releases)
    return float(acc.get_epsilon(delta))


@dataclass(frozen=True)
class BudgetRow:
    eps: float
    mu_gdp: float
    mu_rdp: float
    rdp_order: float
    snr: dict
    tokens_gdp: dict
    tokens_rdp: dict


def budget_table(eps_list: Iterable[float] = (1, 2, 4, 8), delta: float = 1e-6,
                 ns: Iterable[int] = (220, 512, 4096, 16384, 32768),
                 b2s: Iterable[float] = (0.01, 0.03, 0.1)) -> list[BudgetRow]:
    rows = []
    for e in eps_list:
        m = gdp_mu(e, delta)
        mr, a = mu_rdp(e, delta)
        rows.append(BudgetRow(
            eps=e, mu_gdp=m, mu_rdp=mr, rdp_order=a,
            snr={n: total_snr(n, m) for n in ns},
            tokens_gdp={b: max_private_tokens(m, b) for b in b2s},
            tokens_rdp={b: max_private_tokens(mr, b) for b in b2s},
        ))
    return rows
