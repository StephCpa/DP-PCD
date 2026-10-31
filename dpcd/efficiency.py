"""Phase 0 efficiency estimate: usable private tokens per record.

Input is a divergence profile: for each token position along student rollouts,
the privileged teacher q(.|x,r,h) and the public prior q0(.|x,h) (or summary
statistics of them). Output, per scheme and eps, is the number of private
tokens a record can afford and the share of the budget that carries signal.

Common currency: GDP-equivalent distinguishability mu_t^2 per token
(addendum, "first-order equivalence"). A record's run is mu-GDP with
mu^2 = sum_t mu_t^2 for Gaussian releases, and approximately so for mixtures
with D_alpha ~ alpha * mu_t^2 / 2.

Schemes
* clipped Gaussian (M1) at clip c: every release is charged c^2/sigma^2, but
  carries clip(lr_t, c)^2/sigma^2, so efficiency = E[clip(lr)^2]/c^2.
* fixed per-token bound (M2) b_2: every token is charged b_2 (as a GDP-equivalent
  mu_t^2) but carries min(D_t, b_2) when the full teacher is already within the
  bound (lambda capped at 1).
* realized-divergence filter (M3): every token is charged what it carries,
  min(D_t, b_max); efficiency 1 up to the RDP-vs-GDP conversion gap.

Tokens are counted in units of the per-record budget mu^2; mu comes from exact
GDP for M1 and from RDP conversion for M2/M3 (see accounting.mu_rdp).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import accounting as A
from .divergence import chi2, log_ratio, renyi, two_sided_renyi


@dataclass
class Profile:
    """Per-token statistics. ``d_mu2`` is the GDP-equivalent mu_t^2 of releasing the
    full privileged teacher at that token: 2/alpha * two-sided D_alpha(q||q0)."""

    d_mu2: np.ndarray
    lr: np.ndarray  # log q(a) - log q0(a) at the token the student actually sampled
    alpha: float

    @classmethod
    def from_distributions(cls, q, q0, sampled_tokens, alpha: float):
        d = two_sided_renyi(q, q0, alpha)
        return cls(d_mu2=2.0 * d / alpha, lr=log_ratio(q, q0, sampled_tokens), alpha=alpha)


def gaussian_efficiency(lr, c):
    return float(np.mean(np.minimum(np.abs(lr), c) ** 2) / c**2)


def fixed_b_efficiency(d_mu2, b2):
    return float(np.mean(np.minimum(d_mu2, b2)) / b2)


def report(profile: Profile, eps: float, delta: float = 1e-6, cs=(0.5, 1.0, 2.0),
           b2s=(0.01, 0.03, 0.1), b_max_mult: float = 1.0) -> dict:
    mu_g = A.gdp_mu(eps, delta)
    mu_r, order = A.mu_rdp(eps, delta)
    out = {"eps": eps, "mu_gdp": mu_g, "mu_rdp": mu_r, "rdp_order": order, "gaussian": {}, "fixed_b": {},
           "filter": {}}
    for c in cs:
        eff = gaussian_efficiency(profile.lr, c)
        out["gaussian"][c] = {"efficiency": eff, "signal_mu2": mu_g**2 * eff,
                              "clip_fraction": float(np.mean(np.abs(profile.lr) > c))}
    for b in b2s:
        eff = fixed_b_efficiency(profile.d_mu2, b)
        n_fixed = mu_r**2 / b
        mean_cost = float(np.mean(np.minimum(profile.d_mu2, b * b_max_mult)))
        n_filter = mu_r**2 / max(mean_cost, 1e-12)
        out["fixed_b"][b] = {"efficiency": eff, "tokens": n_fixed, "signal_mu2": mu_r**2 * eff,
                             "share_lambda_capped": float(np.mean(profile.d_mu2 <= b))}
        out["filter"][b] = {"tokens": n_filter, "signal_mu2": mu_r**2, "gain_tokens_vs_fixed": n_filter / n_fixed,
                            "gain_signal_vs_fixed": 1.0 / max(eff, 1e-12)}
    return out


def concentration(d_mu2, top=0.1):
    """Share of total divergence carried by the top ``top`` fraction of tokens."""
    s = np.sort(np.asarray(d_mu2))[::-1]
    k = max(1, int(round(top * len(s))))
    return float(s[:k].sum() / max(s.sum(), 1e-300))
