"""Phase 0 accountant tests (plan: 'Accountant tests')."""
import math

import numpy as np
import pytest

from dpcd import accounting as A
from dpcd import audit, divergence as D, witness as W
from dpcd.mechanisms import RenyiFilter, filtered_step

DELTA = 1e-6


@pytest.mark.parametrize("eps,mu", [(1, 0.237), (2, 0.448), (4, 0.838), (8, 1.532)])
def test_gdp_mu_matches_feasibility_note(eps, mu):
    assert round(A.gdp_mu(eps, DELTA), 3) == mu


def test_gdp_roundtrip():
    for eps in [0.5, 1, 4, 8]:
        assert A.gdp_eps(A.gdp_mu(eps, DELTA), DELTA) == pytest.approx(eps, abs=1e-6)


@pytest.mark.parametrize("eps,R", [(1, 1), (4, 100), (8, 1600)])
def test_gaussian_composition_against_pld(eps, R):
    """R Gaussian releases calibrated by GDP give eps under the PLD accountant."""
    mu = A.gdp_mu(eps, DELTA)
    sigma_over_sens = math.sqrt(R) / mu
    e_pld = A.gaussian_composition_eps_pld(sigma_over_sens, R, DELTA)
    assert e_pld == pytest.approx(eps, rel=0.02)


def test_rdp_conversion_matches_dp_accounting():
    import dp_accounting
    from dp_accounting import rdp

    sigma = 3.0
    acc = rdp.RdpAccountant(orders=list(A.DEFAULT_ORDERS))
    acc.compose(dp_accounting.GaussianDpEvent(sigma), 50)
    ours, _ = A.rdp_to_eps(lambda a: 50 * a / (2 * sigma**2), DELTA)
    assert ours == pytest.approx(acc.get_epsilon(DELTA), rel=1e-3)


def test_rdp_mu_below_gdp_mu():
    for eps in [1, 4, 8]:
        assert A.mu_rdp(eps, DELTA)[0] < A.gdp_mu(eps, DELTA)


def test_order2_only_cannot_certify_small_eps():
    # with only order 2 tracked, even zero RDP spend certifies eps >= ~12.4 at delta 1e-6
    assert A.eps_from_single_order(0.0, 2.0, DELTA) > 12


def test_snr_reproduces_note():
    assert A.total_snr(4096, A.gdp_mu(1, DELTA)) == pytest.approx(7.6, abs=0.05)
    assert A.total_snr(16384, A.gdp_mu(8, DELTA)) == pytest.approx(98, abs=0.5)
    # minimum experiment sigma per release: 338c / 95c / 52c
    for eps, s in [(1, 338), (4, 95), (8, 52)]:
        assert A.gaussian_sigma(1600, 2.0, A.gdp_mu(eps, DELTA)) == pytest.approx(s, abs=1)


def test_renyi_bernoulli_closed_form_and_limits():
    p = np.array([0.3, 0.7])
    q = np.array([0.6, 0.4])
    a = 3.0
    closed = math.log(0.3**3 * 0.6**-2 + 0.7**3 * 0.4**-2) / 2
    assert D.renyi(p, q, a) == pytest.approx(closed)
    assert D.renyi(p, q, 1.0001) == pytest.approx(D.renyi(p, q, 1.0), rel=1e-3)
    assert D.renyi(p, q, 2.0) == pytest.approx(math.log1p(D.chi2(p, q)))


def test_small_mixing_first_order():
    rng = np.random.default_rng(0)
    q = rng.dirichlet(np.ones(20))
    q0 = rng.dirichlet(np.ones(20))
    lam = 1e-3
    p = D.mixture(q, q0, lam)
    for a in [2.0, 4.0, 8.0]:
        assert D.renyi(p, q0, a) == pytest.approx(a / 2 * lam**2 * D.chi2(q, q0), rel=0.05)


def test_lambda_search_hits_bound():
    rng = np.random.default_rng(1)
    q = rng.dirichlet(0.2 * np.ones(50), size=64)
    q0 = rng.dirichlet(np.ones(50), size=64)
    for two in [True, False]:
        lam = D.lambda_for_bound(q, q0, 0.01, 4.0, two_sided=two)
        p = D.mixture(q, q0, lam)
        d = D.two_sided_renyi(p, q0, 4.0) if two else D.renyi(p, q0, 4.0)
        assert np.all(d <= 0.01 + 1e-9)
        assert np.all((lam == 1.0) | (d > 0.0099))


def test_filter_never_overspends_and_falls_back():
    rng = np.random.default_rng(2)
    filt = RenyiFilter(n_records=8, budget=0.05, alpha=4.0)
    idx = np.arange(8)
    for _ in range(200):
        q = rng.dirichlet(0.3 * np.ones(30), size=8)
        q0 = rng.dirichlet(np.ones(30), size=8)
        tok, lam, cost = filtered_step(q, q0, filt.remaining(idx), 0.01, 4.0, rng)
        filt.charge(idx, cost)
    assert np.all(filt.spent <= 0.05 + 1e-9)
    assert np.all(filt.spent > 0.049)  # exhausted, then free fallback to q0


def test_filter_witness_known_toy():
    """Exact output distribution of the filtered sampler stays within its RDP budget."""
    alpha, B = 4.0, 0.2
    for seed in range(5):
        lm = W.hashed_toy_lm(V=3, T=6, seed=seed, strength=5.0)
        lp, lp0, ledger = W.enumerate_mechanism(lm, "filter", alpha, budget=B, b_max=0.08)
        f, b = W.exact_renyi(lp, lp0, alpha)
        assert ledger <= B + 1e-9
        assert f <= B + 1e-9 and b <= B + 1e-9


def test_one_run_audit_ceiling():
    # a perfect attacker with 128 canaries cannot certify eps >= 4
    assert audit.max_certifiable_eps(128) < 4
    assert audit.one_run_eps_lower_bound(64, 32, 128) == 0.0


def test_filter_detects_parallel_overspend():
    filt = RenyiFilter(n_records=2, budget=1.0, alpha=4.0)
    filt.charge(np.array([0, 0]), np.array([0.4, 0.4]))
    with pytest.raises(RuntimeError):
        filt.charge(np.array([0, 0, 1]), np.array([0.15, 0.15, 0.1]))  # 0.8 + 0.3 > 1


def test_pilot_m3_ledger_within_budget():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dpcd import pilot as P
    task = P.make_task(P.TaskCfg(), 0)
    x, r = P.make_records(task, 512, np.random.default_rng(0))
    _, log = P.train(task, P.RunCfg(arm="M3", eps=4.0, lr=0.01, g=4.0, lam_hat=0.2, epochs=2), x, r)
    assert log["ledger_mean_frac"] <= 1.0 + 1e-9
