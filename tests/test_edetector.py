"""Correctness and validity of the betting e-detector."""

import numpy as np
import pytest

from sentinel_e.betting import AdaptiveLinearBet, LinearBet, MixtureBet, PowerBet
from sentinel_e.edetector import ChangepointPrior, EDetector


# --------------------------------------------------------------------------- #
# Betting functions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kappa", [0.05, 0.3, 0.9])
def test_power_bet_integrates_to_one(kappa):
    """int_eps^1 kappa p^(kappa-1) dp = 1 - eps^kappa.

    The integrand has an integrable singularity at zero, so it is integrated on
    a log-spaced grid and compared against the exact antiderivative rather than
    trapezoided on a uniform grid.
    """
    eps = 1e-8
    p = np.exp(np.linspace(np.log(eps), 0.0, 400_000))
    num = np.trapezoid(PowerBet(kappa).factors(p), p)
    assert num == pytest.approx(1.0 - eps**kappa, rel=1e-4)


@pytest.mark.parametrize("lam", [0.0, 0.4, 1.0])
def test_linear_bet_is_an_e_value_and_non_negative(lam):
    p = np.linspace(0.0, 1.0, 100001)
    f = LinearBet(lam).factors(p)
    assert f.min() >= 0.0
    assert np.trapezoid(f, p) == pytest.approx(1.0, rel=1e-6)


def test_betting_functions_are_non_increasing():
    p = np.linspace(1e-6, 1.0, 10000)
    assert np.all(np.diff(PowerBet(0.3).factors(p)) <= 1e-12)
    assert np.all(np.diff(LinearBet(0.7).factors(p)) <= 1e-12)


def test_mixture_prior_normalises():
    m = MixtureBet("power", n_grid=16)
    assert m.prior.sum() == pytest.approx(1.0)
    assert len(m) == 16
    assert np.all(m.factors(0.5) > 0)


def test_adaptive_bet_learns_towards_small_p():
    bet = AdaptiveLinearBet(lam0=0.1)
    for _ in range(500):
        bet.update(0.01)
    assert bet.lam > 0.5
    bet.reset()
    assert bet.lam == pytest.approx(0.1)


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        PowerBet(1.5)
    with pytest.raises(ValueError):
        EDetector(alpha=0.0)
    with pytest.raises(ValueError):
        ChangepointPrior(kind="nonsense")


# --------------------------------------------------------------------------- #
# The recursion
# --------------------------------------------------------------------------- #
def test_recursion_matches_brute_force_mixture():
    """The O(1) recursion must equal the explicit sum over changepoint locations."""
    rng = np.random.default_rng(0)
    p = rng.uniform(size=60)
    rho = 0.02
    det = EDetector(alpha=0.01, family="power", n_grid=1,
                    prior=ChangepointPrior("geometric", rho), restart=False)
    kappa = det.mixture.grid[0]
    f = kappa * np.clip(p, 1e-12, 1.0) ** (kappa - 1.0)

    for t in range(1, len(p) + 1):
        det.step(p[t - 1])
        brute = sum(
            rho * (1 - rho) ** (j - 1) * np.prod(f[j - 1:t]) for j in range(1, t + 1)
        ) + (1 - rho) ** t
        assert det.log_wealth() == pytest.approx(np.log(brute), rel=1e-9)


def test_scale_free_prior_tail_mass_is_exact():
    det = EDetector(alpha=0.01, family="power", n_grid=1,
                    prior=ChangepointPrior("scale_free"), restart=False)
    for t in range(1, 30):
        det.step(1.0)
        # Q_t = 1/(t+1) for the scale-free prior.
        assert np.exp(det.state.log_Q) == pytest.approx(1.0 / (t + 1), rel=1e-10)


def test_wealth_starts_at_one():
    det = EDetector(alpha=0.01, restart=False)
    assert det.log_wealth() == pytest.approx(0.0)
    assert det.e_value() == pytest.approx(1.0)


def test_abstention_is_a_neutral_bet():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=200)
    a = EDetector(alpha=0.01, restart=False)
    before = a.log_wealth()
    for x in p:
        a.step(x, abstain=True)
    # Only the changepoint prior mass moves; the betting factor is exactly one.
    assert a.log_wealth() <= before + 1e-12
    assert a.log_wealth() == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Statistical guarantees
# --------------------------------------------------------------------------- #
def test_expected_wealth_is_at_most_one():
    """E_inf[W_t] <= 1 for every fixed t (the supermartingale property)."""
    rng = np.random.default_rng(2)
    T, reps = 200, 2500
    finals = np.empty(reps)
    for r in range(reps):
        det = EDetector(alpha=0.001, family="power", n_grid=8, restart=False)
        lw, _ = det.run(rng.uniform(size=T))
        finals[r] = np.exp(lw[-1])
    mean = finals.mean()
    se = finals.std(ddof=1) / np.sqrt(reps)
    assert mean <= 1.0 + 4 * se


@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_ville_bound_holds_under_the_null(alpha):
    """P_inf(sup_t W_t >= 1/alpha) <= alpha, uniformly over the horizon."""
    rng = np.random.default_rng(3)
    T, reps = 2000, 800
    fired = 0
    for r in range(reps):
        det = EDetector(alpha=alpha, family="power", n_grid=16, restart=False)
        _, al = det.run(rng.uniform(size=T))
        fired += int(al.any())
    rate = fired / reps
    assert rate <= alpha + 3.5 * np.sqrt(alpha * (1 - alpha) / reps)


def test_predictable_modulation_preserves_validity():
    """Arbitrary *predictable* hazard/stake choices must not break the bound."""
    rng = np.random.default_rng(4)
    alpha, T, reps = 0.05, 1200, 700
    fired = 0
    for r in range(reps):
        det = EDetector(alpha=alpha, family="linear", n_grid=8, restart=False)
        p = rng.uniform(size=T)
        prev = 0.5
        for t in range(T):
            # Depends only on the previous p-value: adversarial but predictable.
            hazard = 0.05 if prev < 0.2 else 1e-4
            stake = 1.0 if prev < 0.5 else 0.1
            res = det.step(p[t], hazard=hazard, stake_scale=stake)
            prev = p[t]
            if res.alarm:
                fired += 1
                break
    rate = fired / reps
    assert rate <= alpha + 3.5 * np.sqrt(alpha * (1 - alpha) / reps)


def test_super_uniform_p_values_are_also_safe():
    """Conservative (stochastically larger) p-values can only reduce the risk."""
    rng = np.random.default_rng(5)
    alpha, reps = 0.05, 500
    fired = 0
    for r in range(reps):
        det = EDetector(alpha=alpha, family="power", n_grid=16, restart=False)
        p = np.minimum(1.0, rng.uniform(size=2000) + 0.05)
        _, al = det.run(p)
        fired += int(al.any())
    assert fired / reps <= alpha + 3.5 * np.sqrt(alpha * (1 - alpha) / reps)


def test_detects_a_real_change_quickly():
    rng = np.random.default_rng(6)
    p = np.concatenate([rng.uniform(size=500), rng.uniform(size=500) ** 8])
    det = EDetector(alpha=0.01, family="power", n_grid=32, restart=False)
    _, al = det.run(p)
    idx = np.flatnonzero(al)
    assert idx.size > 0 and 500 <= idx[0] < 600


def test_restart_resets_the_wealth():
    rng = np.random.default_rng(7)
    p = rng.uniform(size=300) ** 10
    det = EDetector(alpha=0.05, family="power", n_grid=8, restart=True)
    lw, al = det.run(p)
    assert al.sum() > 1                      # keeps monitoring after an alarm
    assert det.state.n_restarts == al.sum()
    fire = np.flatnonzero(al)[0]
    assert lw[fire + 1] < lw[fire]           # wealth dropped back after restart
