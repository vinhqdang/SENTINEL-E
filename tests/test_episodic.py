"""The episodic (Markov-modulated) e-process --- the algorithmic core."""

import numpy as np
import pytest

from sentinel_e.edetector import ChangepointPrior, EDetector
from sentinel_e.episodic import EpisodePrior, EpisodicEDetector


# --------------------------------------------------------------------------- #
# Prior grid
# --------------------------------------------------------------------------- #
def test_prior_grid_is_a_normalised_product_grid():
    pr = EpisodePrior(kappa=(0.1, 0.5), eta=(1e-3, 1e-2), pi=(0.5, 1.0))
    k, e, p, lp = pr.grid()
    assert k.size == e.size == p.size == lp.size == 8 == len(pr)
    assert np.exp(lp).sum() == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [
    dict(kappa=(0.0,)), dict(kappa=(1.5,)), dict(eta=(1.0,)),
    dict(pi=(0.0,)), dict(pi=(1.2,)), dict(rho=0.0), dict(rho=1.0),
])
def test_prior_grid_rejects_invalid_parameters(bad):
    with pytest.raises(ValueError):
        EpisodePrior(**bad).grid()


def test_detector_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        EpisodicEDetector(alpha=0.0)


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #
def test_wealth_starts_at_one_and_posterior_at_zero():
    d = EpisodicEDetector(0.01)
    assert d.log_wealth() == pytest.approx(0.0)
    assert d.e_value() == pytest.approx(1.0)
    assert d.episode_posterior() == pytest.approx(0.0)


def test_recursion_matches_brute_force_over_all_trajectories():
    """The forward recursion must equal the explicit 2^T sum over chain paths."""
    rng = np.random.default_rng(0)
    T = 12
    p = rng.uniform(size=T)
    kappa, eta, pi, rho = 0.3, 0.05, 0.6, 0.02
    prior = EpisodePrior(kappa=(kappa,), eta=(eta,), pi=(pi,), rho=rho)
    det = EpisodicEDetector(0.001, prior, restart=False)

    f = (1 - pi) + pi * kappa * np.clip(p, 1e-12, 1.0) ** (kappa - 1.0)

    for t in range(1, T + 1):
        det.step(p[t - 1])
        total = 0.0
        for mask in range(1 << t):                     # every trajectory z_{1:t}
            z = [(mask >> s) & 1 for s in range(t)]
            prob, prev = 1.0, 0                        # chain starts quiet
            for zt in z:
                prob *= (rho if zt else 1 - rho) if prev == 0 else (
                    (1 - eta) if zt else eta
                )
                prev = zt
            contrib = prob
            for s, zt in enumerate(z):
                if zt:
                    contrib *= f[s]
            total += contrib
        assert det.log_wealth() == pytest.approx(np.log(total), rel=1e-9)


def test_eta_zero_recovers_the_changepoint_mixture_exactly():
    """The episodic process strictly generalises the classical construction."""
    rng = np.random.default_rng(1)
    p = np.concatenate([rng.uniform(size=200), rng.uniform(size=200) ** 6])
    ep = EpisodicEDetector(
        0.01, EpisodePrior(kappa=(0.3,), eta=(0.0,), pi=(1.0,), rho=1e-3),
        restart=False,
    )
    cp = EDetector(0.01, family="power", n_grid=1,
                   prior=ChangepointPrior("geometric", 1e-3), restart=False)
    cp.mixture.grid = np.array([0.3])
    cp.log_prior = np.zeros(1)
    lw_e, _ = ep.run(p)
    lw_c, _ = cp.run(p)
    assert np.allclose(lw_e, lw_c, atol=1e-9)


def test_dilated_emission_is_a_valid_betting_function():
    """(1-pi) + pi*f integrates to at most one and is non-increasing."""
    for pi in (0.2, 0.6, 1.0):
        for kappa in (0.05, 0.4, 0.9):
            d = EpisodicEDetector(
                0.01, EpisodePrior(kappa=(kappa,), eta=(1e-3,), pi=(pi,))
            )
            grid = np.exp(np.linspace(np.log(1e-8), 0.0, 200_000))
            vals = np.array([d._emission(x, 1.0)[0] for x in grid])
            assert np.all(np.diff(vals) <= 1e-9)          # non-increasing
            assert np.trapezoid(vals, grid) <= 1.0 + 1e-3  # integrates to <= 1


def test_neutral_stake_and_abstention_leave_the_wealth_at_one():
    rng = np.random.default_rng(2)
    for kw in ({"stake_scale": 0.0}, {"abstain": True}):
        d = EpisodicEDetector(0.01, restart=False)
        for x in rng.uniform(size=300):
            d.step(x, **kw)
        assert d.log_wealth() == pytest.approx(0.0, abs=1e-9)


def test_fast_path_matches_the_scalar_loop():
    rng = np.random.default_rng(3)
    p = np.concatenate([rng.uniform(size=300), rng.uniform(size=300) ** 5])
    act = rng.random(600) > 0.15
    for restart in (False, True):
        a = EpisodicEDetector(0.01, restart=restart)
        b = EpisodicEDetector(0.01, restart=restart)
        lw1, al1 = a.run(p, active=act)
        lw2 = np.empty(p.size)
        al2 = np.zeros(p.size, dtype=bool)
        for i in range(p.size):
            r = b.step(p[i], abstain=not act[i])
            lw2[i], al2[i] = r.log_wealth, r.alarm
            if r.alarm and not restart:
                lw2[i + 1:] = r.log_wealth
                break
        assert np.allclose(lw1, lw2, atol=1e-9)
        assert np.array_equal(al1, al2)


# --------------------------------------------------------------------------- #
# Guarantees
# --------------------------------------------------------------------------- #
def test_expected_wealth_is_at_most_one():
    rng = np.random.default_rng(4)
    reps, T = 1500, 200
    finals = np.array([
        np.exp(EpisodicEDetector(0.001, restart=False).run(rng.uniform(size=T))[0][-1])
        for _ in range(reps)
    ])
    assert finals.mean() <= 1.0 + 4 * finals.std(ddof=1) / np.sqrt(reps)


@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_ville_bound_holds_under_the_null(alpha):
    rng = np.random.default_rng(5)
    reps, T = 700, 1500
    fired = sum(
        int(EpisodicEDetector(alpha, restart=False).run(rng.uniform(size=T))[1].any())
        for _ in range(reps)
    )
    assert fired / reps <= alpha + 3.5 * np.sqrt(alpha * (1 - alpha) / reps)


def test_predictable_modulation_preserves_validity():
    rng = np.random.default_rng(6)
    alpha, reps, T = 0.05, 500, 1200
    fired = 0
    for _ in range(reps):
        det = EpisodicEDetector(alpha, restart=False)
        prev = 0.5
        for x in rng.uniform(size=T):
            res = det.step(x, hazard=0.05 if prev < 0.2 else 1e-4,
                           stake_scale=1.0 if prev < 0.5 else 0.2)
            prev = x
            if res.alarm:
                fired += 1
                break
    assert fired / reps <= alpha + 3.5 * np.sqrt(alpha * (1 - alpha) / reps)


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #
def test_posterior_rises_inside_an_episode_and_falls_after_it():
    rng = np.random.default_rng(7)
    p = rng.uniform(size=1200)
    p[400:600] = rng.uniform(size=200) ** 8
    det = EpisodicEDetector(1e-9, restart=False)     # threshold out of reach
    post = det.posteriors(p)
    assert post[:380].mean() < 0.2
    assert post[450:600].mean() > 0.7
    assert post[900:].mean() < post[450:600].mean()


def test_recurring_episodes_are_detected_more_reliably_than_by_a_changepoint_mixture():
    """The regime the episodic process exists for: events that recur."""
    rng = np.random.default_rng(8)
    T, REPS = 6000, 60
    miss_ep = miss_cp = 0
    for _ in range(REPS):
        p = rng.uniform(size=T)
        t = 1000
        first = t
        for _ in range(8):                      # eight short, intermittent bursts
            if t + 80 >= T:
                break
            idx = np.arange(t, t + 80)
            act = rng.random(80) < 0.4
            p[idx[act]] = rng.uniform(size=act.sum()) ** 2
            t += 80 + 500
        ep = EpisodicEDetector(0.01, restart=False)
        cp = EDetector(0.01, family="power", n_grid=32,
                       prior=ChangepointPrior("geometric", 1e-3), restart=False)
        miss_ep += int(not ep.run(p)[1][first:].any())
        miss_cp += int(not cp.run(p)[1][first:].any())
    assert miss_ep < miss_cp
