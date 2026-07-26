"""Policy-surprise weighting math and no-op / replication behavior."""

from __future__ import annotations

import numpy as np

from engine.policy_surprise import (
    frequency_weights_from_surprises,
    policy_kl_target_from_prior,
    replicate_counts_from_weights,
)
from engine.selfplay import Sample, expand_samples_by_policy_surprise


def test_kl_zero_when_prior_matches_target() -> None:
    p = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    assert policy_kl_target_from_prior(p, p) == 0.0


def test_kl_positive_when_target_differs() -> None:
    prior = np.array([0.8, 0.1, 0.1], dtype=np.float64)
    target = np.array([0.1, 0.1, 0.8], dtype=np.float64)
    kl = policy_kl_target_from_prior(prior, target)
    assert kl > 0.5


def test_kl_handles_unnormalized_and_empty() -> None:
    assert policy_kl_target_from_prior([1.0, 1.0], [3.0, 1.0]) >= 0.0
    assert policy_kl_target_from_prior([], []) == 0.0
    assert policy_kl_target_from_prior([1.0], [1.0, 2.0]) == 0.0


def test_frequency_weights_identity_when_alpha_zero_or_no_surprise() -> None:
    s = [0.0, 1.0, 2.0]
    np.testing.assert_allclose(frequency_weights_from_surprises(s, 0.0), [1, 1, 1])
    np.testing.assert_allclose(frequency_weights_from_surprises([0, 0, 0], 0.5), [1, 1, 1])


def test_frequency_weights_half_uniform_half_proportional() -> None:
    s = [0.0, 1.0, 3.0]
    w = frequency_weights_from_surprises(s, 0.5)
    # base 0.5 + 0.5 * 3 * s_i / 4
    expected = np.array([0.5, 0.5 + 0.5 * 3 * 1 / 4, 0.5 + 0.5 * 3 * 3 / 4])
    np.testing.assert_allclose(w, expected)
    assert abs(float(w.sum()) - 3.0) < 1e-9


def test_expand_weight_zero_is_same_list_object() -> None:
    samples = [
        Sample(np.zeros((20, 8, 8), np.float16), np.zeros(8, np.float16), True, policy_surprise=1.0)
    ]
    out = expand_samples_by_policy_surprise(samples, 0.0)
    assert out is samples


def test_expand_expected_count_near_n() -> None:
    samples = [
        Sample(
            np.zeros((20, 8, 8), np.float16),
            np.zeros(8, np.float16),
            True,
            value=float(i),
            policy_surprise=float(i),
        )
        for i in range(20)
    ]
    rng = np.random.default_rng(0)
    totals = [
        len(expand_samples_by_policy_surprise(samples, 0.5, rng=rng))
        for _ in range(200)
    ]
    mean = float(np.mean(totals))
    assert 18.0 <= mean <= 22.0


def test_replicate_counts_nonnegative() -> None:
    rng = np.random.default_rng(1)
    counts = replicate_counts_from_weights([0.0, 0.4, 1.7, 2.2], rng)
    assert all(c >= 0 for c in counts)
    assert counts[0] == 0
    assert counts[2] in (1, 2)
