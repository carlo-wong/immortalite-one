"""KataGo-style policy-surprise frequency weighting helpers.

Among samples from one game, redistribute frequency so ``(1 - α)`` is uniform and
``α`` is proportional to ``KL(π_target || π_prior)`` on the legal-move support.
Replication uses ``floor(w)`` plus a Bernoulli fractional copy.

``α = 0`` yields all-ones weights (identity when callers skip replication).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def policy_kl_target_from_prior(
    prior: np.ndarray | Sequence[float],
    target: np.ndarray | Sequence[float],
) -> float:
    """KL(π_target ‖ π_prior) over aligned legal-move weights.

    Both arrays must be same length and non-negative. They are renormalized
    independently. Returns 0 when either distribution has non-positive mass.
    """
    p = np.asarray(prior, dtype=np.float64).reshape(-1)
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    if p.size == 0 or t.size == 0 or p.size != t.size:
        return 0.0
    if np.any(p < 0) or np.any(t < 0):
        return 0.0
    ps = float(p.sum())
    ts = float(t.sum())
    if ps <= 0.0 or ts <= 0.0:
        return 0.0
    p = np.clip(p / ps, 1e-12, None)
    t = np.clip(t / ts, 1e-12, None)
    return float(np.sum(t * np.log(t / p)))


def frequency_weights_from_surprises(
    surprises: Sequence[float],
    weight: float,
) -> np.ndarray:
    """Return per-sample frequency weights with expected sum ≈ n.

    ``weight`` is α in [0, 1]. If all surprises are 0 (or α<=0), every weight is 1.
    """
    n = len(surprises)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    alpha = float(weight)
    if alpha <= 0.0:
        return np.ones(n, dtype=np.float64)
    if alpha > 1.0:
        alpha = 1.0
    s = np.asarray(surprises, dtype=np.float64).reshape(-1)
    if s.size != n:
        raise ValueError("surprises length mismatch")
    total = float(s.sum())
    if total <= 0.0:
        return np.ones(n, dtype=np.float64)
    return (1.0 - alpha) + alpha * n * (s / total)


def replicate_counts_from_weights(
    weights: Sequence[float],
    rng: np.random.Generator,
) -> list[int]:
    """Integer replication counts from frequency weights (KataGo write rule)."""
    counts: list[int] = []
    for w in weights:
        wf = float(w)
        copies = int(np.floor(wf))
        if float(rng.random()) < (wf - copies):
            copies += 1
        counts.append(max(0, copies))
    return counts
