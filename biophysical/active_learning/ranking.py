"""Candidate ranking for follow-up experiments.

Every score here is a documented heuristic. None is exact expected information gain, and none is
claimed to be optimal; retrospective comparisons against random selection are in
biophysical/active_learning/simulate.py.

  uncertainty : ensemble disagreement over genes
  magnitude   : predicted response size (excludes targets), for finding strong responders
  expected_improvement : Gaussian approximation over member magnitudes, for the objective
                "exceed the largest response magnitude measured so far"
  diversity   : greedy k-center in feature space, starting from measured conditions
  hybrid      : greedy trade-off of normalised disagreement and normalised distance from the
                measured and already-selected conditions
  random      : seeded baseline
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

STRATEGIES = ("uncertainty", "magnitude", "expected_improvement", "diversity", "hybrid", "random")


@dataclass
class CandidateSet:
    names: list[str]
    X: np.ndarray                      # standardised features
    uncertainty: np.ndarray
    magnitude: np.ndarray
    member_magnitude: np.ndarray | None = None   # (n_members, n)
    meta: dict = field(default_factory=dict)


def _norm01(v: np.ndarray) -> np.ndarray:
    span = np.ptp(v)
    return (v - v.min()) / span if span > 0 else np.zeros_like(v, dtype=float)


def _min_dist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    if Y.shape[0] == 0:
        return np.full(X.shape[0], np.inf)
    d = np.full(X.shape[0], np.inf)
    for s in range(0, Y.shape[0], 256):
        d = np.minimum(d, np.linalg.norm(X[:, None, :] - Y[s:s + 256][None, :, :], axis=2).min(axis=1))
    return d


def expected_improvement(member_magnitude: np.ndarray, best: float) -> np.ndarray:
    """E[max(M - best, 0)] for M ~ N(mean, var) with mean and var from ensemble members."""
    mu, sd = member_magnitude.mean(axis=0), member_magnitude.std(axis=0, ddof=1)
    sd = np.maximum(sd, 1e-9)
    z = (mu - best) / sd
    return (mu - best) * stats.norm.cdf(z) + sd * stats.norm.pdf(z)


def select(c: CandidateSet, budget: int, strategy: str = "hybrid", X_measured: np.ndarray | None = None, best_magnitude: float | None = None,
           diversity_weight: float = 0.5, seed: int = 0) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return selected indices in order, a rationale per selection, and the score at selection."""
    n = len(c.names)
    budget = min(budget, n)
    measured = X_measured if X_measured is not None else np.zeros((0, c.X.shape[1]))
    if strategy == "random":
        idx = np.random.default_rng(seed).permutation(n)[:budget]
        return idx, ["random baseline"] * budget, np.full(budget, np.nan)
    if strategy == "uncertainty":
        idx = np.argsort(-c.uncertainty, kind="stable")[:budget]
        return idx, [f"ensemble disagreement {c.uncertainty[i]:.4f}" for i in idx], c.uncertainty[idx]
    if strategy == "magnitude":
        idx = np.argsort(-c.magnitude, kind="stable")[:budget]
        return idx, [f"predicted response magnitude {c.magnitude[i]:.4f}" for i in idx], c.magnitude[idx]
    if strategy == "expected_improvement":
        if c.member_magnitude is None or best_magnitude is None:
            raise ValueError("expected_improvement requires member magnitudes and the best measured magnitude")
        ei = expected_improvement(c.member_magnitude, best_magnitude)
        idx = np.argsort(-ei, kind="stable")[:budget]
        return idx, [f"EI {ei[i]:.5f} over best measured magnitude {best_magnitude:.3f} (Gaussian approximation)" for i in idx], ei[idx]
    if strategy not in ("diversity", "hybrid"):
        raise ValueError(f"unknown strategy '{strategy}'; choose from {STRATEGIES}")

    dist = _min_dist(c.X, measured)
    u = _norm01(c.uncertainty)
    selected: list[int] = []
    reasons: list[str] = []
    scores: list[float] = []
    for _ in range(budget):
        finite = dist[np.isfinite(dist)]
        scale = finite.max() if finite.size and finite.max() > 0 else 1.0
        d = np.where(np.isfinite(dist), dist / scale, 1.0)
        score = d if strategy == "diversity" else (1 - diversity_weight) * u + diversity_weight * d
        score = score.copy()
        score[selected] = -np.inf
        j = int(np.argmax(score))
        selected.append(j)
        scores.append(float(score[j]))
        if strategy == "diversity":
            reasons.append(f"farthest from measured and selected conditions (distance {dist[j]:.2f})")
        else:
            reasons.append(f"disagreement {c.uncertainty[j]:.4f} and distance {dist[j]:.2f} from measured and selected")
        dist = np.minimum(dist, np.linalg.norm(c.X - c.X[j], axis=1))
    return np.array(selected, dtype=int), reasons, np.array(scores)


def ranking_table(c: CandidateSet, idx: np.ndarray, reasons: list[str], scores: np.ndarray, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    t = pd.DataFrame({
        "rank": np.arange(1, len(idx) + 1),
        "candidate": [c.names[i] for i in idx],
        "predicted_magnitude": c.magnitude[idx],
        "uncertainty": c.uncertainty[idx],
        "score": scores,
        "rationale": reasons,
    })
    if extra is not None:
        t = pd.concat([t, extra.iloc[idx].reset_index(drop=True)], axis=1)
    return t
