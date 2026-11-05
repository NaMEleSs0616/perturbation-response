"""Held-out residual calibration of per-gene prediction intervals.

This is split-conformal style: residuals on calibration perturbations, normalised per gene, give a
quantile that scales interval widths. Three separate quantities should not be confused:
  * model disagreement: spread across ensemble members (depends on the training sample and seeds)
  * measurement variability: standard error of the observed response (depends on cell number)
  * calibrated residual interval: empirical held-out error, which includes both and any model bias

Formal conformal coverage assumes exchangeable scores. Genes within a perturbation are dependent, and
calibration perturbations were also used for hyperparameter selection and early stopping, so the
coverage here is an empirical property that is measured on test perturbations, not a guarantee.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class GeneIntervalCalibrator:
    alpha: float = 0.1
    sigma: np.ndarray | None = None
    q: float = float("nan")
    n_calibration: int = 0

    def fit(self, Y: np.ndarray, P: np.ndarray, mask: np.ndarray) -> "GeneIntervalCalibrator":
        resid = np.where(mask, np.nan, np.abs(Y - P))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a gene masked in every calibration condition
            sigma = np.nanmedian(resid, axis=0)
        floor = max(float(np.nanmedian(sigma)) * 0.1, 1e-4)
        self.sigma = np.maximum(np.nan_to_num(sigma, nan=floor), floor).astype(np.float32)
        scores = (resid / self.sigma)[~mask]
        n = scores.size
        level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.q = float(np.quantile(scores, level))
        self.n_calibration = int(Y.shape[0])
        return self

    def half_width(self) -> np.ndarray:
        return self.q * self.sigma

    def interval(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hw = self.half_width()
        return P - hw, P + hw

    def coverage(self, Y: np.ndarray, P: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Per-perturbation fraction of genes whose observation falls inside the interval."""
        inside = np.abs(Y - P) <= self.half_width()
        return np.nanmean(np.where(mask, np.nan, inside.astype(float)), axis=1)

    def state(self) -> dict:
        return {"alpha": self.alpha, "sigma": self.sigma, "q": self.q, "n_calibration": self.n_calibration}

    @classmethod
    def from_state(cls, s: dict) -> "GeneIntervalCalibrator":
        return cls(alpha=s["alpha"], sigma=s["sigma"], q=s["q"], n_calibration=s["n_calibration"])


def disagreement(members: np.ndarray) -> np.ndarray:
    """Per-perturbation root mean gene-wise variance across ensemble members. (n_members, n, G) -> (n,)"""
    return np.sqrt(members.var(axis=0, ddof=1).mean(axis=1))


def error_uncertainty_relation(errors: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Does an uncertainty score rank perturbations by actual error?"""
    ok = np.isfinite(errors) & np.isfinite(scores)
    if ok.sum() < 5 or np.ptp(scores[ok]) == 0:
        return {"n": int(ok.sum()), "spearman": float("nan"), "p": float("nan")}
    r = stats.spearmanr(errors[ok], scores[ok])
    return {"n": int(ok.sum()), "spearman": float(r.statistic), "p": float(r.pvalue)}
