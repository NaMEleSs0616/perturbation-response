"""Per-perturbation metrics. The perturbation, not the gene or cell, is the unit of inference."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

METRIC_DIRECTION = {  # +1: larger is better, -1: smaller is better
    "mae": -1, "mse": -1, "pearson": 1, "spearman": 1, "pearson_top": 1, "mae_top": -1, "sign_acc_top": 1, "topk_recall": 1, "r2_vs_zero": 1,
    "pearson_specific": 1, "spearman_specific": 1, "pearson_specific_top": 1, "skill_vs_mean": 1, "pearson_interaction": 1, "pearson_interaction_top": 1, "skill_vs_additive": 1,
}


def _corr(a: np.ndarray, b: np.ndarray, kind: str = "pearson") -> float:
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.pearsonr(a, b)[0] if kind == "pearson" else stats.spearmanr(a, b)[0])


def perturbation_metrics(y: np.ndarray, p: np.ndarray, mask: np.ndarray | None = None, top_k: int = 20,
                         reference: np.ndarray | None = None) -> dict[str, float]:
    """Metrics for one perturbation; masked entries (the perturbed genes themselves) are excluded.

    The top-k genes are chosen by observed |response|, which is the same set for every model on a
    given perturbation, so these metrics are comparable across models.

    `reference` is the training-mean response. Most perturbations in a screen share a common shift
    from controls, so raw response correlation rewards predicting that shift alone. The *_specific
    metrics correlate deviations from the reference, and skill_vs_mean is 1 - SSE(model)/SSE(mean);
    the mean model itself has an undefined specific correlation and a skill of exactly 0.
    """
    keep = ~mask if mask is not None else np.ones_like(y, dtype=bool)
    y, p = y[keep], p[keep]
    specific: dict[str, float] = {}
    if reference is not None:
        r = reference[keep]
        dy, dp = y - r, p - r
        spec_top = np.argsort(-np.abs(dy), kind="stable")[:min(top_k, y.size)]
        sse_ref = float((dy ** 2).sum())
        specific = {
            "pearson_specific": _corr(dy, dp),
            "spearman_specific": _corr(dy, dp, "spearman"),
            "pearson_specific_top": _corr(dy[spec_top], dp[spec_top]),
            "skill_vs_mean": 1 - float(((y - p) ** 2).sum()) / sse_ref if sse_ref > 0 else float("nan"),
        }
    err = y - p
    k = min(top_k, y.size)
    top = np.argsort(-np.abs(y), kind="stable")[:k]
    pred_top = np.argsort(-np.abs(p), kind="stable")[:k]
    denom = float((y ** 2).sum())
    constant = np.ptp(p) == 0
    return {
        "mae": float(np.abs(err).mean()),
        "mse": float((err ** 2).mean()),
        "pearson": _corr(y, p),
        "spearman": _corr(y, p, "spearman"),
        "pearson_top": _corr(y[top], p[top]),
        "mae_top": float(np.abs(err[top]).mean()),
        "sign_acc_top": float("nan") if constant else float((np.sign(y[top]) == np.sign(p[top])).mean()),
        "topk_recall": float("nan") if constant else float(np.intersect1d(top, pred_top).size / k),
        "r2_vs_zero": 1 - float((err ** 2).sum()) / denom if denom > 0 else float("nan"),
        **specific,
    }


def metrics_frame(ids: list[str], Y: np.ndarray, P: np.ndarray, mask: np.ndarray, top_k: int = 20,
                  reference: np.ndarray | None = None, **extra) -> pd.DataFrame:
    rows = []
    for i, cid in enumerate(ids):
        ref = None if reference is None else reference[i]
        rows.append({"condition": cid, **extra, **perturbation_metrics(Y[i], P[i], mask[i], top_k, ref)})
    return pd.DataFrame(rows)


def response_magnitude(Y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Root mean square response per perturbation, excluding the perturbed genes."""
    return np.sqrt(np.where(mask, 0.0, Y ** 2).sum(axis=1) / np.maximum((~mask).sum(axis=1), 1))


def perturbation_ranking(Y: np.ndarray, P: np.ndarray, mask: np.ndarray) -> float:
    """Spearman correlation across perturbations between predicted and observed response magnitude."""
    return _corr(response_magnitude(Y, mask), response_magnitude(P, mask), "spearman")


def noise_ceiling(Y_a: np.ndarray, Y_b: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    """Reproducibility between two independent response estimates of each perturbation.

    half_r is the raw correlation. reliability applies Spearman-Brown to estimate the reliability of
    the full-data response, and max_r = sqrt(reliability) approximates the correlation a noiseless
    prediction of the true response could reach. These assume independent, equal-variance noise.
    """
    rows = []
    for i in range(Y_a.shape[0]):
        keep = ~mask[i] & np.isfinite(Y_a[i]) & np.isfinite(Y_b[i])
        r = _corr(Y_a[i][keep], Y_b[i][keep]) if keep.sum() >= 3 else float("nan")
        rel = 2 * r / (1 + r) if np.isfinite(r) and r > 0 else 0.0 if np.isfinite(r) else float("nan")
        rows.append({"half_r": r, "reliability": rel, "max_r": float(np.sqrt(rel)) if np.isfinite(rel) else float("nan")})
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap of the mean over perturbations. Combinations sharing a gene are not
    independent, so intervals on combination strata are somewhat too narrow."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, (n_boot, v.size))].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_comparison(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Compare model a against model b over the same perturbations (difference = a - b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    if d.size == 0:
        return {"n": 0, "mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "wilcoxon_p": float("nan"), "frac_a_gt_b": float("nan")}
    mean, lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    p = float("nan")
    if d.size >= 5 and np.any(d != 0):
        p = float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
    return {"n": int(d.size), "mean_diff": mean, "ci_low": lo, "ci_high": hi, "wilcoxon_p": p, "frac_a_gt_b": float((d > 0).mean())}
