"""Retrospective acquisition experiment on measured single-gene perturbations.

Fixed test genes are withheld. Each strategy starts from the same random seed set, repeatedly acquires
batches from the remaining pool using a model trained only on what it has acquired, and is scored on the
same test genes after each round. Pairs are excluded here, so this addresses the single-gene question only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from biophysical.active_learning.ranking import CandidateSet, select
from biophysical.evaluation.metrics import bootstrap_ci, perturbation_metrics, response_magnitude
from biophysical.models.baselines import BootstrapRidgeEnsemble, FitData, MeanModel
from biophysical.uncertainty.calibration import disagreement


def _score(model, reference: np.ndarray, test: FitData) -> dict[str, float]:
    P = model.predict(test.cond_genes)
    rows = [perturbation_metrics(test.Y[i], P[i], test.mask[i], reference=reference) for i in range(len(P))]
    return {m: float(np.nanmean([r[m] for r in rows])) for m in ["pearson_specific", "skill_vs_mean", "mse", "pearson"]}


def run_simulation(full: FitData, single_idx: np.ndarray, strategies: list[str], n_repeats: int = 10, initial: int = 15, batch: int = 5, rounds: int = 8,
                   test_fraction: float = 0.25, seed: int = 0, alpha: float = 100.0, fixed_idx: np.ndarray | None = None, base_kind: str = "mean") -> pd.DataFrame:
    """`single_idx` is the pool and test population. `fixed_idx` conditions are always in training, as
    when single-gene responses are already measured and pairs are being chosen."""
    rows = []
    fixed = np.array([], dtype=int) if fixed_idx is None else np.asarray(fixed_idx, dtype=int)
    for rep in range(n_repeats):
        rng = np.random.default_rng(seed + rep)
        perm = rng.permutation(single_idx)
        n_test = max(5, int(round(test_fraction * perm.size)))
        test_idx, pool_all = perm[:n_test], perm[n_test:]
        test = full.subset(test_idx)
        start = pool_all[:initial]
        # one reference per repeat, so scores are comparable across strategies and rounds
        reference = MeanModel().fit(full.subset(pool_all)).overall
        for strat in strategies:
            acquired = start.copy()
            remaining = pool_all[initial:].copy()
            for rnd in range(rounds + 1):
                train = full.subset(np.concatenate([fixed, acquired]))
                model = BootstrapRidgeEnsemble(n_members=8, alphas=(alpha,), seed=seed + rep, base_kind=base_kind).fit(train)
                rows.append({"repeat": rep, "strategy": strat, "round": rnd, "n_acquired": int(acquired.size), **_score(model, reference, test)})
                if rnd == rounds or remaining.size == 0:
                    break
                pool = full.subset(remaining)
                members = model.predict_members(pool.cond_genes)
                X = model.scaler.transform(model.featurizer.transform(pool.cond_genes))
                Xm = model.scaler.transform(model.featurizer.transform(train.cond_genes))
                mag = response_magnitude(members.mean(axis=0), pool.mask)
                c = CandidateSet(names=[str(i) for i in remaining], X=X, uncertainty=disagreement(members), magnitude=mag)
                idx, _, _ = select(c, batch, strat, X_measured=Xm, seed=seed + 1000 * rep + rnd)
                acquired = np.concatenate([acquired, remaining[idx]])
                remaining = np.delete(remaining, idx)
    return pd.DataFrame(rows)


def summarise_simulation(df: pd.DataFrame, metric: str = "pearson_specific", baseline: str = "random", n_boot: int = 2000) -> pd.DataFrame:
    """Per-repeat area under the learning curve, compared with the baseline on the same test genes."""
    auc = df.groupby(["repeat", "strategy"])[metric].mean().unstack("strategy")
    final = df[df["round"] == df["round"].max()].set_index(["repeat", "strategy"])[metric].unstack("strategy")
    rows = []
    for s in auc.columns:
        d = (auc[s] - auc[baseline]).to_numpy()
        m, lo, hi = bootstrap_ci(d, n_boot=n_boot)
        rows.append({"strategy": s, "metric": metric, "mean_auc": float(auc[s].mean()), "mean_final": float(final[s].mean()), "auc_diff_vs_baseline": m, "ci_low": lo, "ci_high": hi,
                     "n_repeats_better": int((d > 0).sum()), "n_repeats": int(d.size)})
    return pd.DataFrame(rows).sort_values("mean_auc", ascending=False)


def prioritisation_report(cond_ids: list[str], scores: dict[str, np.ndarray], truth: dict[str, np.ndarray], top_fraction: float = 0.2, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """How many truly top perturbations does each out-of-sample ranking find?

    scores: ranking name -> predicted score per perturbation. truth: objective name -> observed value.
    Random selection has an expected precision equal to top_fraction. Intervals resample perturbations.
    """
    n = len(cond_ids)
    k = max(1, int(round(top_fraction * n)))
    rng = np.random.default_rng(seed)
    rows = []
    for objective, y in truth.items():
        true_top = set(np.argsort(-y, kind="stable")[:k].tolist())
        for name, s in scores.items():
            if objective not in name.split("::")[0] and "::" in name:
                continue
            label = name.split("::")[-1]
            if np.ptp(s) == 0:
                rows.append({"objective": objective, "ranking": label, "n": n, "k": k, "precision_at_k": float("nan"), "enrichment_vs_random": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "spearman": float("nan")})
                continue
            picked = np.argsort(-s, kind="stable")[:k].tolist()
            precision = len(true_top & set(picked)) / k
            boots = []
            for _ in range(n_boot):
                idx = rng.integers(0, n, n)
                yb, sb = y[idx], s[idx]
                tb = set(np.argsort(-yb, kind="stable")[:k].tolist())
                pb = set(np.argsort(-sb, kind="stable")[:k].tolist())
                boots.append(len(tb & pb) / k)
            rows.append({"objective": objective, "ranking": label, "n": n, "k": k, "precision_at_k": precision, "enrichment_vs_random": precision / (k / n), "ci_low": float(np.quantile(boots, 0.025)), "ci_high": float(np.quantile(boots, 0.975)),
                         "spearman": float(pd.Series(y).corr(pd.Series(s), method="spearman"))})
    return pd.DataFrame(rows)
