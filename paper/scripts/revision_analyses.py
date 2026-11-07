"""POST-HOC analyses added in revision, after the confirmatory results were seen. None are confirmatory.

  features <dataset>   Rebuild control-derived features (co-expression embedding, target baseline mean and
                       detection rate) from the TRAINING half of control cells only, so that no feature uses the
                       control cells behind the evaluation targets. Cached in results/paper/revision/.
  ablate               For every included dataset, on the primary unseen-gene folds, fit ridge and the MLP ensemble
                       with each feature set below, and compute standard and magnitude-stratified interval coverage.

Feature sets (gene vector = sum over the condition's targets, plus a pair indicator):
  full                 co-expression embedding + baseline mean + detection rate   (the primary model)
  coexpression_only    co-expression embedding
  baseline_mean_only   target's mean log expression in control cells
  detection_only       target's detection rate in control cells
  baseline_both        baseline mean + detection rate
  full_split0          as 'full', computed from the training half of controls only
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from biophysical.config import get_paths, load_yaml
from biophysical.data.inspect import read_obs_var
from biophysical.data.registry import DatasetRegistry
from biophysical.evaluation.benchmark import make_fit_data, model_factories
from biophysical.evaluation.metrics import perturbation_metrics
from biophysical.models.baselines import MeanModel
from biophysical.models.features import Featurizer
from biophysical.preprocessing.pipeline import apply_cell_filters, coexpression_embedding, load_response_data
from biophysical.preprocessing.pseudobulk import extract_dense_cells
from biophysical.preprocessing.splits import unseen_gene_folds
from biophysical.uncertainty.calibration import GeneIntervalCalibrator

OUT = Path("results/paper/revision")
STUDY = Path("results/study/primary")
N_STRATA = 5


def _dataset_meta(dataset_id: str) -> tuple[dict, Path]:
    d = next(x for x in json.loads((STUDY / "datasets.json").read_text()) if x["dataset_id"] == dataset_id)
    return d, get_paths().processed / dataset_id / d["processed_version"]


def split0_features(dataset_id: str, seed: int = 0, max_cells: int = 8000) -> Path:
    """Co-expression embedding and baseline statistics from training-half control cells only."""
    meta, processed = _dataset_meta(dataset_id)
    data = load_response_data(processed)
    cfg = data.meta["config"]
    entry = DatasetRegistry().get(dataset_id)
    obs, var = read_obs_var(entry.resolved_path())
    keep, _ = apply_cell_filters(obs, cfg["cell_filters"])
    # identical draw to build_groups(): one integer per cell in file order, same seed
    halves = np.random.default_rng(seed).integers(0, 2, size=len(obs))
    ctrl = np.flatnonzero((obs[cfg["perturbation_col"]].astype(object).to_numpy() == cfg["control_label"]) & keep & (halves == 0))
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(ctrl, size=min(max_cells, ctrl.size), replace=False))
    name_to_idx = {g: i for i, g in enumerate(var.index.astype(str))}
    gene_idx = np.array([name_to_idx[g] for g in data.genes])
    t0 = time.time()
    X = extract_dense_cells(entry.resolved_path(), sample, gene_idx, target_sum=cfg["target_sum"])
    emb = coexpression_embedding(X, cfg["embedding_dim"], seed=seed)
    base = np.stack([X.mean(axis=0), (X > 0).mean(axis=0)], axis=1).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"split0_features_{dataset_id}.npz"
    np.savez(path, embedding=emb, baseline=base, n_cells=sample.size, n_split0_controls=ctrl.size, seconds=time.time() - t0)
    print(dataset_id, "split-0 controls:", ctrl.size, "used:", sample.size, f"{time.time() - t0:.0f}s", flush=True)
    return path


def featurizers(data, dataset_id: str) -> dict[str, Featurizer]:
    genes, E, B = list(data.embedding_genes), data.gene_embedding, data.embedding_baseline
    out = {
        "full": Featurizer(genes, {"c": E}, B),
        "coexpression_only": Featurizer(genes, {"c": E}, B, use_baseline=False),
        "baseline_mean_only": Featurizer(genes, {"m": B[:, :1]}, B, use_baseline=False),
        "detection_only": Featurizer(genes, {"d": B[:, 1:]}, B, use_baseline=False),
        "baseline_both": Featurizer(genes, {}, B),
    }
    p = OUT / f"split0_features_{dataset_id}.npz"
    if p.exists():
        z = np.load(p)
        out["full_split0"] = Featurizer(genes, {"c": z["embedding"]}, z["baseline"])
    return out


def stratified_coverage(Yc, Pc, Mc, Rc, Yt, Pt, Mt, Rt, alpha: float = 0.1, top_k: int = 20) -> dict[str, np.ndarray]:
    """Standard per-gene intervals vs intervals whose quantile is calibrated separately within strata of
    predicted response magnitude |prediction - training mean| (ranked within each perturbation). Strata
    use only predictions, so they are available at prediction time."""
    std = GeneIntervalCalibrator(alpha=alpha).fit(Yc, Pc, Mc)
    sigma = std.sigma

    def strata(P, R, M):
        dev = np.where(M, -1.0, np.abs(P - R))
        ranks = dev.argsort(axis=1).argsort(axis=1) / max(dev.shape[1] - 1, 1)
        return np.minimum((ranks * N_STRATA).astype(int), N_STRATA - 1)

    sc, st = strata(Pc, Rc, Mc), strata(Pt, Rt, Mt)
    scores = np.abs(Yc - Pc) / sigma
    q = np.zeros(N_STRATA)
    for s in range(N_STRATA):
        v = scores[(sc == s) & ~Mc]
        n = v.size
        q[s] = np.quantile(v, min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)) if n else std.q
    hw_std = np.broadcast_to(std.half_width(), Yt.shape)
    hw_str = q[st] * sigma
    res = {"cov_std": [], "cov_str": [], "top_std": [], "top_str": [], "width_std_top": [], "width_str_top": []}
    for i in range(len(Yt)):
        keep = ~Mt[i]
        top = np.argsort(-np.where(Mt[i], -1, np.abs(Yt[i] - Rt[i])), kind="stable")[:top_k]
        for key, hw in (("std", hw_std[i]), ("str", hw_str[i])):
            inside = np.abs(Yt[i] - Pt[i]) <= hw
            res[f"cov_{key}"].append(inside[keep].mean())
            res[f"top_{key}"].append(inside[top].mean())
            res[f"width_{key}_top"].append(hw[top].mean())
    return {k: np.array(v) for k, v in res.items()}


def ablate(seed: int = 0) -> pd.DataFrame:
    base_cfg = load_yaml("configs/benchmark_norman.yaml")
    rows = []
    for d in json.loads((STUDY / "datasets.json").read_text()):
        if not d.get("included"):
            continue
        dataset_id = d["dataset_id"]
        _, processed = _dataset_meta(dataset_id)
        data = load_response_data(processed)
        factories = model_factories(base_cfg, "unseen_gene", seed)
        ids = data.conditions["condition"].to_numpy()
        for fname, feat in featurizers(data, dataset_id).items():
            fit = make_fit_data(data, data.Y_fit, feat)
            for fold in unseen_gene_folds(fit.cond_genes, base_cfg["splits"]["n_folds"], base_cfg["splits"]["calibration_fraction"], seed):
                train, calib = fit.subset(fold.train), fit.subset(fold.calib)
                test_genes = [fit.cond_genes[i] for i in fold.test]
                calib_genes = [fit.cond_genes[i] for i in fold.calib]
                mean_model = MeanModel().fit(train)
                Rt, Rc = mean_model.predict(test_genes), mean_model.predict(calib_genes)
                singles = [j for j, i in enumerate(fold.test) if fold.strata[int(i)] == "single_unseen"]
                for mname in ("ridge", "mlp_ensemble"):
                    model = factories[mname]().fit(train, calib)
                    Pt, Pc = model.predict(test_genes), model.predict(calib_genes)
                    cov = stratified_coverage(data.Y_eval[fold.calib], Pc, data.self_mask[fold.calib], Rc,
                                              data.Y_eval[fold.test], Pt, data.self_mask[fold.test], Rt) if fname == "full" else None
                    for j in singles:
                        i = fold.test[j]
                        m = perturbation_metrics(data.Y_eval[i], Pt[j], data.self_mask[i], reference=Rt[j])
                        row = {"dataset_id": dataset_id, "features": fname, "model": mname, "fold": fold.fold, "condition": ids[i],
                               "mse": m["mse"], "skill_vs_mean": m["skill_vs_mean"], "pearson_specific": m["pearson_specific"]}
                        if cov is not None:
                            row.update({k: float(v[j]) for k, v in cov.items()})
                        rows.append(row)
            print(dataset_id, fname, "done", flush=True)
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "feature_ablation.csv", index=False)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "features":
        for ds in sys.argv[2:]:
            split0_features(ds)
    elif sys.argv[1] == "ablate":
        t = ablate()
        print(t.groupby(["dataset_id", "features", "model"])[["skill_vs_mean", "pearson_specific"]].mean().round(4).to_string())
