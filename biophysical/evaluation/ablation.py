"""Ablation experiments. Each compares models under one changed factor on the unseen_gene task."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biophysical.evaluation.benchmark import per_condition
from biophysical.logging_utils import get_logger
from biophysical.models.baselines import GeneLoadingRidge, MeanModel, RidgeModel
from biophysical.models.features import Featurizer, default_featurizer
from biophysical.models.mlp import MLPConfig, MLPEnsemble
from biophysical.preprocessing.pipeline import load_response_data, preprocess_dataset
from biophysical.preprocessing.splits import unseen_gene_folds
from biophysical.preprocessing.targets import ResponseData

log = get_logger(__name__)
KEY_METRICS = ["pearson_specific", "skill_vs_mean", "pearson", "mse"]


def _aggregate(frame: pd.DataFrame, group: str, name: str) -> pd.DataFrame:
    d = per_condition(frame)
    out = d.groupby(["task", "stratum", "model"], as_index=False, observed=True)[KEY_METRICS].mean()
    out.insert(0, "variant", name)
    out.insert(0, "ablation", group)
    return out


def _simple_models(cfg: dict[str, Any], seed: int, with_mlp: bool = False) -> dict[str, Any]:
    alphas = tuple(cfg["models"]["ridge_alpha"])
    out: dict[str, Any] = {"mean": MeanModel, "ridge": lambda: RidgeModel(alphas), "gene_loading_ridge": lambda: GeneLoadingRidge(alphas=alphas)}
    if with_mlp:
        out["mlp_ensemble"] = lambda: MLPEnsemble(MLPConfig(**{**cfg["models"]["mlp"], "seed": seed}))
    return out


def _run_variant(data: ResponseData, cfg: dict[str, Any], featurizer: Featurizer, seed: int, Y_train: np.ndarray | None = None, Y_test: np.ndarray | None = None, models=None) -> pd.DataFrame:
    from biophysical.evaluation.benchmark import evaluate_folds

    folds = unseen_gene_folds(data.condition_genes(), cfg["splits"]["n_folds"], cfg["splits"]["calibration_fraction"], seed)
    res = evaluate_folds(data, folds, models or _simple_models(cfg, seed), featurizer, data.Y_fit if Y_train is None else Y_train, data.Y_eval if Y_test is None else Y_test)
    return res["per_perturbation"]


def representation_variants(data: ResponseData, seed: int = 0, extra: dict[str, dict[str, np.ndarray]] | None = None) -> dict[str, Featurizer]:
    """coexpression, baseline-only features, and a permuted embedding as a negative control."""
    rng = np.random.default_rng(seed)
    variants = {
        "coexpression": default_featurizer(data),
        "baseline_only": Featurizer(list(data.embedding_genes), {}, data.embedding_baseline),
        "permuted_coexpression": Featurizer(list(data.embedding_genes), {"permuted": data.gene_embedding[rng.permutation(len(data.embedding_genes))]}, data.embedding_baseline),
    }
    for name, matrices in (extra or {}).items():
        variants[name] = Featurizer(list(data.embedding_genes), matrices, data.embedding_baseline)
    return variants


def run_ablations(cfg: dict[str, Any], config_path: str | Path, data: ResponseData, run_dir: Path, seed: int) -> dict[str, Any]:
    rows: list[pd.DataFrame] = []

    # 1. feature representation (includes a negative control), with the trainable model
    for name, feat in representation_variants(data, seed).items():
        log.info("ablation representation: %s", name)
        rows.append(_aggregate(_run_variant(data, cfg, feat, seed, models={k: v for k, v in _simple_models(cfg, seed, with_mlp=True).items() if k != "gene_loading_ridge"}), "representation", name))

    # 2. shared versus disjoint control cells between training and evaluation targets
    feat = default_featurizer(data)
    rows.append(_aggregate(_run_variant(data, cfg, feat, seed), "control_cells", "disjoint"))
    rows.append(_aggregate(_run_variant(data, cfg, feat, seed, Y_train=data.Y_eval), "control_cells", "shared"))

    # 3. alternative target: log of mean expression instead of mean of log expression
    rows.append(_aggregate(_run_variant(data, cfg, feat, seed, Y_train=data.Y_logmean, Y_test=data.Y_logmean), "target", "log_of_pseudobulk_mean_shared_controls"))
    rows.append(_aggregate(_run_variant(data, cfg, feat, seed, Y_train=data.Y_eval, Y_test=data.Y_eval), "target", "pseudobulk_mean_of_log_shared_controls"))

    # 4. preprocessing variants. Absolute scores differ with the gene set, so compare the
    # models within a variant; variants are not directly ranked against each other.
    for name, overrides in cfg.get("ablations", {}).get("preprocessing", {}).items():
        log.info("ablation preprocessing: %s", name)
        path = preprocess_dataset(cfg["dataset_id"], config_path=config_path, overrides=overrides)
        variant = load_response_data(path)
        rows.append(_aggregate(_run_variant(variant, cfg, default_featurizer(variant), seed), "preprocessing", name))
    # 5. stability across fold partitions, on both tasks
    from biophysical.evaluation.benchmark import evaluate_folds, model_factories
    from biophysical.preprocessing.splits import unseen_combo_folds

    for s in cfg.get("ablations", {}).get("split_seeds", []):
        genes = data.condition_genes()
        n, frac = cfg["splits"]["n_folds"], cfg["splits"]["calibration_fraction"]
        for task, folds, names in (("unseen_gene", unseen_gene_folds(genes, n, frac, s), ["mean", "ridge", "gene_loading_ridge", "mlp_ensemble"]),
                                   ("unseen_combo", unseen_combo_folds(genes, n, frac, s), ["mean", "additive", "scaled_additive", "additive_plus_mean_residual", "ridge_on_additive", "mlp_on_additive"])):
            factories = model_factories(cfg, task, seed)
            res = evaluate_folds(data, folds, {m: factories[m] for m in names}, feat, data.Y_fit, data.Y_eval)
            rows.append(_aggregate(res["per_perturbation"], "split_seed", f"seed_{s}"))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(run_dir / "ablations.csv", index=False)
    return {"ablation_groups": sorted(out["ablation"].unique().tolist())}
