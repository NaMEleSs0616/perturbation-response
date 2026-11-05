"""Reproducible benchmark: python -m biophysical.cli benchmark run --config configs/benchmark_norman.yaml

Every model on a task and fold uses the same training perturbations, calibration perturbations, test
perturbations and response target. Training uses Y_fit and evaluation uses Y_eval (disjoint controls).
"""
from __future__ import annotations

import pickle
import resource
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from biophysical.active_learning.simulate import prioritisation_report, run_simulation, summarise_simulation
from biophysical.config import dump_yaml, get_paths, load_yaml
from biophysical.evaluation.metrics import _corr, bootstrap_ci, metrics_frame, paired_comparison, perturbation_ranking, response_magnitude
from biophysical.logging_utils import add_json_file_handler, get_logger
from biophysical.models.baselines import (AdditiveModel, AdditivePlusMeanResidual, BootstrapRidgeEnsemble, FitData, GeneLoadingRidge, KNNModel, MeanModel, RidgeModel, ScaledAdditive, SinglesOnly, ZeroModel)
from biophysical.models.features import Featurizer, default_featurizer
from biophysical.models.mlp import MLPConfig, MLPEnsemble
from biophysical.preprocessing.pipeline import load_response_data, preprocess_dataset
from biophysical.preprocessing.splits import Fold, assert_no_leakage, folds_table, unseen_combo_folds, unseen_gene_folds
from biophysical.preprocessing.targets import ResponseData
from biophysical.provenance import array_digest, environment_info, json_digest, new_run_id, write_json
from biophysical.uncertainty.calibration import GeneIntervalCalibrator, disagreement, error_uncertainty_relation

log = get_logger(__name__)
REPORT_METRICS = ["skill_vs_mean", "pearson_specific", "pearson_specific_top", "pearson_interaction", "pearson_interaction_top", "skill_vs_additive", "pearson", "pearson_top", "spearman", "mae", "mse", "mae_top", "sign_acc_top", "topk_recall", "r2_vs_zero"]
ModelFactory = Callable[[], Any]


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3


def make_fit_data(data: ResponseData, Y: np.ndarray, featurizer: Featurizer) -> FitData:
    return FitData(data.condition_genes(), Y, data.self_mask, featurizer, {g: i for i, g in enumerate(data.genes)})


def model_factories(cfg: dict[str, Any], task: str, seed: int) -> dict[str, ModelFactory]:
    mcfg = cfg.get("models", {})
    alphas = tuple(mcfg.get("ridge_alpha", (0.1, 1, 10, 100, 1000, 10000)))
    ks = tuple(mcfg.get("knn_k", (1, 3, 5, 10)))
    mlp = {**mcfg.get("mlp", {}), "seed": seed}
    out: dict[str, ModelFactory] = {
        "zero": ZeroModel,
        "mean": MeanModel,
        "knn": lambda: KNNModel(ks),
        "ridge": lambda: RidgeModel(alphas),
        "gene_loading_ridge": lambda: GeneLoadingRidge(alphas=alphas),
        "bootstrap_ridge": lambda: BootstrapRidgeEnsemble(alphas=alphas, seed=seed),
        "mlp_ensemble": lambda: MLPEnsemble(MLPConfig(**mlp)),
        "ridge_singles_only": lambda: SinglesOnly(RidgeModel(alphas), "ridge_singles_only"),
        "gene_loading_singles_only": lambda: SinglesOnly(GeneLoadingRidge(alphas=alphas), "gene_loading_singles_only"),
    }
    if task == "unseen_combo":
        out.update({
            "additive": AdditiveModel,
            "scaled_additive": ScaledAdditive,
            "additive_plus_mean_residual": AdditivePlusMeanResidual,
            "knn_on_additive": lambda: KNNModel(ks, base_kind="additive", name="knn_on_additive"),
            "ridge_on_additive": lambda: RidgeModel(alphas, base_kind="additive", name="ridge_on_additive"),
            "bootstrap_ridge_on_additive": lambda: BootstrapRidgeEnsemble(alphas=alphas, base_kind="additive", seed=seed, name="bootstrap_ridge_on_additive"),
            "mlp_on_additive": lambda: MLPEnsemble(MLPConfig(**mlp), base_kind="additive", name="mlp_on_additive"),
        })
    return out


def evaluate_folds(data: ResponseData, folds: list[Fold], factories: dict[str, ModelFactory], featurizer: Featurizer, Y_train: np.ndarray, Y_test: np.ndarray,
                   top_k: int = 20, alpha: float = 0.1) -> dict[str, Any]:
    """Fit all models on every fold. Returns per-perturbation metrics, OOF predictions and runtimes."""
    cond_ids = data.conditions["condition"].tolist()
    fit = make_fit_data(data, Y_train, featurizer)
    rows, runtime, oof, ceilings = [], [], {}, []
    for fold in folds:
        assert_no_leakage(fold, fit.cond_genes)
        train, calib = fit.subset(fold.train), fit.subset(fold.calib)
        test_genes = [fit.cond_genes[i] for i in fold.test]
        calib_genes = [fit.cond_genes[i] for i in fold.calib]
        test_ids = [cond_ids[i] for i in fold.test]
        Y_te, M_te = Y_test[fold.test], data.self_mask[fold.test]
        Y_cal, M_cal = Y_test[fold.calib], data.self_mask[fold.calib]
        reference = MeanModel().fit(train).predict(test_genes)
        strata = [fold.strata[int(i)] for i in fold.test]
        additive_te = AdditiveModel().fit(train).predict(test_genes) if fold.task == "unseen_combo" else None
        ceilings.append(_ceiling_rows(data, fold, reference, test_ids, strata))
        for name, make in factories.items():
            t0 = time.perf_counter()
            model = make().fit(train, calib)
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            P = model.predict(test_genes)
            pred_s = time.perf_counter() - t0
            frame = metrics_frame(test_ids, Y_te, P, M_te, top_k, reference=reference, task=fold.task, fold=fold.fold, model=name)
            frame["stratum"] = strata
            frame["n_cells"] = data.conditions["n_cells"].to_numpy()[fold.test]
            cal = GeneIntervalCalibrator(alpha=alpha).fit(Y_cal, model.predict(calib_genes), M_cal)
            frame["coverage"] = cal.coverage(Y_te, P, M_te)
            frame["coverage_top"] = _coverage_top(cal, Y_te, P, M_te, reference, top_k)
            if fold.task == "unseen_combo":
                _add_interaction_metrics(frame, Y_te, P, M_te, additive_te, top_k)
            frame["interval_half_width"] = float(np.median(cal.half_width()))
            frame["rmse"] = np.sqrt(frame["mse"])
            if hasattr(model, "predict_members"):
                frame["disagreement"] = disagreement(model.predict_members(test_genes))
            rows.append(frame)
            runtime.append({"task": fold.task, "fold": fold.fold, "model": name, "fit_seconds": fit_s, "predict_seconds": pred_s, "n_train": len(train.cond_genes), "selected": _hyperparams(model)})
            oof.setdefault(name, {}).setdefault(fold.fold, {"test": fold.test, "P": P})
        log.info("%s fold %d done (%d train, %d calib, %d test)", fold.task, fold.fold, len(fold.train), len(fold.calib), len(fold.test))
    return {"per_perturbation": pd.concat(rows, ignore_index=True), "runtime": pd.DataFrame(runtime), "oof": oof, "ceilings": pd.concat(ceilings, ignore_index=True)}


def _coverage_top(cal: GeneIntervalCalibrator, Y: np.ndarray, P: np.ndarray, M: np.ndarray, reference: np.ndarray, top_k: int) -> np.ndarray:
    """Interval coverage on each perturbation's top genes by observed deviation from the training mean.
    Overall coverage is dominated by genes that do not respond; this is the coverage that matters."""
    out = np.full(len(Y), np.nan)
    hw = cal.half_width()
    for i in range(len(Y)):
        dev = np.where(M[i], -1.0, np.abs(Y[i] - reference[i]))
        top = np.argsort(-dev, kind="stable")[:top_k]
        out[i] = float((np.abs(Y[i][top] - P[i][top]) <= hw[top]).mean())
    return out


def _add_interaction_metrics(frame: pd.DataFrame, Y: np.ndarray, P: np.ndarray, M: np.ndarray, additive: np.ndarray, top_k: int) -> None:
    """Scores on the non-additive component (pair - sum of singles), which is the part of a pair's
    response that the singles do not already explain."""
    r, r_top, skill = [], [], []
    for i in range(len(Y)):
        keep = ~M[i]
        dy, dp = (Y[i] - additive[i])[keep], (P[i] - additive[i])[keep]
        top = np.argsort(-np.abs(dy), kind="stable")[:top_k]
        sse_ref = float((dy ** 2).sum())
        r.append(_corr(dy, dp))
        r_top.append(_corr(dy[top], dp[top]))
        skill.append(1 - float(((dy - dp) ** 2).sum()) / sse_ref if sse_ref > 0 else float("nan"))
    frame["pearson_interaction"], frame["pearson_interaction_top"], frame["skill_vs_additive"] = r, r_top, skill


def _hyperparams(model) -> str:
    inner = getattr(model, "inner", model)
    keys = [k for k in ("alpha", "k", "scale") if hasattr(inner, k)]
    if isinstance(inner, MLPEnsemble):
        return f"epochs={[h['epochs'] for h in inner.history]}"
    return ",".join(f"{k}={getattr(inner, k)}" for k in keys)


def _ceiling_rows(data: ResponseData, fold: Fold, reference: np.ndarray, ids: list[str], strata: list[str]) -> pd.DataFrame:
    """Reproducibility of each test perturbation, for raw and perturbation-specific responses."""
    rows = []
    cond_lookup = {frozenset(g): k for k, g in enumerate(data.condition_genes())}
    for j, i in enumerate(fold.test):
        keep = ~data.self_mask[i]

        def rel(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
            ok = keep & np.isfinite(a) & np.isfinite(b)
            r = _corr(a[ok], b[ok])
            if not np.isfinite(r):
                return float("nan"), float("nan")
            sb = 2 * r / (1 + r) if r > 0 else 0.0
            return r, float(np.sqrt(sb))

        ha, hb, ba, bb = data.Y_half_a[i], data.Y_half_b[i], data.Y_batch_a[i], data.Y_batch_b[i]
        r_raw, max_raw = rel(ha, hb)
        r_spec, max_spec = rel(ha - reference[j], hb - reference[j])
        r_batch, max_batch = rel(ba - reference[j], bb - reference[j])
        row = {"task": fold.task, "fold": fold.fold, "condition": ids[j], "stratum": strata[j], "half_r": r_raw, "max_r": max_raw, "specific_half_r": r_spec, "specific_max_r": max_spec,
               "specific_batch_r": r_batch, "specific_batch_max_r": max_batch}
        if fold.task == "unseen_combo":
            # interaction estimated separately from disjoint cell halves of the pair and of both singles
            a, b = cond_lookup[frozenset(data.condition_genes()[i])], [cond_lookup[frozenset([g])] for g in data.condition_genes()[i]]
            ia = data.Y_half_a[a] - data.Y_half_a[b].sum(axis=0)
            ib = data.Y_half_b[a] - data.Y_half_b[b].sum(axis=0)
            row["interaction_half_r"], row["interaction_max_r"] = rel(ia, ib)
        rows.append(row)
    return pd.DataFrame(rows)


def per_condition(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per model, stratum and condition. A pair held out in two folds is averaged."""
    num = frame.select_dtypes("number").columns.difference(["fold"])
    return frame.groupby(["task", "stratum", "model", "condition"], as_index=False, observed=True)[list(num)].mean()


def summarise(frame: pd.DataFrame, strong: set[tuple[str, str]] | None = None, n_boot: int = 2000) -> pd.DataFrame:
    rows = []
    d = per_condition(frame)
    groups = [("all", d)]
    if strong is not None:
        mask = [(c, s) in strong for c, s in zip(d["condition"], d["stratum"])]
        groups.append(("reproducible", d[mask]))
    for subset, sub in groups:
        for (task, stratum, model), g in sub.groupby(["task", "stratum", "model"], observed=True):
            for m in REPORT_METRICS:
                if m not in g:
                    continue
                mean, lo, hi = bootstrap_ci(g[m].to_numpy(), n_boot=n_boot)
                rows.append({"task": task, "stratum": stratum, "subset": subset, "model": model, "metric": m, "mean": mean, "ci_low": lo, "ci_high": hi, "median": float(g[m].dropna().median()) if g[m].notna().any() else float("nan"),
                             "n_perturbations": int(g[m].notna().sum())})
    return pd.DataFrame(rows)


def compare(frame: pd.DataFrame, references: dict[str, list[str]], n_boot: int = 2000) -> pd.DataFrame:
    """Paired per-perturbation comparison of each model with each reference model."""
    d = per_condition(frame)
    rows = []
    for (task, stratum), g in d.groupby(["task", "stratum"], observed=True):
        wide = {m: gg.set_index("condition") for m, gg in g.groupby("model", observed=True)}
        for ref in references.get(task, []):
            if ref not in wide:
                continue
            for model, t in wide.items():
                if model == ref:
                    continue
                for metric in ["mse", "mae", "pearson", "pearson_top", "skill_vs_mean", "pearson_specific", "skill_vs_additive", "pearson_interaction"]:
                    if metric not in t:
                        continue
                    a, b = t[metric], wide[ref][metric].reindex(t.index)
                    rows.append({"task": task, "stratum": stratum, "model": model, "reference": ref, "metric": metric, **paired_comparison(a.to_numpy(), b.to_numpy(), n_boot=n_boot)})
    return pd.DataFrame(rows)


def magnitude_ranking(data: ResponseData, results: dict[str, Any], Y_test: np.ndarray) -> pd.DataFrame:
    """Spearman across perturbations between predicted and observed response magnitude."""
    rows = []
    f = results["per_perturbation"]
    idx = data.index()
    for (task, stratum), g in f.groupby(["task", "stratum"], observed=True):
        for model, per_fold in results["oof"].items():
            collected, truth, masks = [], [], []
            for fold, rec in per_fold.items():
                conds = set(g[(g["fold"] == fold) & (g["model"] == model)]["condition"])
                for pos, i in enumerate(rec["test"]):
                    if data.conditions["condition"].iloc[i] in conds:
                        collected.append(rec["P"][pos])
                        truth.append(Y_test[i])
                        masks.append(data.self_mask[i])
            if len(collected) >= 5:
                rows.append({"task": task, "stratum": stratum, "model": model, "spearman_magnitude": perturbation_ranking(np.stack(truth), np.stack(collected), np.stack(masks)), "n": len(collected)})
    return pd.DataFrame(rows)


def uncertainty_report(frame: pd.DataFrame, target: float) -> pd.DataFrame:
    rows = []
    for (task, stratum, model), g in frame.groupby(["task", "stratum", "model"], observed=True):
        row = {"task": task, "stratum": stratum, "model": model, "target_coverage": target, "mean_coverage": float(g["coverage"].mean()), "p10_coverage": float(g["coverage"].quantile(0.1)),
               "median_interval_half_width": float(g["interval_half_width"].median()), "mean_coverage_top_genes": float(g["coverage_top"].mean())}
        if "disagreement" in g and g["disagreement"].notna().any():
            rel = error_uncertainty_relation(g["rmse"].to_numpy(), g["disagreement"].to_numpy())
            row.update({"spearman_disagreement_vs_rmse": rel["spearman"], "p": rel["p"], "n": rel["n"]})
        rows.append(row)
    return pd.DataFrame(rows)


def train_serving_bundle(data: ResponseData, cfg: dict[str, Any], results: dict[str, dict[str, Any]], featurizer: Featurizer, seed: int, serving: dict[str, str], alpha: float) -> dict[str, Any]:
    """Refit serving models on all measured conditions, with calibrators from out-of-fold predictions."""
    all_fit = make_fit_data(data, data.Y_all, featurizer)
    bundle: dict[str, Any] = {"models": {}, "calibrators": {}, "serving": serving}
    for context, (task, model_name, strata) in serving.items():
        factories = model_factories(cfg, task, seed)
        model = factories[model_name]().fit(all_fit)
        P, Y, M = [], [], []
        oof = results[task]["oof"][model_name]
        f = results[task]["per_perturbation"]
        for fold, rec in oof.items():
            conds = set(f[(f["fold"] == fold) & (f["model"] == model_name) & (f["stratum"].isin(strata))]["condition"])
            for pos, i in enumerate(rec["test"]):
                if data.conditions["condition"].iloc[i] in conds:
                    P.append(rec["P"][pos])
                    Y.append(data.Y_eval[i])
                    M.append(data.self_mask[i])
        bundle["models"][context] = model
        bundle["calibrators"][context] = GeneIntervalCalibrator(alpha=alpha).fit(np.stack(Y), np.stack(P), np.stack(M)).state()
        log.info("serving model for %s: %s, calibrated on %d out-of-fold predictions", context, model_name, len(P))
    return bundle


def pair_prioritisation(data: ResponseData, results: dict[str, Any], top_fraction: float, n_boot: int) -> pd.DataFrame:
    """Out-of-fold ranking of pairs and unseen singles for two objectives: largest response magnitude,
    and largest non-additive component. Each ranking's scores come from models that never saw those conditions."""
    rows = []
    # pairs with measured singles
    oof = results["unseen_combo"]["oof"]
    test = np.concatenate([rec["test"] for rec in oof["additive"].values()])
    order = np.argsort(test)
    test = test[order]
    ids = data.conditions["condition"].iloc[test].tolist()
    M = data.self_mask[test]
    collect = lambda name: np.concatenate([rec["P"] for rec in oof[name].values()])[order]
    additive = collect("additive")
    Y = data.Y_eval[test]
    truth_mag = response_magnitude(Y, M)
    truth_int = response_magnitude(Y - additive, M)
    scores: dict[str, np.ndarray] = {}
    for name in oof:
        P = collect(name)
        scores[f"magnitude::{name}"] = response_magnitude(P, M)
        if name not in ("additive", "zero"):
            scores[f"interaction::{name}"] = response_magnitude(P - additive, M)
    # a pair's weaker interaction may partly reflect weak singles; this ranks by summed single magnitude
    scores["interaction::additive_magnitude"] = response_magnitude(additive, M)
    r = prioritisation_report(ids, scores, {"magnitude": truth_mag, "interaction": truth_int}, top_fraction, n_boot)
    r.insert(0, "population", "pairs_with_measured_singles")
    rows.append(r)
    # unseen singles
    oof = results["unseen_gene"]["oof"]
    f = results["unseen_gene"]["per_perturbation"]
    singles = set(f[f["stratum"] == "single_unseen"]["condition"])
    positions = {}
    for name, per in oof.items():
        if name in ("zero",):
            continue
        P_list, I_list = [], []
        for rec in per.values():
            for pos, i in enumerate(rec["test"]):
                if data.conditions["condition"].iloc[i] in singles:
                    P_list.append(rec["P"][pos])
                    I_list.append(i)
        positions[name] = (np.stack(P_list), np.array(I_list))
    any_name = next(iter(positions))
    idx = positions[any_name][1]
    order = np.argsort(idx)
    idx = idx[order]
    M = data.self_mask[idx]
    truth = {"magnitude": response_magnitude(data.Y_eval[idx], M)}
    scores = {f"magnitude::{n}": response_magnitude(P[np.argsort(I)], M) for n, (P, I) in positions.items()}
    r = prioritisation_report(data.conditions["condition"].iloc[idx].tolist(), scores, truth, top_fraction, n_boot)
    r.insert(0, "population", "unseen_singles")
    rows.append(r)
    return pd.concat(rows, ignore_index=True)


def serving_evidence(frame: pd.DataFrame, serving: dict[str, Any], n_boot: int = 2000) -> dict[str, dict[str, Any]]:
    """For each serving context: does the served model beat its baseline on held-out perturbations?"""
    d = per_condition(frame)
    out = {}
    for context, (task, model_name, strata) in serving.items():
        baseline = "scaled_additive" if task == "unseen_combo" else "mean"
        g = d[(d["task"] == task) & d["stratum"].isin(strata)]
        a = g[g["model"] == model_name].set_index(["stratum", "condition"])["mse"]
        b = g[g["model"] == baseline].set_index(["stratum", "condition"])["mse"].reindex(a.index)
        c = paired_comparison(a.to_numpy(), b.to_numpy(), n_boot=n_boot)
        verdict = "better_than_baseline" if c["ci_high"] < 0 else ("worse_than_baseline" if c["ci_low"] > 0 else "not_distinguishable_from_baseline")
        served = g[g["model"] == model_name]
        rel = error_uncertainty_relation(served["rmse"].to_numpy(), served["disagreement"].to_numpy()) if "disagreement" in served else {"spearman": float("nan")}
        out[context] = {"task": task, "model": model_name, "strata": strata, "baseline": baseline, "metric": "mse (lower is better); paired, model minus baseline", **c, "verdict": verdict,
                        "mean_skill_vs_mean": float(served["skill_vs_mean"].mean()), "empirical_interval_coverage": float(served["coverage"].mean()), "empirical_interval_coverage_top_genes": float(served["coverage_top"].mean()),
                        "spearman_disagreement_vs_rmse": rel["spearman"]}
    return out


def run_benchmark(config_path: str | Path, quick: bool = False, out_dir: str | Path | None = None) -> Path:
    cfg = load_yaml(config_path)
    if quick:
        cfg = _quick_config(cfg)
    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(cfg.get("torch_threads", 2)))
    dataset_id = cfg["dataset_id"]
    paths = get_paths()
    run_id = new_run_id("quick" if quick else "bench")
    run_dir = Path(out_dir) if out_dir else paths.results / dataset_id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    handler = add_json_file_handler(run_dir / "run.log.jsonl")
    t_start = time.time()
    try:
        processed = preprocess_dataset(dataset_id, config_path=config_path) if not cfg.get("_processed_path") else Path(cfg["_processed_path"])
        data = load_response_data(processed)
        featurizer = default_featurizer(data)
        ev = cfg.get("evaluation", {})
        top_k, n_boot, alpha = ev.get("top_k", 20), ev.get("n_bootstrap", 2000), ev.get("coverage_alpha", 0.1)
        n_folds, cal_frac = cfg["splits"]["n_folds"], cfg["splits"]["calibration_fraction"]

        genes_all = data.condition_genes()
        tasks = {"unseen_gene": unseen_gene_folds(genes_all, n_folds, cal_frac, seed), "unseen_combo": unseen_combo_folds(genes_all, n_folds, cal_frac, seed)}
        pd.concat([folds_table(f, data.conditions) for f in tasks.values()], ignore_index=True).to_csv(run_dir / "split_assignments.csv", index=False)

        results = {task: evaluate_folds(data, folds, model_factories(cfg, task, seed), featurizer, data.Y_fit, data.Y_eval, top_k, alpha) for task, folds in tasks.items()}
        frame = pd.concat([r["per_perturbation"] for r in results.values()], ignore_index=True)
        ceilings = pd.concat([r["ceilings"] for r in results.values()], ignore_index=True)
        frame.to_csv(run_dir / "per_perturbation_metrics.csv", index=False)
        ceilings.to_csv(run_dir / "noise_ceiling.csv", index=False)
        ceil_cond = ceilings.groupby(["stratum", "condition"], as_index=False, observed=True).mean(numeric_only=True)
        strong = {(c, s) for c, s, r in zip(ceil_cond["condition"], ceil_cond["stratum"], ceil_cond["specific_batch_r"]) if r >= 0.5}
        summary = summarise(frame, strong, n_boot)
        summary.to_csv(run_dir / "summary.csv", index=False)
        references = {"unseen_gene": ["mean", "ridge"], "unseen_combo": ["mean", "additive", "scaled_additive", "additive_plus_mean_residual", "ridge_on_additive"]}
        compare(frame, references, n_boot).to_csv(run_dir / "paired_comparisons.csv", index=False)
        ranking = pd.concat([magnitude_ranking(data, r, data.Y_eval) for r in results.values()], ignore_index=True)
        ranking.to_csv(run_dir / "perturbation_magnitude_ranking.csv", index=False)
        uncertainty_report(frame, 1 - alpha).to_csv(run_dir / "uncertainty.csv", index=False)
        runtime = pd.concat([r["runtime"] for r in results.values()], ignore_index=True)
        runtime.to_csv(run_dir / "runtime.csv", index=False)

        extras: dict[str, Any] = {}
        if cfg.get("ablations", {}).get("enabled", False):
            from biophysical.evaluation.ablation import run_ablations

            extras["ablations"] = run_ablations(cfg, config_path, data, run_dir, seed)
        if cfg.get("active_learning", {}).get("enabled", True):
            al = cfg.get("active_learning", {})
            fit_eval = FitData(genes_all, data.Y_eval, data.self_mask, featurizer, {g: i for i, g in enumerate(data.genes)})
            singles = np.flatnonzero(data.conditions["n_perts"].to_numpy() == 1)
            sim = run_simulation(fit_eval, singles, al.get("strategies", ["random", "uncertainty", "diversity", "hybrid"]), al.get("n_repeats", 10), al.get("initial", 15), al.get("batch", 5), al.get("rounds", 8), seed=seed)
            pairs = np.flatnonzero(data.conditions["n_perts"].to_numpy() == 2)
            sim["pool"] = "singles"
            sim2 = run_simulation(fit_eval, pairs, al.get("strategies", ["random", "uncertainty", "diversity", "hybrid"]), al.get("n_repeats", 10), al.get("initial_pairs", 10), al.get("batch_pairs", 10), al.get("rounds", 8), seed=seed, fixed_idx=singles, base_kind="additive")
            sim2["pool"] = "pairs_with_singles_known"
            pd.concat([sim, sim2], ignore_index=True).to_csv(run_dir / "acquisition_curves.csv", index=False)
            acq = []
            for pool, d in (("singles", sim), ("pairs_with_singles_known", sim2)):
                for m in ["pearson_specific", "skill_vs_mean", "mse"]:
                    t = summarise_simulation(d, m, n_boot=n_boot)
                    t.insert(0, "pool", pool)
                    acq.append(t)
            pd.concat(acq).to_csv(run_dir / "acquisition_summary.csv", index=False)
            pair_prioritisation(data, results, al.get("top_fraction", 0.2), n_boot).to_csv(run_dir / "prioritisation.csv", index=False)

        serving = cfg.get("serving", {
            "unseen_single": ["unseen_gene", "mlp_ensemble", ["single_unseen"]],
            "pair_with_measured_singles": ["unseen_combo", "mlp_on_additive", ["combo_seen_singles"]],
            "pair_with_unmeasured_single": ["unseen_gene", "mlp_ensemble", ["combo_one_unseen", "combo_two_unseen"]],
        })
        bundle = train_serving_bundle(data, cfg, results, featurizer, seed, serving, alpha)
        bundle["evidence"] = serving_evidence(frame, serving, n_boot)
        write_json(bundle["evidence"], run_dir / "serving_evidence.json")
        bundle["model_version"] = f"{dataset_id}-{data.meta['version']}-{run_dir.name}"
        with open(run_dir / "serving_bundle.pkl", "wb") as fh:
            pickle.dump(bundle, fh)

        record = {
            "run_id": run_dir.name, "quick": quick, "config_path": str(config_path), "config": cfg, "config_digest": json_digest(cfg), "seed": seed,
            "dataset_id": dataset_id, "processed_path": str(processed), "processed_version": data.meta["version"], "source": data.meta["source"],
            "target_digests": {"Y_fit": array_digest(data.Y_fit), "Y_eval": array_digest(data.Y_eval)}, "environment": environment_info(),
            "seconds": round(time.time() - t_start, 1), "peak_rss_mb": round(peak_rss_mb(), 1), "n_reproducible_specific": len(strong), **extras,
        }
        write_json(record, run_dir / "run.json")
        dump_yaml(cfg, run_dir / "config.yaml")
        if not quick or not (paths.results / "latest_run.txt").exists():
            (paths.results / "latest_run.txt").write_text(str(run_dir))
        log.info("benchmark finished in %.0fs, peak RSS %.0f MB: %s", time.time() - t_start, peak_rss_mb(), run_dir)
        return run_dir
    finally:
        import logging

        logging.getLogger("biophysical").removeHandler(handler)
        handler.close()


def _quick_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Smoke-test settings. Results from a quick run are not benchmark evidence."""
    cfg = {**cfg, "splits": {**cfg["splits"], "n_folds": 2}, "evaluation": {**cfg.get("evaluation", {}), "n_bootstrap": 100}, "ablations": {"enabled": False}}
    models = dict(cfg.get("models", {}))
    models["mlp"] = {**models.get("mlp", {}), "n_members": 2, "max_epochs": 20, "patience": 5}
    cfg["models"] = models
    cfg["active_learning"] = {**cfg.get("active_learning", {}), "n_repeats": 1, "rounds": 1, "strategies": ["random", "hybrid"]}
    return cfg
