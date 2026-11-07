"""Multi-dataset study that implements docs/analysis_plan.md.

    python -m biophysical.cli study run --config configs/study.yaml

Per dataset: inclusion checks, preprocessing under shared rules, the unseen_gene task with fixed models
and representations, fold-seed robustness, reproducibility limits, ranking and uncertainty summaries.
Across datasets: Holm-corrected confirmatory tests (C1-C3) and the exploratory analyses (E1, E2).
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from biophysical.active_learning.simulate import prioritisation_report
from biophysical.config import get_paths, load_yaml
from biophysical.data.inspect import read_obs_var
from biophysical.data.registry import DatasetRegistry
from biophysical.evaluation.benchmark import evaluate_folds, make_fit_data, model_factories, per_condition
from biophysical.evaluation.metrics import bootstrap_ci, paired_comparison, perturbation_metrics, response_magnitude
from biophysical.logging_utils import add_json_file_handler, get_logger
from biophysical.models.baselines import GeneLoadingRidge, MeanModel, RidgeModel
from biophysical.models.features import Featurizer, default_featurizer
from biophysical.preprocessing.pipeline import load_response_data, parse_condition, preprocess_dataset, resolve_target_genes
from biophysical.preprocessing.splits import unseen_gene_folds
from biophysical.preprocessing.targets import ResponseData
from biophysical.provenance import environment_info, new_run_id, write_json
from biophysical.uncertainty.calibration import error_uncertainty_relation

log = get_logger(__name__)
REPLICATE_CANDIDATES = ("gemgroup", "batch", "library")
FILTER_CANDIDATES = {"good_coverage": True, "number_of_cells": 1}


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (NaN entries are left as NaN and not counted)."""
    p = np.asarray(pvalues, dtype=float)
    ok = np.flatnonzero(np.isfinite(p))
    out = np.full(p.shape, np.nan)
    order = ok[np.argsort(p[ok], kind="stable")]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()


def one_sided_wilcoxon(d: np.ndarray, alternative: str) -> float:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size < 5 or not np.any(d != 0):
        return float("nan")
    return float(stats.wilcoxon(d, alternative=alternative, zero_method="wilcox").pvalue)


def inclusion_check(entry, paths, plan: dict[str, Any]) -> dict[str, Any]:
    """Apply the plan's inclusion rules using only labels and metadata (no expression values)."""
    rec: dict[str, Any] = {"dataset_id": entry.dataset_id, "checks": {}}
    path = entry.resolved_path(paths)
    if not path.exists():
        rec.update(included=False, reason="file not downloaded")
        return rec
    v = entry.validation_record(paths)
    rec["checks"]["validation"] = v["status"] if v else "not run"
    if not v or v["status"] != "valid":
        rec.update(included=False, reason=f"validation status: {rec['checks']['validation']}")
        return rec
    obs, var = read_obs_var(path)
    labels = obs["perturbation"].astype(object)
    n_ctrl = int((labels == "control").sum())
    rec["checks"]["n_control_cells"] = n_ctrl
    if n_ctrl < plan["min_control_cells"]:
        rec.update(included=False, reason=f"{n_ctrl} control cells (< {plan['min_control_cells']})")
        return rec
    filters = {c: v for c, v in FILTER_CANDIDATES.items() if c in obs.columns}
    keep = np.ones(len(obs), dtype=bool)
    for c, value in filters.items():
        keep &= (obs[c] == value).to_numpy()
    counts = labels[keep].value_counts()
    singles = {c: int(n) for c, n in counts.items() if c != "control" and len(parse_condition(c)) == 1}
    min_cells = plan["min_cells_per_condition"]
    if sum(n >= min_cells for n in singles.values()) < plan["min_single_conditions"]:
        min_cells = plan["fallback_min_cells_per_condition"]
    n_singles = sum(n >= min_cells for n in singles.values())
    rec["checks"].update(min_cells_used=min_cells, n_single_conditions=n_singles, n_multi_target_labels=int(sum(1 for c in counts.index if c != "control" and len(parse_condition(c)) > 1)))
    if n_singles < plan["min_single_conditions"]:
        rec.update(included=False, reason=f"{n_singles} single-gene conditions with >= {min_cells} cells (< {plan['min_single_conditions']})")
        return rec
    targets = sorted({parse_condition(c)[0] for c, n in singles.items() if n >= min_cells})
    resolved, rr = resolve_target_genes(targets, _var_with_ensembl(var), use_hgnc=True)
    frac = len(resolved) / max(len(targets), 1)
    rec["checks"]["target_resolution_fraction"] = round(frac, 4)
    if frac < plan["min_target_resolution"]:
        rec.update(included=False, reason=f"only {frac:.0%} of targets resolve to genes")
        return rec
    replicate = next((c for c in REPLICATE_CANDIDATES if c in obs.columns), None)
    rec.update(included=True, reason="passes all rules", replicate_col=replicate, cell_filters=filters, min_cells=min_cells,
               n_batches=int(obs[replicate].nunique()) if replicate else 1)
    return rec


def _var_with_ensembl(var: pd.DataFrame) -> pd.DataFrame:
    if not any(c in var.columns for c in ("ensembl_id", "ensemble_id", "gene_ids")) and var.index.astype(str).str.startswith("ENSG").mean() > 0.5:
        return var.assign(ensembl_id=var.index.astype(str))
    return var


def representation_featurizers(data: ResponseData, seed: int, gene2vec_path: str | None) -> tuple[dict[str, Featurizer], dict[str, Any]]:
    genes, E, base = list(data.embedding_genes), data.gene_embedding, data.embedding_baseline
    rng = np.random.default_rng(seed)
    feats = {"coexpression": default_featurizer(data), "shuffled_coexpression": Featurizer(genes, {"p": E[rng.permutation(len(genes))]}, base)}
    report: dict[str, Any] = {}
    if gene2vec_path and Path(gene2vec_path).exists():
        from biophysical.annotations.gene_ids import load_mapper
        from biophysical.models.external import align_embeddings, read_text_embeddings

        M, report["gene2vec"] = align_embeddings(read_text_embeddings(gene2vec_path), genes, load_mapper())
        pos = {g: i for i, g in enumerate(genes)}
        targets = sorted({g for gs in data.condition_genes() for g in gs})
        report["gene2vec"]["targets_with_vector_fraction"] = float(np.mean([M[pos[t]].any() for t in targets]))
        feats["gene2vec"] = Featurizer(genes, {"g": M}, base)
    return feats, report


def run_dataset(dataset_id: str, cfg: dict[str, Any], plan: dict[str, Any], out: Path, gene2vec_path: str | None) -> dict[str, Any]:
    reg = DatasetRegistry()
    entry = reg.get(dataset_id)
    inc = inclusion_check(entry, reg.paths, plan)
    write_json(inc, out / dataset_id / "inclusion.json")
    if not inc["included"]:
        log.info("%s excluded: %s", dataset_id, inc["reason"])
        return {"dataset_id": dataset_id, "included": False, "reason": inc["reason"]}
    overrides = {"replicate_col": inc["replicate_col"], "cell_filters": inc["cell_filters"], "min_cells_per_condition": inc["min_cells"], "max_perturbed_genes": 1, "control_label": "control"}
    t0 = time.time()
    processed = preprocess_dataset(dataset_id, config_path=cfg["base_config"], overrides=overrides)
    data = load_response_data(processed)
    prep_s = time.time() - t0
    base_cfg = load_yaml(cfg["base_config"])
    seed = int(cfg.get("seed", 0))
    n_folds, frac = base_cfg["splits"]["n_folds"], base_cfg["splits"]["calibration_fraction"]
    genes = data.condition_genes()
    factories = model_factories(base_cfg, "unseen_gene", seed)
    feats, rep_report = representation_featurizers(data, seed, gene2vec_path)
    frames, ceilings, oof = [], None, None
    t0 = time.time()
    for variant, feat in feats.items():
        names = cfg["models"] if variant == "coexpression" else cfg["representation_models"]
        res = evaluate_folds(data, unseen_gene_folds(genes, n_folds, frac, seed), {m: factories[m] for m in names}, feat, data.Y_fit, data.Y_eval)
        f = res["per_perturbation"].assign(variant=variant, split_seed=seed)
        frames.append(f)
        if variant == "coexpression":
            ceilings, oof = res["ceilings"], res["oof"]
        log.info("%s: %s done", dataset_id, variant)
    for s in cfg.get("robustness_seeds", []):
        res = evaluate_folds(data, unseen_gene_folds(genes, n_folds, frac, s), {m: factories[m] for m in cfg["confirmatory_models"] + ["mean"]}, feats["coexpression"], data.Y_fit, data.Y_eval)
        frames.append(res["per_perturbation"].assign(variant="coexpression", split_seed=s))
    eval_s = time.time() - t0
    frame = pd.concat(frames, ignore_index=True).assign(dataset_id=dataset_id)
    frame.to_csv(out / dataset_id / "per_perturbation.csv", index=False)
    ceilings.assign(dataset_id=dataset_id).to_csv(out / dataset_id / "noise_ceiling.csv", index=False)
    _ranking(data, oof).assign(dataset_id=dataset_id).to_csv(out / dataset_id / "ranking.csv", index=False)
    metric_calibration(data).assign(dataset_id=dataset_id).to_csv(out / dataset_id / "metric_calibration.csv", index=False)
    summary = {
        "dataset_id": dataset_id, "included": True, "processed_version": data.meta["version"], "source_md5": data.meta["source"].get("md5"),
        "n_conditions": int(len(data.conditions)), "n_genes": len(data.genes), "n_control_cells": data.n_control_cells, "n_batches": data.meta["n_batches"],
        "replicate_col": inc["replicate_col"], "cell_filters": inc["cell_filters"], "min_cells": inc["min_cells"], "excluded_conditions": data.meta["selection"]["n_conditions_excluded"],
        "target_aliases": data.meta.get("target_aliases", {}), "unresolved_targets": data.meta["target_resolution"]["unresolved"],
        "pooled_control_fallbacks": data.meta.get("n_condition_batches_using_pooled_control"), "representations": rep_report,
        "seconds_preprocess": round(prep_s, 1), "seconds_evaluate": round(eval_s, 1), "registry": {k: getattr(entry, k) for k in ("name", "publication", "accession", "assay", "cell_context")},
    }
    write_json(summary, out / dataset_id / "summary.json")
    return summary


CALIBRATION_METRICS = ["mse", "skill_vs_mean", "pearson", "pearson_specific", "spearman_specific", "pearson_specific_top", "topk_recall"]
LOWER_IS_BETTER = {"mse"}


def metric_calibration(data: ResponseData, top_k: int = 20) -> pd.DataFrame:
    """E4: per condition, does a technical duplicate beat the mean of other conditions on each metric?"""
    Y_ref = np.where(data.self_mask, np.nan, data.Y_fit)
    total, count = np.nansum(Y_ref, axis=0), np.sum(~np.isnan(Y_ref), axis=0)
    rows = []
    for i, cond in enumerate(data.conditions["condition"]):
        if not (np.isfinite(data.Y_half_a[i]).all() and np.isfinite(data.Y_half_b[i]).all()):
            continue
        own = np.nan_to_num(Y_ref[i])
        ref = (total - own) / np.maximum(count - (~np.isnan(Y_ref[i])).astype(int), 1)
        pos = perturbation_metrics(data.Y_half_b[i], data.Y_half_a[i], data.self_mask[i], top_k, reference=ref)
        neg = perturbation_metrics(data.Y_half_b[i], ref, data.self_mask[i], top_k, reference=ref)
        row = {"condition": cond}
        for m in CALIBRATION_METRICS:
            a, b = pos[m], neg[m]
            b = 0.0 if not np.isfinite(b) and m not in LOWER_IS_BETTER else b
            row[f"{m}_duplicate"], row[f"{m}_mean"] = a, b
            row[f"{m}_duplicate_wins"] = bool(a < b) if m in LOWER_IS_BETTER else bool(np.nan_to_num(a, nan=-np.inf) > b)
        rows.append(row)
    return pd.DataFrame(rows)


def _ranking(data: ResponseData, oof: dict) -> pd.DataFrame:
    any_model = next(iter(oof))
    test = np.concatenate([r["test"] for r in oof[any_model].values()])
    order = np.argsort(test)
    idx = test[order]
    M = data.self_mask[idx]
    truth = {"magnitude": response_magnitude(data.Y_eval[idx], M)}
    scores = {f"magnitude::{m}": response_magnitude(np.concatenate([r["P"] for r in per.values()])[order], M) for m, per in oof.items() if m != "zero"}
    return prioritisation_report(data.conditions["condition"].iloc[idx].tolist(), scores, truth, 0.2, 1000)


def learning_curve(dataset_id: str, cfg: dict[str, Any], sizes: list[int], repeats: int, seed: int = 0) -> pd.DataFrame:
    """E1: skill vs mean as the number of training perturbations grows (test folds fixed)."""
    reg = DatasetRegistry()
    inc = inclusion_check(reg.get(dataset_id), reg.paths, cfg["plan"])
    overrides = {"replicate_col": inc["replicate_col"], "cell_filters": inc["cell_filters"], "min_cells_per_condition": inc["min_cells"], "max_perturbed_genes": 1, "control_label": "control"}
    data = load_response_data(preprocess_dataset(dataset_id, config_path=cfg["base_config"], overrides=overrides))
    base_cfg = load_yaml(cfg["base_config"])
    alphas = tuple(base_cfg["models"]["ridge_alpha"])
    fit = make_fit_data(data, data.Y_fit, default_featurizer(data))
    rows = []
    rng = np.random.default_rng(seed)
    for fold in unseen_gene_folds(fit.cond_genes, base_cfg["splits"]["n_folds"], base_cfg["splits"]["calibration_fraction"], seed):
        test_genes = [fit.cond_genes[i] for i in fold.test]
        for n in sizes + [len(fold.train)]:
            if n > len(fold.train):
                continue
            for r in range(repeats if n < len(fold.train) else 1):
                tr = fit.subset(np.sort(rng.choice(fold.train, size=n, replace=False)))
                ref = MeanModel().fit(tr).predict(test_genes)
                for name, model in (("ridge", RidgeModel(alphas)), ("gene_loading_ridge", GeneLoadingRidge(alphas=alphas))):
                    P = model.fit(tr, fit.subset(fold.calib)).predict(test_genes)
                    for j, i in enumerate(fold.test):
                        m = perturbation_metrics(data.Y_eval[i], P[j], data.self_mask[i], reference=ref[j])
                        rows.append({"dataset_id": dataset_id, "fold": fold.fold, "n_train": n, "repeat": r, "model": name, "condition": data.conditions["condition"].iloc[i],
                                     "skill_vs_mean": m["skill_vs_mean"], "pearson_specific": m["pearson_specific"]})
        log.info("learning curve %s fold %d done", dataset_id, fold.fold)
    return pd.DataFrame(rows)


def cross_context(a_id: str, b_id: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """E2: predict dataset B's response to a target from dataset A's measured response to it."""
    reg = DatasetRegistry()
    datas = {}
    for d in (a_id, b_id):
        inc = inclusion_check(reg.get(d), reg.paths, cfg["plan"])
        overrides = {"replicate_col": inc["replicate_col"], "cell_filters": inc["cell_filters"], "min_cells_per_condition": inc["min_cells"], "max_perturbed_genes": 1, "control_label": "control"}
        datas[d] = load_response_data(preprocess_dataset(d, config_path=cfg["base_config"], overrides=overrides))
    rows = []
    for src, dst in ((a_id, b_id), (b_id, a_id)):
        S, D = datas[src], datas[dst]
        shared_genes = [g for g in D.genes if g in set(S.genes)]
        gs, gd = [S.genes.index(g) for g in shared_genes], [D.genes.index(g) for g in shared_genes]
        s_idx = {gs_[0]: i for i, gs_ in enumerate(S.condition_genes())}
        d_idx = {gs_[0]: i for i, gs_ in enumerate(D.condition_genes())}
        shared_targets = sorted(set(s_idx) & set(d_idx))
        Yd_fit, Yd_eval, Md = D.Y_fit[:, gd], D.Y_eval[:, gd], D.self_mask[:, gd]
        Ys = S.Y_all[:, gs]
        # global scale for the transferred response, fitted on targets other than the one predicted
        for t in shared_targets:
            i, k = d_idx[t], s_idx[t]
            others = np.array([d_idx[o] for o in shared_targets if o != t])
            others_s = np.array([s_idx[o] for o in shared_targets if o != t])
            ref = np.nanmean(np.where(Md[others], np.nan, Yd_fit[others]), axis=0)
            ref = np.nan_to_num(ref)
            src_mean = np.nan_to_num(np.nanmean(np.where(S.self_mask[others_s][:, gs], np.nan, Ys[others_s]), axis=0))
            dev_s, dev_d = Ys[others_s] - src_mean, Yd_fit[others] - ref
            keep = ~Md[others]
            beta = float((dev_s[keep] * dev_d[keep]).sum() / max((dev_s[keep] ** 2).sum(), 1e-12))
            preds = {"mean_of_other_targets": ref, "transfer_raw": Ys[k], "transfer_deviation_scaled": ref + beta * (Ys[k] - src_mean)}
            for name, p in preds.items():
                m = perturbation_metrics(Yd_eval[i], p, Md[i], reference=ref)
                rows.append({"source": src, "target_dataset": dst, "target": t, "predictor": name, "beta": beta, "n_shared_genes": len(shared_genes), **m})
    return pd.DataFrame(rows)


def confirmatory_tests(frame: pd.DataFrame, cfg: dict[str, Any], n_boot: int = 2000) -> pd.DataFrame:
    rows = []
    d = frame[(frame["stratum"] == "single_unseen") & (frame["split_seed"] == cfg.get("seed", 0))]
    num = ["mse", "skill_vs_mean", "pearson_specific"]
    d = d.groupby(["dataset_id", "variant", "model", "condition"], as_index=False, observed=True)[num].mean()
    for ds, g in d.groupby("dataset_id", observed=True):
        co = g[g["variant"] == "coexpression"]
        base = co[co["model"] == "mean"].set_index("condition")
        for m in cfg["confirmatory_models"]:
            a = co[co["model"] == m].set_index("condition")
            diff = a["mse"] - base["mse"].reindex(a.index)
            c = paired_comparison(a["mse"].to_numpy(), base["mse"].reindex(a.index).to_numpy(), n_boot)
            rows.append({"test": "C1", "dataset_id": ds, "model": m, "comparison": "mse(model) - mse(mean)", **c, "p_raw": one_sided_wilcoxon(diff.to_numpy(), "less"),
                         "mean_skill": float(a["skill_vs_mean"].mean()), "median_skill": float(a["skill_vs_mean"].median())})
        for m in cfg["representation_models"]:
            ref = co[co["model"] == m].set_index("condition")["skill_vs_mean"]
            sh = g[(g["variant"] == "shuffled_coexpression") & (g["model"] == m)].set_index("condition")["skill_vs_mean"].reindex(ref.index)
            c = paired_comparison(ref.to_numpy(), sh.to_numpy(), n_boot)
            rows.append({"test": "C2", "dataset_id": ds, "model": m, "comparison": "skill(coexpression) - skill(shuffled)", **c, "p_raw": c["wilcoxon_p"]})
            g2 = g[(g["variant"] == "gene2vec") & (g["model"] == m)].set_index("condition")["skill_vs_mean"].reindex(ref.index)
            if g2.notna().any():
                c = paired_comparison(g2.to_numpy(), ref.to_numpy(), n_boot)
                rows.append({"test": "C3", "dataset_id": ds, "model": m, "comparison": "skill(gene2vec) - skill(coexpression)", **c, "p_raw": c["wilcoxon_p"]})
    out = pd.DataFrame(rows, columns=["test", "dataset_id", "model", "comparison", "n", "mean_diff", "ci_low", "ci_high", "wilcoxon_p", "frac_a_gt_b", "p_raw", "mean_skill", "median_skill"]) if not rows else pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for t in ("C1", "C3"):
        m = out["test"] == t
        out.loc[m, "p_holm"] = holm(out.loc[m, "p_raw"].tolist())
    out["beats_mean"] = np.where(out["test"] == "C1", (out["p_holm"] < 0.05) & (out["ci_high"] < 0), np.nan)
    out["coexpression_informative"] = np.where(out["test"] == "C2", out["ci_low"] > 0, np.nan)
    out["gene2vec_better"] = np.where(out["test"] == "C3", (out["p_holm"] < 0.05) & (out["mean_diff"] > 0), np.nan)
    return out


def run_study(config_path: str | Path, out_dir: str | Path | None = None, only: list[str] | None = None) -> Path:
    cfg = load_yaml(config_path)
    paths = get_paths()
    out = Path(out_dir) if out_dir else paths.results / "study" / new_run_id("study")
    out.mkdir(parents=True, exist_ok=True)
    handler = add_json_file_handler(out / "study.log.jsonl")
    try:
        plan_sha = file_sha256(cfg["plan_file"])
        meta = {"plan_file": cfg["plan_file"], "plan_sha256": plan_sha, "addendum_sha256": {f: file_sha256(f) for f in cfg.get("plan_addenda", [])}, "plan_sha256_expected": cfg.get("plan_sha256"), "config": cfg, "environment": environment_info(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        if cfg.get("plan_sha256") and cfg["plan_sha256"] != plan_sha:
            meta["warning"] = "analysis plan changed after its hash was recorded"
            log.warning(meta["warning"])
        write_json(meta, out / "study.json")
        datasets = [d for d in cfg["datasets"] if not only or d in only]
        summaries = [run_dataset(d, cfg, cfg["plan"], out, cfg.get("gene2vec_path")) for d in datasets]
        pd.DataFrame(summaries).to_json(out / "datasets.json", orient="records", indent=2)
        included = [s["dataset_id"] for s in summaries if s.get("included")]
        frame = pd.concat([pd.read_csv(out / d / "per_perturbation.csv") for d in included], ignore_index=True)
        frame.to_csv(out / "per_perturbation_all.csv", index=False)
        confirm = confirmatory_tests(frame[frame["dataset_id"].isin(cfg["confirmation_datasets"])], cfg)
        confirm.to_csv(out / "confirmatory_tests.csv", index=False)
        confirmatory_tests(frame[frame["dataset_id"].isin(cfg.get("discovery_datasets", []))], cfg).to_csv(out / "discovery_tests.csv", index=False)
        e1 = [learning_curve(d, cfg, cfg["learning_curve"]["sizes"], cfg["learning_curve"]["repeats"]) for d in cfg["learning_curve"]["datasets"] if d in included]
        if e1:
            pd.concat(e1, ignore_index=True).to_csv(out / "learning_curve.csv", index=False)
        a, b = cfg["cross_context"]
        if a in included and b in included:
            cross_context(a, b, cfg).to_csv(out / "cross_context.csv", index=False)
        meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        meta["included"] = included
        write_json(meta, out / "study.json")
        (paths.results / "latest_study.txt").write_text(str(out))
        return out
    finally:
        import logging

        logging.getLogger("biophysical").removeHandler(handler)
        handler.close()
