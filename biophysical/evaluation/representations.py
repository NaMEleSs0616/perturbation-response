"""Do external gene representations improve held-out prediction?

Runs the same folds, models and target with different perturbation representations, and compares each
with co-expression features on the same perturbations. A permuted embedding is the negative control.

STRING edges are functional associations. An improvement here means the network helps prediction;
it does not establish direct physical or causal relationships.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from biophysical.config import get_paths, load_yaml
from biophysical.evaluation.benchmark import compare, evaluate_folds, model_factories, per_condition, summarise
from biophysical.logging_utils import get_logger
from biophysical.models.features import Featurizer, default_featurizer
from biophysical.networks.features import graph_summary, neighbour_average, spectral_embedding
from biophysical.preprocessing.pipeline import load_response_data, preprocess_dataset
from biophysical.preprocessing.splits import unseen_combo_folds, unseen_gene_folds
from biophysical.provenance import write_json

log = get_logger(__name__)
TASK_MODELS = {"unseen_gene": ["mean", "ridge", "knn", "mlp_ensemble"], "unseen_combo": ["mean", "additive", "ridge_on_additive", "mlp_on_additive"]}


def build_variants(data, seed: int, use_string: bool = True, gene2vec_path: str | Path | None = None) -> tuple[dict[str, Featurizer], dict]:
    genes, E, base = list(data.embedding_genes), data.gene_embedding, data.embedding_baseline
    rng = np.random.default_rng(seed)
    variants: dict[str, Featurizer] = {
        "coexpression": default_featurizer(data),
        "baseline_only": Featurizer(genes, {}, base),
        "permuted_coexpression": Featurizer(genes, {"p": E[rng.permutation(len(genes))]}, base),
    }
    report: dict = {}
    if use_string:
        from biophysical.networks.build import string_adjacency
        from biophysical.networks.string_db import StringClient

        adj, report["string"] = string_adjacency(genes, StringClient(), channel_sets={"combined": None, "experimental_curated": ["experimental", "curated"]})
        targets = sorted({g for gs in data.condition_genes() for g in gs})
        for name, A in adj.items():
            report[f"graph_{name}"] = graph_summary(A, genes, targets)
            variants[f"coexpr+string_neighbours_{name}"] = Featurizer(genes, {"c": E, "n": neighbour_average(A, E)}, base)
            variants[f"string_spectral_{name}"] = Featurizer(genes, {"s": spectral_embedding(A, 32)}, base)
    if gene2vec_path:
        from biophysical.annotations.gene_ids import load_mapper
        from biophysical.models.external import align_embeddings, read_text_embeddings

        M, report["gene2vec"] = align_embeddings(read_text_embeddings(gene2vec_path), genes, load_mapper())
        variants["gene2vec_frozen"] = Featurizer(genes, {"g": M}, base)
        variants["coexpr+gene2vec_frozen"] = Featurizer(genes, {"c": E, "g": M}, base)
        report["gene2vec"]["path"] = str(gene2vec_path)
        targets = sorted({g for gs in data.condition_genes() for g in gs})
        pos = {g: i for i, g in enumerate(genes)}
        report["gene2vec"]["targets_without_vector"] = [t for t in targets if not M[pos[t]].any()]
    return variants, report


def run_representation_benchmark(config_path: str | Path, out_dir: str | Path, use_string: bool = True, gene2vec_path: str | Path | None = None) -> Path:
    cfg = load_yaml(config_path)
    seed = int(cfg.get("seed", 0))
    data = load_response_data(preprocess_dataset(cfg["dataset_id"], config_path=config_path))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    variants, report = build_variants(data, seed, use_string, gene2vec_path)
    genes_all = data.condition_genes()
    n_folds, frac = cfg["splits"]["n_folds"], cfg["splits"]["calibration_fraction"]
    tasks = {"unseen_gene": unseen_gene_folds(genes_all, n_folds, frac, seed), "unseen_combo": unseen_combo_folds(genes_all, n_folds, frac, seed)}
    frames = []
    for variant, feat in variants.items():
        for task, folds in tasks.items():
            factories = model_factories(cfg, task, seed)
            res = evaluate_folds(data, folds, {m: factories[m] for m in TASK_MODELS[task]}, feat, data.Y_fit, data.Y_eval)
            f = res["per_perturbation"]
            f["variant"] = variant
            frames.append(f)
        log.info("representation %s done", variant)
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(out / "representation_per_perturbation.csv", index=False)
    rows, paired = [], []
    for variant, f in frame.groupby("variant", observed=True):
        s = summarise(f.drop(columns="variant"), None, 1000)
        s.insert(0, "variant", variant)
        rows.append(s)
    pd.concat(rows, ignore_index=True).to_csv(out / "representation_summary.csv", index=False)
    # paired comparison of each variant against co-expression, per model and metric
    from biophysical.evaluation.metrics import paired_comparison

    d = frame.copy()
    num = d.select_dtypes("number").columns.difference(["fold"])
    d = d.groupby(["variant", "task", "stratum", "model", "condition"], as_index=False, observed=True)[list(num)].mean()
    ref = d[d["variant"] == "coexpression"]
    for variant, g in d[d["variant"] != "coexpression"].groupby("variant", observed=True):
        for (task, stratum, model), gg in g.groupby(["task", "stratum", "model"], observed=True):
            r = ref[(ref.task == task) & (ref.stratum == stratum) & (ref.model == model)].set_index("condition")
            for metric in ["pearson_specific", "skill_vs_mean", "mse"]:
                gi = gg.set_index("condition")
                paired.append({"variant": variant, "reference": "coexpression", "task": task, "stratum": stratum, "model": model, "metric": metric,
                               **paired_comparison(gi[metric].to_numpy(), r[metric].reindex(gi.index).to_numpy(), n_boot=1000)})
    pd.DataFrame(paired).to_csv(out / "representation_paired.csv", index=False)
    write_json({"config": str(config_path), "seed": seed, "variants": list(variants), **report}, out / "representation_report.json")
    return out
