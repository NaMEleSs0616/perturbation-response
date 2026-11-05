"""End-to-end preprocessing: raw H5AD -> versioned ResponseData on disk. Raw data is never modified."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

from biophysical.config import get_paths, load_yaml
from biophysical.data.h5sparse import x_layout
from biophysical.data.inspect import read_obs_var
from biophysical.data.registry import DatasetRegistry
from biophysical.logging_utils import get_logger
from biophysical.preprocessing.pseudobulk import build_groups, compute_group_stats, extract_dense_cells, select_genes
from biophysical.preprocessing.targets import ResponseData, build_responses, parse_condition
from biophysical.provenance import file_md5, json_digest, read_json, write_json

log = get_logger(__name__)

DEFAULTS: dict[str, Any] = {
    "perturbation_col": "perturbation",
    "replicate_col": None,
    "control_label": "control",
    "min_cells_per_condition": 50,
    "max_perturbed_genes": 2,
    "target_sum": 10000,
    "gene_set": "hvg",
    "n_genes": 2000,
    "min_control_detection": 0.01,
    "embedding_dim": 32,
    "max_control_cells": 8000,
    "condition_separator": "_",
    "cell_filters": {},            # obs column -> required value, applied before any aggregation
    "resolve_targets_with_hgnc": True,
}


def apply_cell_filters(obs: pd.DataFrame, filters: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Boolean keep-mask for cells, with a per-filter record. An unknown column is an error."""
    keep = np.ones(len(obs), dtype=bool)
    record: dict[str, Any] = {"filters": dict(filters), "removed_by_filter": {}}
    for col, value in (filters or {}).items():
        if col not in obs.columns:
            raise ValueError(f"cell filter column '{col}' not in obs; available: {list(obs.columns)}")
        passed = (obs[col] == value).to_numpy()
        record["removed_by_filter"][col] = int((keep & ~passed).sum())
        keep &= passed
    record["n_cells_after_filters"] = int(keep.sum())
    return keep, record


def resolve_target_genes(targets: list[str], var: pd.DataFrame, use_hgnc: bool = True) -> tuple[dict[str, str], dict[str, Any]]:
    """Map perturbation target labels to var names.

    Labels present in var are used as is. Others are resolved through HGNC to an Ensembl ID, which must
    match exactly one var row. Anything unresolved or ambiguous is reported and never guessed.
    """
    var_names = set(var.index.astype(str))
    resolved = {t: t for t in targets if t in var_names}
    unresolved = [t for t in targets if t not in var_names]
    record: dict[str, Any] = {"resolved_via_hgnc": [], "unresolved": []}
    if not unresolved:
        return resolved, record
    ens_col = next((c for c in ("ensembl_id", "ensemble_id", "gene_ids") if c in var.columns), None)
    mapper = None
    if use_hgnc and ens_col:
        try:
            from biophysical.annotations.gene_ids import load_mapper

            mapper = load_mapper()
            record["hgnc_source"] = mapper.source
        except Exception as exc:  # no network or file: report, do not guess
            record["hgnc_error"] = repr(exc)
    ens = var[ens_col].astype(str).str.split(".").str[0].to_numpy() if ens_col else None
    for t in unresolved:
        m = mapper.map(t) if mapper else None
        hits = var.index[ens == m.ensembl_gene_id].astype(str).tolist() if m and m.ensembl_gene_id else []
        if len(hits) == 1:
            resolved[t] = hits[0]
            record["resolved_via_hgnc"].append({"target": t, "hgnc_status": m.status, "hgnc_symbol": m.symbol, "ensembl": m.ensembl_gene_id, "var_name": hits[0]})
        else:
            record["unresolved"].append({"target": t, "hgnc_status": m.status if m else None, "n_var_matches": len(hits)})
    return resolved, record


def select_conditions(obs: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply condition-level filters and return the kept table plus a full exclusion record."""
    col, ctrl = cfg["perturbation_col"], cfg["control_label"]
    labels = obs[col].astype(object)
    counts = labels.value_counts()
    rows, excluded = [], []
    for cond, n in counts.items():
        if cond == ctrl:
            continue
        genes = parse_condition(cond, cfg["condition_separator"])
        if len(genes) == 0:
            excluded.append({"condition": cond, "n_cells": int(n), "reason": "no gene tokens"})
        elif len(genes) > cfg["max_perturbed_genes"]:
            excluded.append({"condition": cond, "n_cells": int(n), "reason": f"more than {cfg['max_perturbed_genes']} genes"})
        elif n < cfg["min_cells_per_condition"]:
            excluded.append({"condition": cond, "n_cells": int(n), "reason": f"fewer than {cfg['min_cells_per_condition']} cells"})
        else:
            rows.append({"condition": cond, "genes": "+".join(genes), "n_perts": len(genes), "n_cells": int(n)})
    table = pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)
    # two labels may parse to the same gene set, e.g. A_B and B_A: that would duplicate a profile
    # across splits, so it is treated as an error rather than handled silently
    dup = table[table["genes"].duplicated(keep=False)]
    if not dup.empty:
        raise ValueError(f"conditions with identical gene sets: {dup['condition'].tolist()}; "
                         "merge these labels before preprocessing to avoid leaking duplicates across splits")
    record = {
        "n_cells_total": int(len(obs)),
        "n_cells_missing_label": int(obs[col].isna().sum()),
        "n_control_cells": int(counts.get(ctrl, 0)),
        "n_conditions_kept": int(len(table)),
        "n_conditions_excluded": len(excluded),
        "n_cells_in_kept_conditions": int(table["n_cells"].sum()) if len(table) else 0,
        "excluded_conditions": excluded,
    }
    return table, record


def coexpression_embedding(control_expr: np.ndarray, dim: int, seed: int = 0) -> np.ndarray:
    """Gene embedding from control cells: right singular vectors of the z-scored matrix, scaled by
    singular values. Uses no perturbed cells, so it cannot leak perturbation labels."""
    mean, sd = control_expr.mean(axis=0), control_expr.std(axis=0)
    Z = (control_expr - mean) / np.where(sd > 0, sd, 1.0)
    dim = min(dim, min(Z.shape) - 1)
    _, s, vt = randomized_svd(Z, n_components=dim, random_state=seed)
    return (vt.T * s / np.sqrt(Z.shape[0])).astype(np.float32)


def resolve_replicate_col(obs: pd.DataFrame, requested: str | None) -> str | None:
    if requested and requested in obs.columns:
        return requested
    if requested:
        log.warning("replicate column '%s' not found; falling back to a single batch for all cells", requested)
    return None


def preprocess_h5ad(path: str | Path, cfg: dict[str, Any], out_root: str | Path, dataset_id: str = "dataset", seed: int = 0,
                    force: bool = False, source: dict[str, Any] | None = None) -> Path:
    cfg = {**DEFAULTS, **(cfg or {})}
    path = Path(path)
    raw_md5 = (source or {}).get("md5") or file_md5(path)
    version = json_digest({"cfg": cfg, "raw_md5": raw_md5, "seed": seed, "schema": 2})
    out = Path(out_root) / dataset_id / version
    if (out / "meta.json").exists() and not force:
        log.info("processed data already exists: %s", out)
        return out

    t0 = time.time()
    obs, var = read_obs_var(path)
    col, ctrl = cfg["perturbation_col"], cfg["control_label"]
    if col not in obs.columns:
        raise ValueError(f"column '{col}' not in obs; available: {list(obs.columns)}")
    if (obs[col].astype(object) == ctrl).sum() == 0:
        raise ValueError(f"no control cells labelled '{ctrl}' in '{col}'. Set control_label in the config; "
                         "the control group is not inferred automatically.")
    replicate_col = resolve_replicate_col(obs, cfg["replicate_col"])
    cell_keep, cell_record = apply_cell_filters(obs, cfg["cell_filters"])
    conditions, record = select_conditions(obs[cell_keep], cfg)
    if conditions.empty:
        raise ValueError("no conditions remain after filtering")
    record["cell_filters"] = cell_record
    record["n_cells_in_file"] = int(len(obs))

    # target labels are resolved to var names; conditions with an unresolvable target are excluded
    labels = sorted({g for gs in conditions["genes"] for g in gs.split("+")})
    resolved, target_record = resolve_target_genes(labels, var, cfg["resolve_targets_with_hgnc"])
    bad = {u["target"] for u in target_record["unresolved"]}
    if bad:
        dropped = conditions[conditions["genes"].apply(lambda gs: any(g in bad for g in gs.split("+")))]
        record["excluded_conditions"] += [{"condition": c, "n_cells": int(n), "reason": "target gene not resolvable to a var row"}
                                          for c, n in zip(dropped["condition"], dropped["n_cells"])]
        record["n_conditions_excluded"] = len(record["excluded_conditions"])
        conditions = conditions.drop(dropped.index).reset_index(drop=True)
        record["n_conditions_kept"] = int(len(conditions))
    conditions["target_labels"] = conditions["genes"]
    conditions["genes"] = conditions["genes"].apply(lambda gs: "+".join(sorted(resolved[g] for g in gs.split("+"))))

    # only kept conditions and controls are aggregated; everything else is excluded here
    keep_labels = set(conditions["condition"]) | {ctrl}
    obs_used = obs[[col] + ([replicate_col] if replicate_col else [])].copy()
    labelled = obs_used[col].astype(object)
    obs_used[col] = labelled.where(labelled.isin(keep_labels) & cell_keep, None)

    log.info("computing group statistics (%s, %d cells x %d genes)", x_layout(path).encoding, *x_layout(path).shape)
    groups, membership = build_groups(obs_used, col, replicate_col, ctrl, seed=seed)
    stats = compute_group_stats(path, groups, membership, target_sum=cfg["target_sum"])

    gene_names = var.index.astype(str).to_numpy()
    cond_rows = groups[groups["kind"] == "condition"].reset_index(drop=True)
    c_i = cond_rows.index[cond_rows["condition"] == ctrl][0]
    n_ctrl = float(cond_rows.loc[c_i, "n_cells"])
    cond_sum = stats.sum_log[(groups["kind"] == "condition").to_numpy()]
    ctrl_mean = cond_sum[c_i] / n_ctrl
    ctrl_var = np.maximum(stats.sum_sq_log[c_i] / n_ctrl - ctrl_mean ** 2, 0)
    ctrl_det = stats.n_detected[c_i] / n_ctrl

    chosen = select_genes(ctrl_mean, ctrl_var, ctrl_det, cfg["gene_set"], cfg["n_genes"], cfg["min_control_detection"])

    # perturbed target genes are added so gene-loading features exist for them; their own entry is
    # masked in the condition that perturbs them, in both training and evaluation
    name_to_idx: dict[str, int] = {}
    n_dup_names = 0
    for i, g in enumerate(gene_names):
        if g in name_to_idx:
            n_dup_names += 1
        else:
            name_to_idx[g] = i
    target_genes = sorted({g for gs in conditions["genes"] for g in gs.split("+")})
    missing_targets = [g for g in target_genes if g not in name_to_idx]
    target_idx = np.array([name_to_idx[g] for g in target_genes if g in name_to_idx], dtype=np.int64)
    gene_idx = np.array(sorted(set(chosen.tolist()) | set(target_idx.tolist())), dtype=np.int64)
    log.info("selected %d genes (%d %s + %d target genes)", gene_idx.size, chosen.size, cfg["gene_set"], gene_idx.size - chosen.size)

    arrays = build_responses(stats, conditions, gene_names, gene_idx, ctrl)

    # co-expression embedding from a capped random sample of control cells
    ctrl_cells = np.flatnonzero((obs[col].astype(object).to_numpy() == ctrl) & cell_keep)
    rng = np.random.default_rng(seed)
    if ctrl_cells.size > cfg["max_control_cells"]:
        ctrl_cells = np.sort(rng.choice(ctrl_cells, size=cfg["max_control_cells"], replace=False))
    control_expr = extract_dense_cells(path, ctrl_cells, gene_idx, target_sum=cfg["target_sum"])
    embedding = coexpression_embedding(control_expr, cfg["embedding_dim"], seed=seed)
    del control_expr
    baseline = np.stack([arrays["control_mean"], arrays["control_detection"]], axis=1)

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "response.npz",
        **{k: arrays[k] for k in ["Y_fit", "Y_eval", "Y_half_a", "Y_half_b", "Y_batch_a", "Y_batch_b", "Y_logmean", "se", "self_mask", "control_mean", "control_detection"]},
        gene_embedding=embedding, embedding_baseline=baseline,
    )
    conditions.to_csv(out / "conditions.csv", index=False)
    genes_table = pd.DataFrame({
        "gene": arrays["genes"],
        "var_index": gene_idx,
        "selected_as": np.where(np.isin(gene_idx, chosen), cfg["gene_set"], "perturbed_target"),
        "control_mean_log1p": arrays["control_mean"],
        "control_detection": arrays["control_detection"],
    })
    ens_col = next((c for c in ("ensembl_id", "ensemble_id", "gene_ids") if c in var.columns), None)
    if ens_col:
        genes_table["ensembl_id"] = var[ens_col].astype(str).to_numpy()[gene_idx]
    genes_table.to_csv(out / "genes.csv", index=False)
    meta = {
        "dataset_id": dataset_id,
        "version": version,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seconds": round(time.time() - t0, 1),
        "source": {"file": path.name, "md5": raw_md5, **(source or {})},
        "config": cfg,
        "seed": seed,
        "replicate_col_used": replicate_col,
        "n_batches": int(obs[replicate_col].nunique()) if replicate_col else 1,
        "n_control_cells": arrays["n_control_cells"],
        "n_control_cells_for_embedding": int(ctrl_cells.size),
        "n_genes_in_file": int(len(gene_names)),
        "n_genes_used": int(gene_idx.size),
        "n_genes_removed": int(len(gene_names) - gene_idx.size),
        "n_duplicate_gene_names_in_file": n_dup_names,
        "target_genes_not_in_var": missing_targets,
        "selection": record,
        "target_resolution": target_record,
        "target_aliases": {k: v for k, v in resolved.items() if k != v},
        "normalisation": f"library-size scaling to {cfg['target_sum']} per cell over all genes in the file, then log1p",
        "target": "cell-weighted mean of per-batch pseudobulk differences in log1p-normalised expression; "
                  "Y_fit and Y_eval use disjoint control cells",
    }
    write_json(meta, out / "meta.json")
    log.info("preprocessing finished in %.0fs: %s", time.time() - t0, out)
    return out


def load_response_data(path: str | Path) -> ResponseData:
    path = Path(path)
    z = np.load(path / "response.npz")
    conditions = pd.read_csv(path / "conditions.csv", keep_default_na=False)
    genes = pd.read_csv(path / "genes.csv", keep_default_na=False)["gene"].astype(str).tolist()
    meta = read_json(path / "meta.json")
    return ResponseData(
        conditions=conditions, genes=genes, Y_fit=z["Y_fit"], Y_eval=z["Y_eval"], Y_half_a=z["Y_half_a"], Y_half_b=z["Y_half_b"],
        Y_batch_a=z["Y_batch_a"], Y_batch_b=z["Y_batch_b"], Y_logmean=z["Y_logmean"], se=z["se"], self_mask=z["self_mask"],
        control_mean=z["control_mean"], control_detection=z["control_detection"], n_control_cells=int(meta["n_control_cells"]),
        embedding_genes=genes, gene_embedding=z["gene_embedding"], embedding_baseline=z["embedding_baseline"], meta=meta,
    )


def preprocess_dataset(dataset_id: str, config_path: str | Path | None = None, force: bool = False, overrides: dict[str, Any] | None = None) -> Path:
    reg = DatasetRegistry()
    entry = reg.get(dataset_id)
    cfg_all = load_yaml(config_path) if config_path else {}
    cfg = {**cfg_all.get("preprocess", {}), **(overrides or {})}
    cfg.setdefault("control_label", entry.control_label)
    path = entry.resolved_path(reg.paths)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `datasets download {dataset_id}` or `datasets set-local {dataset_id} <path>`.")
    source = {"accession": entry.accession, "doi": entry.doi, "url": entry.download_url, "publication": entry.publication}
    if entry.md5 and not entry.local_path:
        source["md5"] = entry.md5
    return preprocess_h5ad(path, cfg, get_paths().processed, dataset_id=dataset_id, seed=cfg_all.get("seed", 0), force=force, source=source)
