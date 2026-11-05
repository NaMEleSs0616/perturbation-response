"""Schema inspection of H5AD perturbation datasets without loading X into memory."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biophysical.logging_utils import get_logger
from biophysical.provenance import write_json

log = get_logger(__name__)

REPLICATE_LIKE = ["gemgroup", "gem_group", "batch", "replicate", "library", "lane", "sample", "donor", "plate"]


def open_backed(path: str | Path):
    import anndata as ad

    return ad.read_h5ad(path, backed="r")


def read_obs_var(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read only obs and var (never X)."""
    import h5py
    from anndata.io import read_elem

    with h5py.File(path, "r") as f:
        return read_elem(f["obs"]), read_elem(f["var"])


def describe_column(s: pd.Series, max_levels: int = 25) -> dict[str, Any]:
    out: dict[str, Any] = {"dtype": str(s.dtype), "n_missing": int(s.isna().sum())}
    if pd.api.types.is_numeric_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype):
        d = s.describe()
        out.update({k: float(d[k]) for k in ["min", "25%", "50%", "75%", "max", "mean"] if k in d})
        out["n_unique"] = int(s.nunique())
    else:
        vc = s.astype(str).value_counts()
        out["n_unique"] = int(vc.size)
        out["top_levels"] = {str(k): int(v) for k, v in vc.head(max_levels).items()}
    return out


def inspect_h5ad(path: str | Path, control_label: str = "control", perturbation_col: str = "perturbation") -> dict[str, Any]:
    import h5py

    from biophysical.data.h5sparse import sample_block, x_layout

    obs, var = read_obs_var(path)
    layout = x_layout(path)
    with h5py.File(path, "r") as f:
        uns_keys = list(f["uns"].keys()) if "uns" in f else []
        layers = list(f["layers"].keys()) if "layers" in f else []
        obsm = list(f["obsm"].keys()) if "obsm" in f else []
    xs = sample_block(path)
    report: dict[str, Any] = {
        "path": str(path),
        "n_obs": int(obs.shape[0]),
        "n_vars": int(var.shape[0]),
        "X_encoding": layout.encoding,
        "X_nnz": layout.nnz,
        "X_sample_dtype": str(xs.dtype),
        "X_sample_min": float(xs.min()),
        "X_sample_max": float(xs.max()),
        "X_sample_integer_valued": bool(np.allclose(xs, np.round(xs))),
        "X_sample_fraction_nonzero": float((xs != 0).mean()),
        "obs_columns": {c: describe_column(obs[c]) for c in obs.columns},
        "var_columns": {c: describe_column(var[c], max_levels=5) for c in var.columns},
        "var_index_example": [str(v) for v in var.index[:10]],
        "var_names_unique": bool(var.index.is_unique),
        "obs_names_unique": bool(obs.index.is_unique),
        "uns_keys": uns_keys,
        "layers": layers,
        "obsm_keys": obsm,
    }
    if perturbation_col in obs:
        pert = obs[perturbation_col].astype(str)
        counts = pert.value_counts()
        report["perturbation_summary"] = {
            "n_conditions_including_control": int(counts.size),
            "n_control_cells": int(counts.get(control_label, 0)),
            "cells_per_condition_quantiles": {q: float(counts.quantile(q)) for q in [0.0, 0.1, 0.5, 0.9, 1.0]},
            "n_conditions_ge_50_cells": int((counts >= 50).sum()),
        }
    report["replicate_like_columns"] = [c for c in obs.columns if any(k in c.lower() for k in REPLICATE_LIKE)]
    return report


def inspect_dataset(entry, paths=None, write: bool = True) -> dict[str, Any]:
    path = entry.resolved_path(paths)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m biophysical.cli datasets download {entry.dataset_id} "
                                f"or register a local copy with `datasets set-local`.")
    report = {"dataset_id": entry.dataset_id, "registry": {k: getattr(entry, k) for k in
              ["name", "publication", "accession", "doi", "license", "organism", "assay", "cell_context", "perturbation_types"]}}
    report["schema"] = inspect_h5ad(path, control_label=entry.control_label)
    if write:
        write_json(report, entry.raw_dir(paths) / "inspection.json")
    return report
