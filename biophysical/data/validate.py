"""Dataset validation: integrity, schema and control-group checks."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from biophysical.data.h5sparse import sample_block
from biophysical.data.inspect import read_obs_var
from biophysical.logging_utils import get_logger
from biophysical.provenance import file_md5, write_json

log = get_logger(__name__)
MIN_CONTROL_CELLS = 100


def validate_tables(obs, var, xs: np.ndarray, expected_obs_fields: list[str], control_label: str,
                    perturbation_col: str = "perturbation") -> list[dict[str, Any]]:
    """Return a list of check records ({check, passed, detail, severity})."""
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str, severity: str = "error") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail, "severity": severity})

    missing = [f for f in expected_obs_fields if f not in obs.columns]
    add("expected_obs_fields", not missing, f"missing: {missing}" if missing else "all present")
    has_pert = perturbation_col in obs.columns
    add("perturbation_column", has_pert, perturbation_col)
    if has_pert:
        pert = obs[perturbation_col].astype(str)
        n_ctrl = int((pert == control_label).sum())
        add("control_cells", n_ctrl >= MIN_CONTROL_CELLS, f"{n_ctrl} cells labelled '{control_label}' (min {MIN_CONTROL_CELLS})")
        n_na = int(obs[perturbation_col].isna().sum())
        add("perturbation_missing_labels", True, f"{n_na} cells with missing perturbation label (excluded downstream)", "warning")
    add("obs_names_unique", obs.index.is_unique, "duplicated cell barcodes would allow leakage across splits")
    add("var_names_unique", var.index.is_unique, "duplicated gene names are made unique downstream" , "warning")
    add("X_nonnegative", float(xs.min()) >= 0, f"sample min={xs.min()}")
    add("X_integer_counts", bool(np.allclose(xs, np.round(xs))), "raw counts expected for library-size normalisation", "warning")
    add("X_not_empty", float((xs != 0).mean()) > 0, f"fraction nonzero={float((xs != 0).mean()):.4f}")
    return checks


def validate_dataset(entry, paths=None, check_md5: bool = True) -> dict[str, Any]:
    path = entry.resolved_path(paths)
    rec: dict[str, Any] = {"dataset_id": entry.dataset_id, "path": str(path), "validated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if not path.exists():
        rec.update(status="missing", checks=[{"check": "file_exists", "passed": False, "detail": str(path), "severity": "error"}])
        write_json(rec, entry.raw_dir(paths) / "validation.json")
        return rec
    checks = [{"check": "file_exists", "passed": True, "detail": str(path), "severity": "error"}]
    if check_md5 and entry.md5 and not entry.local_path:
        got = file_md5(path)
        checks.append({"check": "md5", "passed": got == entry.md5, "detail": f"expected {entry.md5}, got {got}", "severity": "error"})
    elif entry.local_path:
        checks.append({"check": "md5", "passed": True, "detail": f"local file, md5={file_md5(path)} (not compared to registry)", "severity": "warning"})
    try:
        obs, var = read_obs_var(path)
        checks += validate_tables(obs, var, sample_block(path), entry.expected_obs_fields, entry.control_label)
        rec["shape"] = [int(obs.shape[0]), int(var.shape[0])]
    except Exception as exc:  # unreadable file is a validation failure, not a crash
        checks.append({"check": "readable_h5ad", "passed": False, "detail": repr(exc), "severity": "error"})
    rec["checks"] = checks
    rec["status"] = "valid" if all(c["passed"] for c in checks if c["severity"] == "error") else "invalid"
    write_json(rec, entry.raw_dir(paths) / "validation.json")
    return rec
