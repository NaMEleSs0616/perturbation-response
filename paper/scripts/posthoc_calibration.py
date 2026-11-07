"""POST-HOC analysis, added after the confirmatory results were seen (disclosed in the paper).

The pre-specified calibration analysis (addendum E4) used the training mean as its only negative control.
The confirmatory results showed that a zero prediction also scores well on perturbation-specific
correlation, so this script adds zero as a second negative control: for each metric, how often does a
technical duplicate beat zero, and how often does it beat the better of mean and zero?

    python paper/scripts/posthoc_calibration.py results/study/primary
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from biophysical.config import get_paths
from biophysical.evaluation.metrics import perturbation_metrics
from biophysical.preprocessing.pipeline import load_response_data

METRICS = ["mse", "skill_vs_mean", "pearson", "pearson_specific", "spearman_specific", "topk_recall"]
LOWER = {"mse"}


def score(m: str, v: float) -> float:
    """Orient so larger is better; undefined correlations count as 0."""
    if m in LOWER:
        return -v
    return 0.0 if not np.isfinite(v) else v


def run(study: Path) -> pd.DataFrame:
    rows = []
    for d in json.loads((study / "datasets.json").read_text()):
        if not d.get("included"):
            continue
        data = load_response_data(get_paths().processed / d["dataset_id"] / d["processed_version"])
        Y_ref = np.where(data.self_mask, np.nan, data.Y_fit)
        total, count = np.nansum(Y_ref, axis=0), np.sum(~np.isnan(Y_ref), axis=0)
        for i in range(len(data.conditions)):
            if not (np.isfinite(data.Y_half_a[i]).all() and np.isfinite(data.Y_half_b[i]).all()):
                continue
            ref = (total - np.nan_to_num(Y_ref[i])) / np.maximum(count - (~np.isnan(Y_ref[i])).astype(int), 1)
            y, mask = data.Y_half_b[i], data.self_mask[i]
            dup = perturbation_metrics(y, data.Y_half_a[i], mask, reference=ref)
            mean = perturbation_metrics(y, ref, mask, reference=ref)
            zero = perturbation_metrics(y, np.zeros_like(ref), mask, reference=ref)
            row = {"dataset_id": d["dataset_id"], "condition": data.conditions["condition"].iloc[i]}
            for m in METRICS:
                a, b, z = score(m, dup[m]), score(m, mean[m]), score(m, zero[m])
                row[f"{m}_beats_zero"] = a > z
                row[f"{m}_beats_best_uninformative"] = a > max(b, z)
                row[f"{m}_zero_score"] = zero[m]
            rows.append(row)
        print(d["dataset_id"], "done", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t = run(Path(sys.argv[1]))
    out = Path("results/paper")
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "calibration_posthoc.csv", index=False)
    print(t.groupby("dataset_id")[[c for c in t.columns if c.endswith("best_uninformative")]].mean().round(2).to_string())
