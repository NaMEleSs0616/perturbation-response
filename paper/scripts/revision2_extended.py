"""Second-revision preprocessing (POST HOC). One pass per dataset that stores, for an extended output gene set:

  * Y_fit_r / Y_eval_r for R_C independent random splits of control cells (r = 0 reproduces the primary split)
  * responses from R_B random partitions of batches into two sets, each against its own batch-matched controls
    (k = 0 reproduces the primary alternating partition)
  * responses from two random halves of the cells within batch set A of partition 0, each against disjoint
    control halves, so perturbations can be selected for reproducibility using set A only and then evaluated
    against set B, which shares no cells or controls with the selection
  * per-condition standard errors and cell counts

Extended output genes: every gene detected in >= 1% of control cells or of at least one included condition's cells,
plus all targets. This includes genes that are almost silent in controls but induced by perturbations.

    python paper/scripts/revision2_extended.py <dataset_id> [...]
Writes results/paper/revision2/<dataset_id>.npz and .json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from biophysical.config import get_paths
from biophysical.data.inspect import read_obs_var
from biophysical.data.registry import DatasetRegistry
from biophysical.logging_utils import get_logger
from biophysical.preprocessing.pipeline import apply_cell_filters, load_response_data
from biophysical.preprocessing.pseudobulk import compute_group_stats

log = get_logger("revision2")
OUT = Path("results/paper/revision2")
R_B, R_C, MIN_CTRL = 5, 5, 20
DETECTION = 0.01


def partitions(levels: list[str], n: int, seed: int = 0) -> list[dict[str, str]]:
    """Partition 0 alternates sorted levels (the primary analysis); others are random balanced splits."""
    out = [{lv: ("A" if i % 2 == 0 else "B") for i, lv in enumerate(levels)}]
    rng = np.random.default_rng(seed + 101)
    for _ in range(1, n):
        perm = rng.permutation(levels)
        half = len(levels) // 2
        out.append({lv: ("A" if i < max(half, 1) else "B") for i, lv in enumerate(perm)})
    return out


def run(dataset_id: str) -> None:
    t0 = time.time()
    meta = next(x for x in json.loads(Path("results/study/primary/datasets.json").read_text()) if x["dataset_id"] == dataset_id)
    primary = load_response_data(get_paths().processed / dataset_id / meta["processed_version"])
    cfg = primary.meta["config"]
    col, ctrl_label = cfg["perturbation_col"], cfg["control_label"]
    path = DatasetRegistry().get(dataset_id).resolved_path()
    obs, var = read_obs_var(path)
    keep, _ = apply_cell_filters(obs, cfg["cell_filters"])
    rep_col = primary.meta["replicate_col_used"]
    rep = obs[rep_col].astype(str).to_numpy() if rep_col else np.array(["all"] * len(obs))
    conds = primary.conditions["condition"].tolist()
    labels = obs[col].astype(object).to_numpy()
    in_study = np.isin(labels, conds + [ctrl_label]) & keep
    halves0 = np.random.default_rng(0).integers(0, 2, size=len(obs))            # primary cell/control halves
    ctrl_halves = [halves0] + [np.random.default_rng(1000 + r).integers(0, 2, size=len(obs)) for r in range(1, R_C)]
    is_ctrl = (labels == ctrl_label) & in_study
    ctrl_counts = pd.Series(rep[is_ctrl]).value_counts()
    levels = sorted(ctrl_counts[ctrl_counts >= MIN_CTRL].index.tolist())
    parts = partitions(levels, R_B) if len(levels) >= 2 else []

    index: dict[tuple[str, str, str], int] = {}
    rows, cols = [], []

    def add(i: int, kind: str, c: str, level: str) -> None:
        rows.append(i)
        cols.append(index.setdefault((kind, c, level), len(index)))

    for i in np.flatnonzero(in_study):
        c = labels[i]
        add(i, "condition", c, "all")
        add(i, "half", c, str(int(halves0[i])))
        for k, part in enumerate(parts):
            if rep[i] in part:
                add(i, "bpart", c, f"{k}|{part[rep[i]]}")
        if parts and parts[0].get(rep[i]) == "A":
            add(i, "Ahalf", c, str(int(halves0[i])))
        if c == ctrl_label:
            add(i, "control_batch", c, rep[i])
            for r in range(R_C):
                add(i, "control_split", c, f"{r}|{rep[i]}|{int(ctrl_halves[r][i])}")
    M = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(obs), len(index)))
    groups = pd.DataFrame([{"kind": k, "condition": c, "level": l} for (k, c, l) in index])
    groups["n_cells"] = np.asarray(M.sum(axis=0)).ravel().astype(int)
    log.info("%s: %d groups x %d genes", dataset_id, len(groups), var.shape[0])
    stats = compute_group_stats(path, groups, M, target_sum=cfg["target_sum"])

    cond_t, cond_m = stats.means("condition")
    pos = {c: i for i, c in enumerate(cond_t["condition"])}
    n_cond = cond_t["n_cells"].to_numpy()
    det = stats.n_detected / np.maximum(n_cond[:, None], 1)
    gene_names = var.index.astype(str).to_numpy()
    targets = sorted({g for gs in primary.conditions["genes"] for g in gs.split("+")})
    name_to_idx = {g: i for i, g in enumerate(gene_names)}
    extended = np.flatnonzero((det.max(axis=0) >= DETECTION) & (cond_m.std(axis=0) > 0))
    gidx = np.array(sorted(set(extended.tolist()) | {name_to_idx[t] for t in targets if t in name_to_idx}))
    genes = gene_names[gidx]

    ci = pos[ctrl_label]
    ctrl_mean, n_c = cond_m[ci], n_cond[ci]
    counts = pd.crosstab(pd.Series(labels[in_study]), pd.Series(rep[in_study]))

    def means(kind: str) -> dict[tuple[str, str], np.ndarray]:
        t, m = stats.means(kind)
        return {(r.condition, r.level): m[i] for i, r in enumerate(t.itertuples(index=False)) if r.n_cells > 0}, {(r.condition, r.level): r.n_cells for r in t.itertuples(index=False)}

    split_m, split_n = means("control_split")
    batch_m, _ = means("control_batch")
    part_m, part_n = means("bpart")
    ah_m, ah_n = means("Ahalf")
    half_m, _ = means("half")

    def ctrl_ref(weights: pd.Series, lookup, fallback: np.ndarray) -> np.ndarray:
        out = np.zeros_like(fallback, dtype=np.float64)
        for lv, w in weights.items():
            if w > 0:
                out += w * lookup(lv, fallback)
        return out

    def pooled_split(r: int, h: int) -> np.ndarray:
        ks = [(k, v) for k, v in split_m.items() if k[1].startswith(f"{r}|") and k[1].endswith(f"|{h}")]
        n = np.array([split_n[k] for k, _ in ks], dtype=float)
        return (np.stack([v for _, v in ks]) * n[:, None]).sum(axis=0) / max(n.sum(), 1)

    C, G = len(conds), len(gidx)
    arrays = {f"Y_fit_{r}": np.zeros((C, G), np.float32) for r in range(R_C)}
    arrays.update({f"Y_eval_{r}": np.zeros((C, G), np.float32) for r in range(R_C)})
    for k in range(len(parts)):
        arrays[f"Y_part{k}_A"] = np.full((C, G), np.nan, np.float32)
        arrays[f"Y_part{k}_B"] = np.full((C, G), np.nan, np.float32)
    arrays["Y_Ahalf_0"] = np.full((C, G), np.nan, np.float32)
    arrays["Y_Ahalf_1"] = np.full((C, G), np.nan, np.float32)
    arrays["Y_half_a"] = np.zeros((C, G), np.float32)
    arrays["Y_half_b"] = np.zeros((C, G), np.float32)
    se = np.zeros((C, G), np.float32)
    ctrl_var = np.maximum(stats.sum_sq_log[ci] / n_c - ctrl_mean ** 2, 0)
    pooled = {(r, h): pooled_split(r, h) for r in range(R_C) for h in (0, 1)}
    for i, c in enumerate(conds):
        j = pos[c]
        w = counts.loc[c] / counts.loc[c].sum()
        for r in range(R_C):
            for h, name in ((0, "fit"), (1, "eval")):
                ref = ctrl_ref(w, lambda lv, fb, r=r, h=h: split_m.get((ctrl_label, f"{r}|{lv}|{h}"), fb), pooled[(r, h)])
                arrays[f"Y_{name}_{r}"][i] = (cond_m[j] - ref)[gidx]
        for k, part in enumerate(parts):
            for s in ("A", "B"):
                key = (c, f"{k}|{s}")
                if key not in part_m:
                    continue
                lv = [l for l, ss in part.items() if ss == s]
                ws = counts.loc[c].reindex(lv).fillna(0)
                if ws.sum() > 0:
                    ref = ctrl_ref(ws / ws.sum(), lambda l, fb: batch_m.get((ctrl_label, l), fb), ctrl_mean)
                    arrays[f"Y_part{k}_{s}"][i] = (part_m[key] - ref)[gidx]
        if parts:
            lvA = [l for l, ss in parts[0].items() if ss == "A"]
            ws = counts.loc[c].reindex(lvA).fillna(0)
            for h in (0, 1):
                key = (c, str(h))
                if key in ah_m and ws.sum() > 0:
                    ref = ctrl_ref(ws / ws.sum(), lambda l, fb, h=h: split_m.get((ctrl_label, f"0|{l}|{h}"), fb), pooled[(0, h)])
                    arrays[f"Y_Ahalf_{h}"][i] = (ah_m[key] - ref)[gidx]
        for h, name in ((0, "a"), (1, "b")):
            arrays[f"Y_half_{name}"][i] = (half_m[(c, str(h))] - half_m[(ctrl_label, str(h))])[gidx]
        n_p = n_cond[j]
        var_p = np.maximum(stats.sum_sq_log[j] / n_p - cond_m[j] ** 2, 0)
        se[i] = np.sqrt(var_p[gidx] / n_p + ctrl_var[gidx] / n_c)
    gpos = {g: k for k, g in enumerate(genes)}
    mask = np.zeros((C, G), bool)
    for i, gs in enumerate(primary.conditions["genes"]):
        for g in gs.split("+"):
            if g in gpos:
                mask[i, gpos[g]] = True
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{dataset_id}.npz", genes=genes, se=se, self_mask=mask, control_mean=ctrl_mean[gidx].astype(np.float32),
                        control_detection=det[ci][gidx].astype(np.float32), max_detection=det.max(axis=0)[gidx].astype(np.float32),
                        in_primary=np.isin(genes, primary.genes), n_cells=np.array([n_cond[pos[c]] for c in conds]), **arrays)
    (OUT / f"{dataset_id}.json").write_text(json.dumps({
        "dataset_id": dataset_id, "n_genes_extended": int(G), "n_genes_primary": len(primary.genes), "n_batch_levels": len(levels),
        "n_batch_partitions": len(parts), "distinct_partitions": len({tuple(sorted(p.items())) for p in parts} | set()),
        "n_control_splits": R_C, "detection_threshold": DETECTION, "seconds": round(time.time() - t0, 1),
        "conditions": conds}, indent=1))
    log.info("%s done: %d extended genes (%d primary) in %.0fs", dataset_id, G, len(primary.genes), time.time() - t0)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        run(d)
