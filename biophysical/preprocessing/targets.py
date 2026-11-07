"""Response target construction.

Primary target (per condition p, gene g):

    Delta[p, g] = sum_r w_{p,r} * (mean_{p,r}[log1p CP10k g] - mean_{control,r}[log1p CP10k g])

where r indexes batches (e.g. 10x lanes) and w_{p,r} is proportional to the cells of p in r, so
batch composition differences between a condition and the controls do not enter the target.
This is a difference in pseudobulk means of log-normalised expression. Interpreting it causally
relies on random guide assignment within batch, which is a property of pooled screens and is
assumed, not tested, here.

Two versions are produced with DISJOINT control cells (Y_fit for training, Y_eval for evaluation).
If both used the same control pool, every response would share the same control noise and the
mean-response baseline would correlate with held-out targets for a purely technical reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from biophysical.preprocessing.pseudobulk import GroupStats


@dataclass
class ResponseData:
    conditions: pd.DataFrame        # condition, genes ("A" or "A+B"), n_perts, n_cells
    genes: list[str]                # prediction genes
    Y_fit: np.ndarray               # (n_cond, n_genes) target for training (control split 0)
    Y_eval: np.ndarray              # (n_cond, n_genes) target for evaluation (control split 1)
    Y_half_a: np.ndarray            # split-half responses (cells and controls both disjoint)
    Y_half_b: np.ndarray
    Y_batch_a: np.ndarray           # responses from disjoint sets of batches; NaN if unavailable
    Y_batch_b: np.ndarray
    Y_logmean: np.ndarray           # alternative target: log1p(mean CP10k) difference
    se: np.ndarray                  # standard error of the response from cell-level variance
    self_mask: np.ndarray           # True where the gene is a perturbed target of the condition
    control_mean: np.ndarray
    control_detection: np.ndarray
    n_control_cells: int
    embedding_genes: list[str] = field(default_factory=list)
    gene_embedding: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    embedding_baseline: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    meta: dict = field(default_factory=dict)

    @property
    def Y_all(self) -> np.ndarray:
        """Average over both control splits; used only for models trained on all measured data."""
        return 0.5 * (self.Y_fit + self.Y_eval)

    def condition_genes(self) -> list[tuple[str, ...]]:
        return [tuple(g.split("+")) for g in self.conditions["genes"]]

    def index(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.conditions["condition"])}


def parse_condition(condition: str, separator: str = "_", control_tokens: frozenset[str] = frozenset({"control", "ctrl", "NegCtrl0"})) -> tuple[str, ...]:
    """'A_B' -> ('A', 'B'); control tokens within a label are dropped ('A_ctrl' -> ('A',))."""
    return tuple(sorted(t for t in str(condition).split(separator) if t and t not in control_tokens))


def _control_reference(weights: pd.Series, controls: dict[str, np.ndarray], fallback: np.ndarray) -> np.ndarray:
    """sum_r w_r ctrl_r over batches. A batch with too few control cells uses the pooled control mean."""
    out = np.zeros_like(fallback, dtype=np.float64)
    for level, w in weights.items():
        if w > 0:
            out += w * controls.get(level, fallback)
    return out


def build_responses(stats: GroupStats, conditions: pd.DataFrame, gene_names: np.ndarray, gene_idx: np.ndarray,
                    control_label: str, batch_info: dict, min_control_cells_per_split: int = 10) -> dict:
    """Compute all response arrays for `conditions` over `gene_idx` (see build_groups for the identity used)."""
    groups = stats.groups
    n_cond, n_genes = len(conditions), len(gene_idx)
    counts: pd.DataFrame = batch_info["batch_counts"]
    batch_set: dict[str, str] = batch_info["batch_set"]

    cond_table, cond_means = stats.means("condition")
    cond_pos = {c: i for i, c in enumerate(cond_table["condition"])}
    c_i = cond_pos[control_label]
    n_c = float(cond_table.loc[c_i, "n_cells"])
    ctrl_mean = cond_means[c_i]
    ctrl_var = np.maximum(stats.sum_sq_log[c_i] / n_c - ctrl_mean ** 2, 0)
    ctrl_mean_norm = stats.sum_norm[c_i] / n_c

    split_table, split_means = stats.means("control_split")
    ctrl_split: dict[int, dict[str, np.ndarray]] = {0: {}, 1: {}}
    pooled_split = {}
    for h in (0, 1):
        m = split_table["level"].str.endswith(f"|{h}").to_numpy()
        n = split_table["n_cells"].to_numpy()[m]
        pooled_split[h] = (split_means[m] * n[:, None]).sum(axis=0) / max(n.sum(), 1)
    for i, r in enumerate(split_table.itertuples(index=False)):
        level, half = r.level.rsplit("|", 1)
        if r.n_cells >= min_control_cells_per_split:
            ctrl_split[int(half)][level] = split_means[i]
    batch_table, batch_means = stats.means("control_batch")
    ctrl_batch = {r.level: batch_means[i] for i, r in enumerate(batch_table.itertuples(index=False)) if r.level in batch_set}
    set_table, set_means = stats.means("batchset")
    set_lookup = {(r.condition, r.level): i for i, r in enumerate(set_table.itertuples(index=False))}
    half_table, half_means = stats.means("half")
    half_ctrl = {r.level: half_means[i] for i, r in enumerate(half_table.itertuples(index=False)) if r.condition == control_label}
    half_lookup = {(r.condition, r.level): i for i, r in enumerate(half_table.itertuples(index=False))}

    Y_fit = np.zeros((n_cond, n_genes), dtype=np.float32)
    Y_eval = np.zeros((n_cond, n_genes), dtype=np.float32)
    Y_a = np.zeros((n_cond, n_genes), dtype=np.float32)
    Y_b = np.zeros((n_cond, n_genes), dtype=np.float32)
    Y_ba = np.full((n_cond, n_genes), np.nan, dtype=np.float32)
    Y_bb = np.full((n_cond, n_genes), np.nan, dtype=np.float32)
    Y_lm = np.zeros((n_cond, n_genes), dtype=np.float32)
    se = np.zeros((n_cond, n_genes), dtype=np.float32)
    n_fallback = 0

    for i, cond in enumerate(conditions["condition"]):
        j = cond_pos[cond]
        n_p = float(cond_table.loc[j, "n_cells"])
        mean_p = cond_means[j]
        w = counts.loc[cond] / counts.loc[cond].sum()
        n_fallback += int(sum(1 for lv, x in w.items() if x > 0 and (lv not in ctrl_split[0] or lv not in ctrl_split[1])))
        Y_fit[i] = (mean_p - _control_reference(w, ctrl_split[0], pooled_split[0]))[gene_idx]
        Y_eval[i] = (mean_p - _control_reference(w, ctrl_split[1], pooled_split[1]))[gene_idx]
        for out, s_name in ((Y_ba, "A"), (Y_bb, "B")):
            k = set_lookup.get((cond, s_name))
            if k is None:
                continue
            levels = [lv for lv, sname in batch_set.items() if sname == s_name]
            ws = counts.loc[cond].reindex(levels).fillna(0)
            if ws.sum() > 0:
                out[i] = (set_means[k] - _control_reference(ws / ws.sum(), ctrl_batch, ctrl_mean))[gene_idx]
        for out, h in ((Y_a, "0"), (Y_b, "1")):
            k = half_lookup.get((cond, h))
            out[i] = (half_means[k] - half_ctrl[h])[gene_idx] if k is not None else np.nan
        var_p = np.maximum(stats.sum_sq_log[j] / n_p - mean_p ** 2, 0)
        se[i] = np.sqrt(var_p[gene_idx] / n_p + ctrl_var[gene_idx] / n_c)
        Y_lm[i] = (np.log1p(stats.sum_norm[j] / n_p) - np.log1p(ctrl_mean_norm))[gene_idx]

    names = [str(gene_names[g]) for g in gene_idx]
    pos = {g: k for k, g in enumerate(names)}
    self_mask = np.zeros((n_cond, n_genes), dtype=bool)
    for i, gs in enumerate(conditions["genes"]):
        for g in gs.split("+"):
            if g in pos:
                self_mask[i, pos[g]] = True

    return {
        "genes": names, "Y_fit": Y_fit, "Y_eval": Y_eval, "Y_half_a": Y_a, "Y_half_b": Y_b, "Y_batch_a": Y_ba, "Y_batch_b": Y_bb,
        "Y_logmean": Y_lm, "se": se, "self_mask": self_mask,
        "control_mean": ctrl_mean[gene_idx].astype(np.float32),
        "control_detection": (stats.n_detected[c_i][gene_idx] / n_c).astype(np.float32),
        "n_control_cells": int(n_c),
        "n_condition_batches_using_pooled_control": n_fallback,
    }
