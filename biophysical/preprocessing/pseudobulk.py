"""Chunked, memory-bounded computation of per-group expression statistics.

Works on the native storage axis of X (see ``biophysical.data.h5sparse``) so the expression
matrix is never fully loaded. All statistics are for log1p(CP-target_sum) values.

Group definitions:
  * condition:  one per condition, used for the response target and measurement error
  * replicate:  condition x replicate column (e.g. 10x lane), used for matched-control deltas
  * half:       condition x random half, used for split-half noise ceilings. Control cells are
                split as well, so the two half-deltas do not share control noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from biophysical.data.h5sparse import cell_library_sizes, iter_major_chunks, x_layout
from biophysical.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class GroupStats:
    """Statistics over all genes in the file.

    ``sum_log`` is (n_groups, n_genes). The other arrays are (n_conditions, n_genes), aligned with
    the rows of ``groups`` where kind == "condition", which keeps memory bounded on a laptop.
    """
    groups: pd.DataFrame  # columns: kind, condition, level, n_cells
    sum_log: np.ndarray
    sum_sq_log: np.ndarray
    sum_norm: np.ndarray
    n_detected: np.ndarray

    def means(self, kind: str) -> tuple[pd.DataFrame, np.ndarray]:
        m = (self.groups["kind"] == kind).to_numpy()
        table = self.groups[m].reset_index(drop=True)
        return table, self.sum_log[m] / np.maximum(table["n_cells"].to_numpy()[:, None], 1)


def build_groups(obs: pd.DataFrame, condition_col: str, replicate_col: str | None, control_label: str = "control",
                 seed: int = 0) -> tuple[pd.DataFrame, sp.csr_matrix]:
    """Return the group table and a (n_cells, n_groups) binary membership matrix.

    Cells whose condition is NaN belong to no group (they are excluded and counted upstream).
    Group kinds: condition, half, replicate, and control_split (control cells by replicate and
    random half; used to give training and evaluation targets disjoint control cells).
    """
    cond = obs[condition_col].astype(object).where(obs[condition_col].notna(), None).to_numpy()
    halves = np.random.default_rng(seed).integers(0, 2, size=len(obs))
    rep = obs[replicate_col].astype(str).to_numpy() if replicate_col else np.array(["all"] * len(obs))

    index: dict[tuple[str, str, str], int] = {}
    rows, cols = [], []

    def add(i: int, kind: str, condition: str, level: str) -> None:
        j = index.setdefault((kind, condition, level), len(index))
        rows.append(i)
        cols.append(j)

    for i, c in enumerate(cond):
        if c is None:
            continue
        add(i, "condition", c, "all")
        add(i, "half", c, str(int(halves[i])))
        add(i, "replicate", c, rep[i])
        if c == control_label:
            add(i, "control_split", c, f"{rep[i]}|{int(halves[i])}")

    membership = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(obs), len(index)))
    groups = pd.DataFrame([{"kind": k, "condition": c, "level": l} for (k, c, l) in index])
    groups["n_cells"] = np.asarray(membership.sum(axis=0)).ravel().astype(int)
    return groups, membership


def _lognorm_csc(block: sp.csc_matrix, size_factors: np.ndarray) -> tuple[sp.csc_matrix, sp.csc_matrix]:
    norm = block.copy()
    norm.data = norm.data.astype(np.float32) * size_factors[norm.indices].astype(np.float32)
    log_ = norm.copy()
    log_.data = np.log1p(log_.data)
    return norm, log_


def compute_group_stats(path: str | Path, groups: pd.DataFrame, membership: sp.csr_matrix, target_sum: float = 1e4, chunk: int = 1000) -> GroupStats:
    layout = x_layout(path)
    n_cells, n_genes = layout.shape
    if membership.shape[0] != n_cells:
        raise ValueError(f"membership has {membership.shape[0]} rows but X has {n_cells} cells")

    totals = cell_library_sizes(path, chunk=chunk * 2)
    if (totals <= 0).any():
        log.warning("%d cells with zero counts; they contribute zeros to their groups", int((totals <= 0).sum()))
    size_factors = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)

    cond_idx = np.flatnonzero((groups["kind"] == "condition").to_numpy())
    n_groups, n_cond = membership.shape[1], cond_idx.size
    M = membership.tocsc()
    M_cond = M[:, cond_idx]

    # float64 accumulators would double memory; float32 relative error on these sums is ~1e-6
    sum_log = np.zeros((n_groups, n_genes), dtype=np.float32)
    sum_sq = np.zeros((n_cond, n_genes), dtype=np.float32)
    sum_norm = np.zeros((n_cond, n_genes), dtype=np.float32)
    n_det = np.zeros((n_cond, n_genes), dtype=np.float32)

    def stats(block_csc: sp.csc_matrix, sf: np.ndarray, m: sp.spmatrix, m_cond: sp.spmatrix):
        norm, log_ = _lognorm_csc(block_csc, sf)
        sq = log_.copy()
        sq.data = sq.data ** 2
        det = log_.copy()
        det.data = (det.data > 0).astype(np.float32)
        return ((m.T @ log_).toarray(), (m_cond.T @ sq).toarray(), (m_cond.T @ norm).toarray(), (m_cond.T @ det).toarray())

    if layout.major_axis == 1:
        for s, e, block in iter_major_chunks(path, chunk=chunk):
            sum_log[:, s:e], sum_sq[:, s:e], sum_norm[:, s:e], n_det[:, s:e] = stats(block, size_factors, M, M_cond)
    else:
        M_csr, M_cond_csr = membership, membership[:, cond_idx]
        for s, e, block in iter_major_chunks(path, chunk=chunk * 5):
            a, b, c, d = stats(block.tocsc(), size_factors[s:e], M_csr[s:e], M_cond_csr[s:e])
            sum_log += a.astype(np.float32)
            sum_sq += b.astype(np.float32)
            sum_norm += c.astype(np.float32)
            n_det += d.astype(np.float32)
    return GroupStats(groups=groups, sum_log=sum_log, sum_sq_log=sum_sq, sum_norm=sum_norm, n_detected=n_det)


def extract_dense_cells(path: str | Path, cell_idx: np.ndarray, gene_idx: np.ndarray, target_sum: float = 1e4, chunk: int = 1000) -> np.ndarray:
    """Dense log1p-normalised expression for selected cells and genes, in the given order."""
    layout = x_layout(path)
    totals = cell_library_sizes(path, chunk=chunk * 2)
    size_factors = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
    out = np.zeros((len(cell_idx), len(gene_idx)), dtype=np.float32)
    cell_pos = -np.ones(layout.shape[0], dtype=np.int64)
    cell_pos[cell_idx] = np.arange(len(cell_idx))
    gene_pos = -np.ones(layout.shape[1], dtype=np.int64)
    gene_pos[gene_idx] = np.arange(len(gene_idx))
    for s, e, block in iter_major_chunks(path, chunk=chunk):
        coo = block.tocoo()
        if layout.major_axis == 1:
            cells, genes = coo.row, coo.col + s
        else:
            cells, genes = coo.row + s, coo.col
        keep = (cell_pos[cells] >= 0) & (gene_pos[genes] >= 0)
        if keep.any():
            vals = np.log1p(coo.data[keep] * size_factors[cells[keep]])
            out[cell_pos[cells[keep]], gene_pos[genes[keep]]] = vals
    return out


def select_genes(mean_log: np.ndarray, var_log: np.ndarray, detection: np.ndarray, gene_set: str, n_genes: int,
                 min_detection: float = 0.01, n_bins: int = 20) -> np.ndarray:
    """Pick genes using statistics from CONTROL cells only, so no perturbation label can leak in.

    hvg: dispersion (var/mean of log-normalised values), z-scored within mean-expression bins.
    top_expressed: highest mean expression.
    """
    eligible = np.flatnonzero(detection >= min_detection)
    if eligible.size == 0:
        raise ValueError("no genes pass the control-cell detection threshold")
    if gene_set == "top_expressed":
        order = eligible[np.argsort(-mean_log[eligible], kind="stable")]
        return np.sort(order[:n_genes])
    if gene_set != "hvg":
        raise ValueError(f"unknown gene_set: {gene_set}")
    disp = var_log[eligible] / np.maximum(mean_log[eligible], 1e-8)
    edges = np.unique(np.quantile(mean_log[eligible], np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.digitize(mean_log[eligible], edges[1:-1]), 0, max(len(edges) - 2, 0))
    z = np.zeros_like(disp)
    for b in np.unique(bins):
        m = bins == b
        sd = disp[m].std()
        z[m] = (disp[m] - disp[m].mean()) / (sd if sd > 0 else 1.0)
    order = eligible[np.argsort(-z, kind="stable")]
    return np.sort(order[:n_genes])
