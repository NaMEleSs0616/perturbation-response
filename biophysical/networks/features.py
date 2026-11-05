"""Gene representations derived from a STRING adjacency matrix.

These use only the network and control-cell features, so they contain no perturbation response
information. Improved prediction would show the graph is useful as a predictor; it would not show
that its edges are causal or direct.
"""
from __future__ import annotations

import numpy as np


def neighbour_average(A: np.ndarray, E: np.ndarray) -> np.ndarray:
    """One-hop weighted average of gene features; genes with no neighbours get zeros."""
    deg = A.sum(axis=1, keepdims=True)
    return (A @ E / np.where(deg > 0, deg, 1.0)).astype(np.float32)


def spectral_embedding(A: np.ndarray, dim: int = 32, clip: float = 5.0) -> np.ndarray:
    """Top non-trivial eigenvectors of the symmetric normalised adjacency, standardised and clipped."""
    deg = A.sum(axis=1)
    d = np.where(deg > 0, 1.0 / np.sqrt(np.where(deg > 0, deg, 1.0)), 0.0)
    S = (A * d[:, None]) * d[None, :]
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(-vals)[1:dim + 1]
    E = vecs[:, order] * vals[order]
    # eigenvectors can localise on very small graph components, which puts a few genes at extreme values
    # and destabilises regression; standardise per dimension and clip those values
    sd = E.std(axis=0)
    Z = (E - E.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    return np.clip(Z, -clip, clip).astype(np.float32)


def graph_summary(A: np.ndarray, genes: list[str], targets: list[str]) -> dict:
    pos = {g: i for i, g in enumerate(genes)}
    deg = (A > 0).sum(axis=1)
    t_deg = np.array([deg[pos[g]] for g in targets if g in pos])
    return {"n_genes": len(genes), "n_edges": int((A > 0).sum() // 2), "n_isolated_genes": int((deg == 0).sum()),
            "n_targets": int(t_deg.size), "n_targets_isolated": int((t_deg == 0).sum()), "median_target_degree": float(np.median(t_deg)) if t_deg.size else 0.0}
