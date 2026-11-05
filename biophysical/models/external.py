"""Frozen external gene embeddings from a plain-text file (word2vec-style: GENE v1 v2 ...).

Only text is parsed, so an embedding file cannot execute code. Genes are matched by name first, then
through HGNC if a mapper is given. Genes with no vector get zeros and are counted in the report.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from biophysical.logging_utils import get_logger

log = get_logger(__name__)


def read_text_embeddings(path: str | Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    dim = None
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 3:
                continue  # word2vec header line, or a blank line
            try:
                vec = np.array(parts[1:], dtype=np.float32)
            except ValueError:
                continue
            if dim is None:
                dim = vec.size
            if vec.size == dim:
                out[parts[0]] = vec
    if not out:
        raise ValueError(f"no embeddings parsed from {path}")
    return out


def align_embeddings(table: dict[str, np.ndarray], genes: list[str], mapper=None) -> tuple[np.ndarray, dict]:
    """(n_genes, d) matrix in `genes` order, plus a coverage report. Ambiguous aliases are not used."""
    dim = next(iter(table.values())).size
    M = np.zeros((len(genes), dim), dtype=np.float32)
    direct, mapped, missing = 0, 0, []
    for i, g in enumerate(genes):
        if g in table:
            M[i] = table[g]
            direct += 1
            continue
        m = mapper.map(g) if mapper else None
        if m is not None and m.symbol and m.symbol in table:
            M[i] = table[m.symbol]
            mapped += 1
        else:
            missing.append(g)
    return M, {"n_genes": len(genes), "direct": direct, "via_hgnc": mapped, "missing": len(missing), "dim": dim}
