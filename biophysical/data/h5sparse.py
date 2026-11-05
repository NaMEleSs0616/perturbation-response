"""Memory-bounded access to the X matrix of an H5AD file along its native storage axis.

scPerturb files may store X as CSC (gene-major) or CSR (cell-major). Row slicing a backed CSC
matrix reads the whole file per slice, so all heavy computations here iterate over contiguous
chunks of the *major* axis and accumulate per-group statistics, which works for either layout.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp


@dataclass
class XLayout:
    encoding: str  # "csr_matrix" | "csc_matrix" | "array"
    shape: tuple[int, int]
    nnz: int | None

    @property
    def major_axis(self) -> int:
        """0 if chunks are cells (CSR/dense), 1 if chunks are genes (CSC)."""
        return 1 if self.encoding == "csc_matrix" else 0


def x_layout(path: str | Path) -> XLayout:
    with h5py.File(path, "r") as f:
        X = f["X"]
        if isinstance(X, h5py.Dataset):
            return XLayout("array", tuple(X.shape), None)
        enc = X.attrs.get("encoding-type", X.attrs.get("h5sparse_format", ""))
        enc = enc.decode() if isinstance(enc, bytes) else str(enc)
        if enc in ("csr", "csc"):
            enc += "_matrix"
        shape = tuple(int(s) for s in X.attrs.get("shape", X.attrs.get("h5sparse_shape")))
        return XLayout(enc, shape, int(X["data"].shape[0]))


def iter_major_chunks(path: str | Path, chunk: int = 2000) -> Iterator[tuple[int, int, sp.spmatrix | np.ndarray]]:
    """Yield (start, stop, block) along the major axis.

    For CSR/dense, block has shape (stop-start, n_genes) as CSR. For CSC, block has shape
    (n_cells, stop-start) as CSC.
    """
    layout = x_layout(path)
    n_major = layout.shape[layout.major_axis]
    with h5py.File(path, "r") as f:
        X = f["X"]
        if layout.encoding == "array":
            for s in range(0, n_major, chunk):
                e = min(s + chunk, n_major)
                yield s, e, sp.csr_matrix(X[s:e])
            return
        indptr = X["indptr"][:]
        data, indices = X["data"], X["indices"]
        n_minor = layout.shape[1 - layout.major_axis]
        for s in range(0, n_major, chunk):
            e = min(s + chunk, n_major)
            a, b = int(indptr[s]), int(indptr[e])
            ip = indptr[s:e + 1] - a
            d = data[a:b].astype(np.float32, copy=False)
            ix = indices[a:b]
            if layout.encoding == "csr_matrix":
                yield s, e, sp.csr_matrix((d, ix, ip), shape=(e - s, n_minor))
            else:
                yield s, e, sp.csc_matrix((d, ix, ip), shape=(n_minor, e - s))


def sample_block(path: str | Path, n: int = 500) -> np.ndarray:
    """A small dense block from the start of the major axis (for cheap sanity checks)."""
    layout = x_layout(path)
    for _, _, block in iter_major_chunks(path, chunk=n if layout.major_axis == 0 else 200):
        dense = block.toarray()
        return dense[:n] if layout.major_axis == 0 else dense
    return np.zeros((0, 0), dtype=np.float32)


def cell_library_sizes(path: str | Path, chunk: int = 2000) -> np.ndarray:
    """Total counts per cell over all genes in the file."""
    layout = x_layout(path)
    n_cells = layout.shape[0]
    totals = np.zeros(n_cells, dtype=np.float64)
    for s, e, block in iter_major_chunks(path, chunk):
        if layout.major_axis == 0:
            totals[s:e] = np.asarray(block.sum(axis=1)).ravel()
        else:
            totals += np.asarray(block.sum(axis=1)).ravel()
    return totals
