"""Perturbation features built from gene-level representations.

A perturbation's feature vector is the sum of its target genes' vectors, plus each target's control
expression summary and a combination indicator. Representations must be computable without any
held-out perturbation response: the default (co-expression across control cells) meets this, as do
STRING-derived vectors and external pretrained embeddings.

In a single-context dataset the baseline expression profile is the same for every example, so
it carries no information; target-gene baseline expression is what varies, and is used here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Featurizer:
    vocab: list[str]
    matrices: dict[str, np.ndarray]     # representation name -> (n_genes, d)
    baseline: np.ndarray                # (n_genes, 2): control mean, control detection
    use_baseline: bool = True
    _pos: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._pos = {g: i for i, g in enumerate(self.vocab)}
        for name, m in self.matrices.items():
            if m.shape[0] != len(self.vocab):
                raise ValueError(f"representation '{name}' has {m.shape[0]} rows, expected {len(self.vocab)}")

    @property
    def gene_matrix(self) -> np.ndarray:
        parts = [m for m in self.matrices.values()]
        if self.use_baseline:
            parts.append(self.baseline)
        return np.concatenate(parts, axis=1).astype(np.float32)

    def supports(self, genes: tuple[str, ...]) -> bool:
        return all(g in self._pos for g in genes)

    def transform(self, cond_genes: list[tuple[str, ...]], covariates: np.ndarray | None = None) -> np.ndarray:
        G = self.gene_matrix
        X = np.zeros((len(cond_genes), G.shape[1] + 1), dtype=np.float32)
        for i, gs in enumerate(cond_genes):
            missing = [g for g in gs if g not in self._pos]
            if missing:
                raise KeyError(f"genes not in the feature vocabulary: {missing}")
            X[i, :-1] = G[[self._pos[g] for g in gs]].sum(axis=0)
            X[i, -1] = float(len(gs) > 1)
        if covariates is not None:
            X = np.concatenate([X, np.asarray(covariates, dtype=np.float32).reshape(len(cond_genes), -1)], axis=1)
        return X


def default_featurizer(data, extra: dict[str, np.ndarray] | None = None, use_coexpression: bool = True) -> Featurizer:
    matrices: dict[str, np.ndarray] = {}
    if use_coexpression:
        matrices["coexpression"] = data.gene_embedding
    if extra:
        matrices.update(extra)
    return Featurizer(vocab=list(data.embedding_genes), matrices=matrices, baseline=data.embedding_baseline)
