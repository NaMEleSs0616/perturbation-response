"""Baseline models. All share the fit/predict interface in ``FitData`` so comparisons use the same
targets and the same test conditions."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from biophysical.models.features import Featurizer


@dataclass
class FitData:
    cond_genes: list[tuple[str, ...]]
    Y: np.ndarray            # (n, G) response
    mask: np.ndarray         # (n, G) True where the gene is a target of that condition
    featurizer: Featurizer
    gene_index: dict[str, int]
    covariates: np.ndarray | None = None

    def subset(self, idx: np.ndarray) -> "FitData":
        return FitData([self.cond_genes[i] for i in idx], self.Y[idx], self.mask[idx], self.featurizer, self.gene_index,
                       None if self.covariates is None else self.covariates[idx])


def masked_mse(Y: np.ndarray, P: np.ndarray, mask: np.ndarray) -> float:
    return float(((Y - P) ** 2)[~mask].mean())


class PerturbationModel:
    name = "model"

    def fit(self, train: FitData, calib: FitData | None = None) -> "PerturbationModel":
        raise NotImplementedError

    def predict(self, cond_genes: list[tuple[str, ...]], covariates: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError


class ZeroModel(PerturbationModel):
    """Predicts no change. Pearson correlation is undefined for it and is reported as NaN."""
    name = "zero"

    def fit(self, train, calib=None):
        self.n_genes = train.Y.shape[1]
        return self

    def predict(self, cond_genes, covariates=None):
        return np.zeros((len(cond_genes), self.n_genes), dtype=np.float32)


class MeanModel(PerturbationModel):
    """Mean training response, computed separately for singles and combinations when both exist."""
    name = "mean"
    min_group = 3

    def fit(self, train, calib=None):
        Y = np.where(train.mask, np.nan, train.Y)
        sizes = np.array([len(g) for g in train.cond_genes])
        self.by_size = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a gene masked in every condition has mean 0
            self.overall = np.nan_to_num(np.nanmean(Y, axis=0)).astype(np.float32)
            for s in np.unique(sizes):
                if (sizes == s).sum() >= self.min_group:
                    self.by_size[int(s)] = np.nan_to_num(np.nanmean(Y[sizes == s], axis=0)).astype(np.float32)
        return self

    def predict(self, cond_genes, covariates=None):
        return np.stack([self.by_size.get(len(g), self.overall) for g in cond_genes]).astype(np.float32)


class AdditiveModel(PerturbationModel):
    """Sum of measured single-gene responses. A constituent with no measured single contributes the
    mean single response instead, so this model is only a pure additive prediction when both singles
    are in training (the unseen_combo task)."""
    name = "additive"

    def fit(self, train, calib=None):
        self.fallback = MeanModel().fit(train)
        self.singles = {}
        for gs, y, m in zip(train.cond_genes, train.Y, train.mask):
            if len(gs) == 1:
                self.singles[gs[0]] = np.where(m, 0.0, y).astype(np.float32)
        return self

    def predict(self, cond_genes, covariates=None):
        fb = self.fallback.by_size.get(1, self.fallback.overall)
        out = np.zeros((len(cond_genes), fb.size), dtype=np.float32)
        for i, gs in enumerate(cond_genes):
            for g in gs:
                out[i] += self.singles.get(g, fb)
        return out


class _ResidualModel(PerturbationModel):
    """Learns the residual from a base model (mean or additive), with target self-entries zeroed."""
    base_kind = "mean"

    def _fit_base(self, train: FitData) -> np.ndarray:
        self.base = (AdditiveModel() if self.base_kind == "additive" else MeanModel()).fit(train)
        return self._residual(train)

    def _residual(self, d: FitData) -> np.ndarray:
        return np.where(d.mask, 0.0, d.Y - self.base.predict(d.cond_genes)).astype(np.float32)


class RidgeModel(_ResidualModel):
    """Ridge regression from perturbation features to response; alpha selected on the calibration set."""
    name = "ridge"

    def __init__(self, alphas=(0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0), base_kind: str = "mean", name: str | None = None):
        self.alphas, self.base_kind = tuple(alphas), base_kind
        if name:
            self.name = name

    def fit(self, train, calib=None):
        R = self._fit_base(train)
        self.featurizer = train.featurizer
        X = self.featurizer.transform(train.cond_genes)
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.alpha = self.alphas[len(self.alphas) // 2]
        if calib is not None and len(calib.cond_genes):
            Xc, Rc = self.scaler.transform(self.featurizer.transform(calib.cond_genes)), self._residual(calib)
            scores = [(masked_mse(Rc, Ridge(alpha=a).fit(Xs, R).predict(Xc), calib.mask), a) for a in self.alphas]
            self.alpha = min(scores)[1]
        self.model = Ridge(alpha=self.alpha).fit(Xs, R)
        return self

    def predict(self, cond_genes, covariates=None):
        X = self.scaler.transform(self.featurizer.transform(cond_genes))
        return (self.base.predict(cond_genes) + self.model.predict(X)).astype(np.float32)


class KNNModel(_ResidualModel):
    """Averages the responses of the k nearest training perturbations with the same number of targets."""
    name = "knn"

    def __init__(self, ks=(1, 3, 5, 10), base_kind: str = "mean", name: str | None = None):
        self.ks, self.base_kind = tuple(ks), base_kind
        if name:
            self.name = name

    def fit(self, train, calib=None):
        self.R = self._fit_base(train)
        self.featurizer = train.featurizer
        X = self.featurizer.transform(train.cond_genes)
        self.scaler = StandardScaler().fit(X)
        self.X = self.scaler.transform(X)
        self.sizes = np.array([len(g) for g in train.cond_genes])
        self.k = self.ks[0]
        if calib is not None and len(calib.cond_genes):
            Rc = self._residual(calib)
            scores = [(masked_mse(Rc, self._neighbours(calib.cond_genes, k), calib.mask), k) for k in self.ks]
            self.k = min(scores)[1]
        return self

    def _neighbours(self, cond_genes, k: int) -> np.ndarray:
        Xq = self.scaler.transform(self.featurizer.transform(cond_genes))
        out = np.zeros((len(cond_genes), self.R.shape[1]), dtype=np.float32)
        for i, gs in enumerate(cond_genes):
            pool = np.flatnonzero(self.sizes == len(gs))
            if pool.size == 0:
                pool = np.arange(self.X.shape[0])
            d = np.linalg.norm(self.X[pool] - Xq[i], axis=1)
            out[i] = self.R[pool[np.argsort(d, kind="stable")[:k]]].mean(axis=0)
        return out

    def predict(self, cond_genes, covariates=None):
        return (self.base.predict(cond_genes) + self._neighbours(cond_genes, self.k)).astype(np.float32)


class GeneLoadingRidge(_ResidualModel):
    """Bilinear ridge in the style of Ahlmann-Eltze, Huber & Anders (Nat Methods 2025).

    Target genes are represented by their loadings in a PCA of the TRAINING response matrix, i.e. how
    the target gene itself responds across other perturbations. The held-out gene's row is measured
    under other perturbations, which is available information, but the fold's test conditions are
    not used to compute it.
    """
    name = "gene_loading_ridge"

    def __init__(self, n_components: int = 10, alphas=(0.1, 1.0, 10.0, 100.0, 1000.0), base_kind: str = "mean", name: str | None = None):
        self.n_components, self.alphas, self.base_kind = n_components, tuple(alphas), base_kind
        if name:
            self.name = name

    def _features(self, cond_genes) -> np.ndarray:
        X = np.zeros((len(cond_genes), self.loadings.shape[1]), dtype=np.float32)
        for i, gs in enumerate(cond_genes):
            for g in gs:
                j = self.gene_index.get(g)
                if j is not None:
                    X[i] += self.loadings[j]
        return X

    def fit(self, train, calib=None):
        R = self._fit_base(train)
        self.gene_index = train.gene_index
        k = min(self.n_components, min(R.shape) - 1)
        _, s, vt = np.linalg.svd(R, full_matrices=False)
        self.loadings = (vt[:k].T * s[:k]).astype(np.float32)
        Xs = self._features(train.cond_genes)
        self.alpha = self.alphas[len(self.alphas) // 2]
        if calib is not None and len(calib.cond_genes):
            Xc, Rc = self._features(calib.cond_genes), self._residual(calib)
            scores = [(masked_mse(Rc, Ridge(alpha=a).fit(Xs, R).predict(Xc), calib.mask), a) for a in self.alphas]
            self.alpha = min(scores)[1]
        self.model = Ridge(alpha=self.alpha).fit(Xs, R)
        return self

    def predict(self, cond_genes, covariates=None):
        return (self.base.predict(cond_genes) + self.model.predict(self._features(cond_genes))).astype(np.float32)


class BootstrapRidgeEnsemble(_ResidualModel):
    """Ridge members fitted on bootstrap resamples of perturbations; member spread is a cheap
    disagreement signal. Used for retrospective acquisition, where many refits are needed."""
    name = "bootstrap_ridge"

    def __init__(self, n_members: int = 8, alphas=(1.0, 10.0, 100.0, 1000.0, 10000.0), base_kind: str = "mean", seed: int = 0, name: str | None = None):
        self.n_members, self.alphas, self.base_kind, self.seed = n_members, tuple(alphas), base_kind, seed
        if name:
            self.name = name

    def fit(self, train, calib=None):
        R = self._fit_base(train)
        self.featurizer = train.featurizer
        X = self.featurizer.transform(train.cond_genes)
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.alpha = self.alphas[len(self.alphas) // 2]
        if calib is not None and len(calib.cond_genes):
            Xc, Rc = self.scaler.transform(self.featurizer.transform(calib.cond_genes)), self._residual(calib)
            scores = [(masked_mse(Rc, Ridge(alpha=a).fit(Xs, R).predict(Xc), calib.mask), a) for a in self.alphas]
            self.alpha = min(scores)[1]
        rng = np.random.default_rng(self.seed)
        self.members = []
        for _ in range(self.n_members):
            idx = rng.integers(0, len(Xs), len(Xs))
            self.members.append(Ridge(alpha=self.alpha).fit(Xs[idx], R[idx]))
        return self

    def predict_members(self, cond_genes, covariates=None) -> np.ndarray:
        X = self.scaler.transform(self.featurizer.transform(cond_genes))
        base = self.base.predict(cond_genes)
        return np.stack([base + m.predict(X) for m in self.members]).astype(np.float32)

    def predict(self, cond_genes, covariates=None):
        return self.predict_members(cond_genes).mean(axis=0)


class ScaledAdditive(PerturbationModel):
    """Additive prediction times one global factor fitted on training pairs (captures saturation)."""
    name = "scaled_additive"

    def fit(self, train, calib=None):
        self.additive = AdditiveModel().fit(train)
        self.scale = 1.0
        pairs = np.array([i for i, g in enumerate(train.cond_genes) if len(g) == 2], dtype=int)
        if pairs.size:
            P = self.additive.predict([train.cond_genes[i] for i in pairs])
            keep = ~train.mask[pairs]
            denom = float((P[keep] ** 2).sum())
            self.scale = float((P[keep] * train.Y[pairs][keep]).sum()) / denom if denom > 0 else 1.0
        return self

    def predict(self, cond_genes, covariates=None):
        return self.additive.predict(cond_genes) * np.array([self.scale if len(g) == 2 else 1.0 for g in cond_genes], dtype=np.float32)[:, None]


class AdditivePlusMeanResidual(PerturbationModel):
    """Additive prediction plus the gene-wise mean of (pair - additive) over training pairs. Captures any
    deviation from additivity shared by all pairs, with no information about the specific pair."""
    name = "additive_plus_mean_residual"

    def fit(self, train, calib=None):
        self.additive = AdditiveModel().fit(train)
        pairs = np.array([i for i, g in enumerate(train.cond_genes) if len(g) == 2], dtype=int)
        self.offset = np.zeros(train.Y.shape[1], dtype=np.float32)
        if pairs.size:
            R = train.Y[pairs] - self.additive.predict([train.cond_genes[i] for i in pairs])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self.offset = np.nan_to_num(np.nanmean(np.where(train.mask[pairs], np.nan, R), axis=0)).astype(np.float32)
        return self

    def predict(self, cond_genes, covariates=None):
        return self.additive.predict(cond_genes) + np.array([1.0 if len(g) == 2 else 0.0 for g in cond_genes], dtype=np.float32)[:, None] * self.offset


class SinglesOnly(PerturbationModel):
    """Trains and selects hyperparameters on single-gene conditions only."""

    def __init__(self, inner: PerturbationModel, name: str):
        self.inner, self.name = inner, name

    def fit(self, train, calib=None):
        keep = lambda d: d.subset(np.array([i for i, g in enumerate(d.cond_genes) if len(g) == 1], dtype=int))
        self.inner.fit(keep(train), keep(calib) if calib is not None else None)
        return self

    def predict(self, cond_genes, covariates=None):
        return self.inner.predict(cond_genes, covariates)

    def __getattr__(self, item):
        if item in ("predict_members", "alpha", "history"):
            return getattr(self.inner, item)
        raise AttributeError(item)
