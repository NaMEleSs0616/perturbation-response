"""Small PyTorch perturbation-response model, trained as a seeded bootstrap ensemble.

Input: perturbation features plus any available covariates. In single-context data, target-gene
baseline expression (part of the features) carries the varying baseline information.
Output: a response in gene space, parameterised as a base model plus a residual predicted in a PCA
basis fitted on training residuals only. This keeps the output low-dimensional for the small
number of training perturbations.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from biophysical.logging_utils import get_logger
from biophysical.models.baselines import AdditiveModel, FitData, MeanModel, _ResidualModel
from biophysical.training.trainer import TrainConfig, train_regressor

log = get_logger(__name__)


@dataclass
class MLPConfig:
    n_members: int = 5
    bootstrap: bool = True
    hidden: list[int] = field(default_factory=lambda: [128, 64])
    n_components: int = 20
    dropout: float = 0.1
    weight_decay: float = 1e-3
    lr: float = 1e-3
    max_epochs: int = 400
    patience: int = 40
    batch_size: int = 32
    loss: str = "huber"
    seed: int = 0


class ResponseNet(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPEnsemble(_ResidualModel):
    name = "mlp_ensemble"

    def __init__(self, config: MLPConfig | None = None, base_kind: str = "mean", name: str | None = None):
        self.config, self.base_kind = config or MLPConfig(), base_kind
        if name:
            self.name = name
        self.members: list[ResponseNet] = []
        self.history: list[dict] = []

    def fit(self, train: FitData, calib: FitData | None = None):
        cfg = self.config
        R = self._fit_base(train)
        self.featurizer = train.featurizer
        X = self.featurizer.transform(train.cond_genes, train.covariates)
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X).astype(np.float32)

        self.r_mean = R.mean(axis=0)
        k = min(cfg.n_components, min(R.shape) - 1)
        _, s, vt = np.linalg.svd(R - self.r_mean, full_matrices=False)
        self.components = vt[:k].astype(np.float32)
        self.score_scale = float(s[0] / np.sqrt(max(R.shape[0] - 1, 1))) or 1.0
        Z = ((R - self.r_mean) @ self.components.T / self.score_scale).astype(np.float32)

        val = None
        if calib is not None and len(calib.cond_genes):
            Xc = self.scaler.transform(self.featurizer.transform(calib.cond_genes, calib.covariates)).astype(np.float32)
            val = (Xc, ((self._residual(calib) - self.r_mean) @ self.components.T / self.score_scale).astype(np.float32))

        self.members, self.history = [], []
        tc = TrainConfig(lr=cfg.lr, weight_decay=cfg.weight_decay, max_epochs=cfg.max_epochs, patience=cfg.patience,
                         batch_size=cfg.batch_size, loss=cfg.loss)
        for m in range(cfg.n_members):
            seed = cfg.seed + 1000 * m
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, len(Xs), len(Xs)) if cfg.bootstrap else np.arange(len(Xs))
            net = ResponseNet(Xs.shape[1], Z.shape[1], list(cfg.hidden), cfg.dropout)
            hist = train_regressor(net, Xs[idx], Z[idx], val, tc, seed=seed)
            self.members.append(net.eval())
            self.history.append(hist)
        return self

    def predict_members(self, cond_genes, covariates=None) -> np.ndarray:
        """(n_members, n, n_genes) predictions."""
        X = self.scaler.transform(self.featurizer.transform(cond_genes, covariates)).astype(np.float32)
        base = self.base.predict(cond_genes)
        out = []
        with torch.no_grad():
            for net in self.members:
                z = net(torch.from_numpy(X)).numpy()
                out.append(base + self.r_mean + z * self.score_scale @ self.components)
        return np.stack(out).astype(np.float32)

    def predict(self, cond_genes, covariates=None):
        return self.predict_members(cond_genes, covariates).mean(axis=0)

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": 1,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config": asdict(self.config),
            "base_kind": self.base_kind,
            "name": self.name,
            "n_in": len(self.scaler.mean_),
            "n_out": self.components.shape[0],
            "state_dicts": [m.state_dict() for m in self.members],
            "scaler_mean": self.scaler.mean_,
            "scaler_scale": self.scaler.scale_,
            "components": self.components,
            "r_mean": self.r_mean,
            "score_scale": self.score_scale,
            "base_state": _base_state(self.base),
            "history": self.history,
            "metadata": metadata or {},
        }, path)

    @classmethod
    def load(cls, path: str | Path, featurizer) -> "MLPEnsemble":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self = cls(MLPConfig(**ckpt["config"]), base_kind=ckpt["base_kind"], name=ckpt["name"])
        self.featurizer = featurizer
        self.scaler = StandardScaler()
        self.scaler.mean_, self.scaler.scale_ = ckpt["scaler_mean"], ckpt["scaler_scale"]
        self.scaler.var_ = self.scaler.scale_ ** 2
        self.scaler.n_features_in_ = len(self.scaler.mean_)
        self.components, self.r_mean, self.score_scale = ckpt["components"], ckpt["r_mean"], ckpt["score_scale"]
        self.base = _restore_base(ckpt["base_state"])
        self.history = ckpt["history"]
        self.metadata = ckpt["metadata"]
        self.members = []
        for sd in ckpt["state_dicts"]:
            net = ResponseNet(ckpt["n_in"], ckpt["n_out"], list(self.config.hidden), self.config.dropout)
            net.load_state_dict(sd)
            self.members.append(net.eval())
        return self


def _base_state(base) -> dict:
    if isinstance(base, AdditiveModel):
        return {"kind": "additive", "singles": base.singles, "fallback": _base_state(base.fallback)}
    return {"kind": "mean", "overall": base.overall, "by_size": base.by_size}


def _restore_base(state: dict):
    if state["kind"] == "additive":
        m = AdditiveModel()
        m.singles, m.fallback = state["singles"], _restore_base(state["fallback"])
        return m
    m = MeanModel()
    m.overall, m.by_size = state["overall"], state["by_size"]
    return m
