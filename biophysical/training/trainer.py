"""Generic regression training loop with early stopping and optional checkpointing."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-3
    max_epochs: int = 400
    patience: int = 40
    batch_size: int = 32
    loss: str = "huber"
    checkpoint_path: str | None = None


def _loss_fn(name: str) -> nn.Module:
    if name == "huber":
        return nn.HuberLoss(delta=1.0)
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"unknown loss: {name}")


def train_regressor(net: nn.Module, X: np.ndarray, Y: np.ndarray, val: tuple[np.ndarray, np.ndarray] | None, cfg: TrainConfig, seed: int = 0) -> dict:
    """Train `net` in place; best-validation weights are restored when a validation set is given."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = _loss_fn(cfg.loss)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    Xv = Yv = None
    if val is not None:
        Xv, Yv = torch.from_numpy(val[0]), torch.from_numpy(val[1])
    best, best_state, wait, train_curve, val_curve = np.inf, None, 0, [], []
    epoch = 0
    for epoch in range(1, cfg.max_epochs + 1):
        net.train()
        order = rng.permutation(len(X))
        total = 0.0
        for s in range(0, len(X), cfg.batch_size):
            b = order[s:s + cfg.batch_size]
            opt.zero_grad()
            loss = loss_fn(net(Xt[b]), Yt[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        train_curve.append(total / len(X))
        if Xv is not None:
            net.eval()
            with torch.no_grad():
                v = loss_fn(net(Xv), Yv).item()
            val_curve.append(v)
            if v < best - 1e-6:
                best, best_state, wait = v, copy.deepcopy(net.state_dict()), 0
            else:
                wait += 1
                if wait >= cfg.patience:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    if cfg.checkpoint_path:
        Path(cfg.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), cfg.checkpoint_path)
    net.eval()
    return {"epochs": epoch, "best_val": None if best == np.inf else best, "final_train": train_curve[-1] if train_curve else None}
