"""Grouped splits matched to specific generalisation claims.

unseen_gene:
    Claim: predict the response to activating a gene with no training data involving that gene.
    Genes are held out, and every condition containing a held-out gene (single or combination) is
    removed from training and used for testing, in strata:
      single_unseen, combo_one_unseen, combo_two_unseen
    A calibration set is built the same way from the remaining genes, so calibration errors
    measure the same kind of generalisation as test errors.

unseen_combo:
    Claim: predict a combination whose constituent single-gene responses were measured but whose
    combination was never measured. Only combinations with both singles present are tested.
    Singles stay in training.

Cell-level random splits are deliberately not offered: they would place cells from the same
perturbation on both sides and would not test either claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Fold:
    task: str
    fold: int
    train: np.ndarray
    calib: np.ndarray
    test: np.ndarray
    strata: dict[int, str] = field(default_factory=dict)
    heldout_genes: list[str] = field(default_factory=list)
    calib_genes: list[str] = field(default_factory=list)


class SplitLeakageError(AssertionError):
    pass


def unseen_gene_folds(cond_genes: list[tuple[str, ...]], n_folds: int = 5, calibration_fraction: float = 0.15, seed: int = 0) -> list[Fold]:
    all_genes = np.array(sorted({g for gs in cond_genes for g in gs}))
    if all_genes.size < n_folds:
        raise ValueError(f"cannot make {n_folds} folds from {all_genes.size} genes")
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(all_genes), n_folds)
    folds = []
    for k, held in enumerate(parts):
        H = set(held.tolist())
        rest = np.array(sorted(set(all_genes.tolist()) - H))
        n_cal = max(1, int(round(calibration_fraction * rest.size)))
        C = set(rng.choice(rest, size=n_cal, replace=False).tolist())
        train, calib, test, strata = [], [], [], {}
        for i, gs in enumerate(cond_genes):
            n_held = sum(g in H for g in gs)
            if n_held:
                test.append(i)
                strata[i] = "single_unseen" if len(gs) == 1 else ("combo_two_unseen" if n_held == 2 else "combo_one_unseen")
            elif any(g in C for g in gs):
                calib.append(i)
            else:
                train.append(i)
        folds.append(Fold("unseen_gene", k, np.array(train, dtype=int), np.array(calib, dtype=int), np.array(test, dtype=int),
                          strata, sorted(H), sorted(C)))
    return folds


def unseen_combo_folds(cond_genes: list[tuple[str, ...]], n_folds: int = 5, calibration_fraction: float = 0.15, seed: int = 0) -> list[Fold]:
    singles = {gs[0] for gs in cond_genes if len(gs) == 1}
    eligible = np.array([i for i, gs in enumerate(cond_genes) if len(gs) == 2 and all(g in singles for g in gs)], dtype=int)
    if eligible.size < n_folds:
        raise ValueError(f"cannot make {n_folds} folds from {eligible.size} eligible combinations")
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(eligible), n_folds)
    folds = []
    for k, test in enumerate(parts):
        test_set = set(test.tolist())
        rest = np.array([i for i in eligible.tolist() if i not in test_set], dtype=int)
        n_cal = max(1, int(round(calibration_fraction * rest.size)))
        calib = np.sort(rng.choice(rest, size=n_cal, replace=False))
        cal_set = set(calib.tolist())
        train = np.array([i for i in range(len(cond_genes)) if i not in test_set and i not in cal_set], dtype=int)
        folds.append(Fold("unseen_combo", k, train, calib, np.sort(test), {int(i): "combo_seen_singles" for i in test}))
    return folds


def assert_no_leakage(fold: Fold, cond_genes: list[tuple[str, ...]]) -> None:
    """Raise SplitLeakageError if a fold violates its generalisation claim."""
    sets = {"train": set(fold.train.tolist()), "calib": set(fold.calib.tolist()), "test": set(fold.test.tolist())}
    for a, b in [("train", "calib"), ("train", "test"), ("calib", "test")]:
        if sets[a] & sets[b]:
            raise SplitLeakageError(f"{fold.task} fold {fold.fold}: {a} and {b} share conditions")
    profile = [frozenset(gs) for gs in cond_genes]
    train_profiles = {profile[i] for i in fold.train}
    for i in list(fold.test) + list(fold.calib):
        if profile[i] in train_profiles:
            raise SplitLeakageError(f"{fold.task} fold {fold.fold}: gene set {set(profile[i])} is duplicated in training")
    if fold.task == "unseen_gene":
        H, C = set(fold.heldout_genes), set(fold.calib_genes)
        train_genes = {g for i in fold.train for g in cond_genes[i]}
        if train_genes & H:
            raise SplitLeakageError(f"held-out genes in training: {sorted(train_genes & H)}")
        if train_genes & C:
            raise SplitLeakageError(f"calibration genes in training: {sorted(train_genes & C)}")
    elif fold.task == "unseen_combo":
        train_singles = {cond_genes[i][0] for i in fold.train if len(cond_genes[i]) == 1}
        for i in fold.test:
            if not set(cond_genes[i]) <= train_singles:
                raise SplitLeakageError(f"test combination {cond_genes[i]} lacks a measured single in training")


def folds_table(folds: list[Fold], conditions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in folds:
        for role, idx in (("train", f.train), ("calib", f.calib), ("test", f.test)):
            for i in idx:
                rows.append({"task": f.task, "fold": f.fold, "role": role, "condition": conditions["condition"].iloc[int(i)],
                             "stratum": f.strata.get(int(i), "")})
    return pd.DataFrame(rows)
