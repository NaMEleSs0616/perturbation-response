"""Input support checks. A prediction is returned only for inputs the model can defensibly address."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STATUS_OK, STATUS_WARN, STATUS_ABSTAIN = "ok", "warning", "abstain"


@dataclass
class SupportAssessment:
    status: str
    warnings: list[str] = field(default_factory=list)
    nn_distance: float | None = None
    nearest_condition: str | None = None
    support_level: str | None = None


@dataclass
class SupportModel:
    """Knows the supported context and how far a query lies from the measured perturbations."""
    cell_context: str
    perturbation_type: str
    max_perturbed_genes: int
    vocab: set[str]
    measured: dict[frozenset, str]
    measured_singles: set[str]
    gene_detection: dict[str, float]
    X_train: np.ndarray                 # standardised features of measured conditions
    train_names: list[str]
    nn_threshold: float                 # 95th percentile of leave-one-out nearest-neighbour distance
    min_detection: float = 0.05

    @staticmethod
    def loo_nn_threshold(X: np.ndarray, quantile: float = 0.95) -> float:
        d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        return float(np.quantile(d.min(axis=1), quantile))

    def assess(self, genes: tuple[str, ...], x: np.ndarray | None = None, perturbation_type: str | None = None, cell_context: str | None = None) -> SupportAssessment:
        out = SupportAssessment(status=STATUS_OK)

        def abstain(msg: str) -> SupportAssessment:
            out.status = STATUS_ABSTAIN
            out.warnings.append(msg)
            return out

        if cell_context is None:
            return abstain(f"cell_context is required; the model only supports {self.cell_context}")
        if cell_context.strip().lower() != self.cell_context.lower():
            return abstain(f"unsupported cell context '{cell_context}'; training data contain only {self.cell_context}")
        if perturbation_type is None:
            return abstain(f"perturbation_type is required; the model only supports {self.perturbation_type}")
        if perturbation_type.strip().lower() != self.perturbation_type.lower():
            return abstain(f"unsupported perturbation type '{perturbation_type}'; training data contain only {self.perturbation_type}")
        if not genes:
            return abstain("no target genes given")
        if len(genes) > self.max_perturbed_genes:
            return abstain(f"{len(genes)} targets exceeds the supported maximum of {self.max_perturbed_genes}")
        missing = [g for g in genes if g not in self.vocab]
        if missing:
            return abstain(f"no representation for genes {missing}: not among genes measured in this dataset's processed gene set")

        key = frozenset(genes)
        n_seen = sum(g in self.measured_singles for g in genes)
        if key in self.measured:
            out.status = STATUS_WARN
            out.support_level = "measured"
            out.warnings.append(f"this condition is in the training data ('{self.measured[key]}'); its prediction is not an out-of-sample estimate")
        elif len(genes) == 2 and n_seen == 2:
            out.support_level = "unmeasured_pair_with_measured_singles"
        elif len(genes) == 2:
            out.support_level = f"pair_with_{2 - n_seen}_unmeasured_single"
            out.status = STATUS_WARN
            out.warnings.append(f"{2 - n_seen} of 2 targets have no measured single-gene response; see the benchmark's combo_with_unseen strata")
        else:
            out.support_level = "unmeasured_single"

        low = [g for g in genes if self.gene_detection.get(g, 0.0) < self.min_detection]
        if low:
            out.status = STATUS_WARN
            out.warnings.append(f"{low} detected in under {self.min_detection:.0%} of control cells, so their co-expression features are poorly determined")
        if x is not None:
            d = np.linalg.norm(self.X_train - x, axis=1)
            j = int(np.argmin(d))
            out.nn_distance, out.nearest_condition = float(d[j]), self.train_names[j]
            if key not in self.measured and d[j] > self.nn_threshold:
                out.status = STATUS_WARN
                out.warnings.append(f"far from measured perturbations (distance {d[j]:.2f} > 95th percentile {self.nn_threshold:.2f}; nearest: {self.train_names[j]})")
        return out
