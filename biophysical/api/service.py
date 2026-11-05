"""Local programmatic interface: predictions, candidate ranking and observation intake.

Every result carries its model version, data versions, input context, uncertainty, warnings and
limitations. Nothing here contacts an external service.
"""
from __future__ import annotations

import json
import pickle
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from biophysical.active_learning.ranking import CandidateSet, ranking_table, select
from biophysical.config import get_paths
from biophysical.evaluation.metrics import response_magnitude
from biophysical.logging_utils import get_logger
from biophysical.models.baselines import FitData
from biophysical.models.features import default_featurizer
from biophysical.preprocessing.pipeline import load_response_data
from biophysical.provenance import read_json
from biophysical.uncertainty.calibration import GeneIntervalCalibrator, disagreement
from biophysical.uncertainty.reliability import STATUS_ABSTAIN, SupportModel

log = get_logger(__name__)
LIMITATIONS = [
    "Predicts a pseudobulk change in log-normalised mRNA expression in K562 under CRISPRa; not protein abundance, phenotype, or clinical effect.",
    "Validated only by held-out perturbations from one screen and one cell line; no prospective validation has been done.",
    "Predicted associations are not evidence of a causal mechanism for a specific gene pair.",
]


def split_targets(text: str) -> tuple[str, ...]:
    for sep in ("+", "_", ","):
        if sep in text:
            return tuple(sorted({t.strip() for t in text.split(sep) if t.strip()}))
    return (text.strip(),) if text.strip() else ()


class PredictionService:
    def __init__(self, run_dir: str | Path, bundle: dict[str, Any] | None = None):
        self.run_dir = Path(run_dir)
        self.run = read_json(self.run_dir / "run.json")
        self.data = load_response_data(self.run["processed_path"])
        if bundle is None:
            with open(self.run_dir / "serving_bundle.pkl", "rb") as fh:
                bundle = pickle.load(fh)
        self.bundle = bundle
        self.model_version = bundle["model_version"]
        self.evidence: dict[str, Any] = bundle.get("evidence", {})
        self.calibrators = {k: GeneIntervalCalibrator.from_state(v) for k, v in bundle["calibrators"].items()}
        self.featurizer = default_featurizer(self.data)
        self.aliases: dict[str, str] = self.data.meta.get("target_aliases", {})
        self.reverse_aliases = {v: k for k, v in self.aliases.items()}
        cond = self.data.conditions
        self.cond_genes = self.data.condition_genes()
        ctx = self.run["config"].get("context", {})
        self.cell_context = ctx.get("cell_context", "K562")
        self.perturbation_type = ctx.get("perturbation_type", "CRISPRa")
        X = self.featurizer.transform(self.cond_genes)
        self.scaler = StandardScaler().fit(X)
        self.X_measured = self.scaler.transform(X)
        detection = dict(zip(self.data.genes, self.data.control_detection.tolist()))
        self.support = SupportModel(
            cell_context=self.cell_context, perturbation_type=self.perturbation_type, max_perturbed_genes=int(self.data.meta["config"]["max_perturbed_genes"]),
            vocab=set(self.data.embedding_genes), measured={frozenset(g): c for g, c in zip(self.cond_genes, cond["condition"])},
            measured_singles={g[0] for g in self.cond_genes if len(g) == 1}, gene_detection=detection,
            X_train=self.X_measured, train_names=cond["condition"].tolist(), nn_threshold=SupportModel.loo_nn_threshold(self.X_measured))
        self._gene_pos = {g: i for i, g in enumerate(self.data.genes)}

    @classmethod
    def from_latest(cls, run: str | None = None) -> "PredictionService":
        if run:
            return cls(run)
        pointer = get_paths().results / "latest_run.txt"
        if not pointer.exists():
            raise FileNotFoundError("no benchmark run found; run `python -m biophysical.cli benchmark run` first")
        return cls(pointer.read_text().strip())

    def resolve(self, text: str) -> tuple[str, ...]:
        return tuple(sorted(self.aliases.get(t, t) for t in split_targets(text)))

    def context_for(self, genes: tuple[str, ...]) -> str:
        if len(genes) == 1:
            return "unseen_single"
        return "pair_with_measured_singles" if all(g in self.support.measured_singles for g in genes) else "pair_with_unmeasured_single"

    def predict_members(self, genes: tuple[str, ...]) -> tuple[np.ndarray, str]:
        context = self.context_for(genes)
        return self.bundle["models"][context].predict_members([genes])[:, 0, :], context

    def blocked_by_benchmark(self, genes: tuple[str, ...]) -> str | None:
        """Reason to abstain when held-out evaluation shows this input type does worse than its baseline."""
        if frozenset(genes) in self.support.measured:
            return None
        ev = self.evidence.get(self.context_for(genes))
        if ev and ev["verdict"] == "worse_than_baseline":
            return (f"On held-out {', '.join(ev['strata'])} perturbations, the {ev['model']} model did worse than the {ev['baseline']} baseline "
                    f"(MSE difference {ev['mean_diff']:.5f}, 95% CI {ev['ci_low']:.5f} to {ev['ci_high']:.5f}), so no defensible prediction is available for this input.")
        return None

    def predict(self, perturbation: str, perturbation_type: str | None = "CRISPRa", cell_context: str | None = "K562", top_k: int = 20, allow_unvalidated: bool = False) -> dict[str, Any]:
        genes = self.resolve(perturbation)
        record: dict[str, Any] = {
            "input": {"perturbation": perturbation, "resolved_targets": list(genes), "perturbation_type": perturbation_type, "cell_context": cell_context},
            "model_version": self.model_version,
            "dataset": self.run["dataset_id"],
            "preprocessing_version": self.run["processed_version"],
            "source_md5": self.run["source"].get("md5"),
            "known_limitations": LIMITATIONS,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        pre = self.support.assess(genes, None, perturbation_type, cell_context)
        if pre.status == STATUS_ABSTAIN:
            record.update({"status": STATUS_ABSTAIN, "warnings": pre.warnings, "predicted_response": None, "uncertainty": None})
            return record
        blocked = self.blocked_by_benchmark(genes)
        if blocked and not allow_unvalidated:
            record.update({"status": STATUS_ABSTAIN, "warnings": [blocked], "predicted_response": None, "uncertainty": None, "benchmark_evidence": self.evidence.get(self.context_for(genes))})
            return record
        x = self.scaler.transform(self.featurizer.transform([genes]))[0]
        a = self.support.assess(genes, x, perturbation_type, cell_context)
        if blocked:
            a.status = "warning"
            a.warnings.append("Returned on explicit request despite this finding: " + blocked)
        members, context = self.predict_members(genes)
        mean, sd = members.mean(axis=0), members.std(axis=0, ddof=1)
        evidence = self.evidence.get(context)
        if evidence and evidence["verdict"] == "not_distinguishable_from_baseline" and a.support_level != "measured":
            a.warnings.append(f"Benchmark: on held-out perturbations of this type, the {evidence['model']} model's MSE was not distinguishable from the {evidence['baseline']} baseline.")
            a.status = "warning"
        if evidence and a.support_level != "measured":
            a.warnings.append(f"Benchmark: intervals covered {evidence['empirical_interval_coverage']:.0%} of all genes but only {evidence['empirical_interval_coverage_top_genes']:.0%} of the most strongly responding genes in held-out data, so they are too narrow for strong responders.")
        cal = self.calibrators[context]
        lo, hi = cal.interval(mean)
        target_idx = [self._gene_pos[g] for g in genes if g in self._gene_pos]
        table = pd.DataFrame({"gene": self.data.genes, "predicted_response": mean, "ensemble_sd": sd, "interval_low": lo, "interval_high": hi, "control_mean_log1p": self.data.control_mean})
        table["is_perturbed_target"] = False
        table.loc[target_idx, "is_perturbed_target"] = True
        ranked = table[~table["is_perturbed_target"]].assign(abs_pred=lambda d: d["predicted_response"].abs()).sort_values("abs_pred", ascending=False).drop(columns=["abs_pred", "is_perturbed_target"])
        if a.support_level == "measured":
            i = self.data.index()[self.support.measured[frozenset(genes)]]
            ranked["observed_response"] = self.data.Y_all[i][ranked.index.to_numpy()]
        keep = np.setdiff1d(np.arange(len(mean)), target_idx)
        record.update({
            "status": a.status,
            "support_level": a.support_level,
            "warnings": a.warnings,
            "nearest_measured_condition": a.nearest_condition,
            "distance_to_nearest": a.nn_distance,
            "serving_context": context,
            "benchmark_evidence": evidence,
            "predicted_magnitude": float(np.sqrt((mean[keep] ** 2).mean())),
            "uncertainty": {
                "model_disagreement_rms": float(np.sqrt((sd[keep] ** 2).mean())),
                "interval_level": 1 - cal.alpha,
                "interval_median_half_width": float(np.median(cal.half_width())),
                "interval_calibration": f"out-of-fold residuals from {cal.n_calibration} benchmark perturbations; coverage is empirical",
                "not_included": "Model disagreement excludes measurement noise and bias common to all members.",
            },
            "top_genes": ranked.head(top_k).round(4).to_dict("records"),
            "predicted_response": table,
        })
        return record

    def default_candidates(self) -> list[tuple[str, ...]]:
        """All unmeasured pairs of genes with a measured single-gene response."""
        singles = sorted(self.support.measured_singles)
        return [p for p in combinations(singles, 2) if frozenset(p) not in self.support.measured]

    def rank(self, candidates: list[str] | None, budget: int = 10, strategy: str = "hybrid", perturbation_type: str = "CRISPRa", cell_context: str = "K562", seed: int = 0, record: bool = False, allow_unvalidated: bool = False) -> pd.DataFrame:
        queries = [self.resolve(c) for c in candidates] if candidates else self.default_candidates()
        queries = list(dict.fromkeys(queries))
        supported, rows = [], []
        for g in queries:
            a = self.support.assess(g, None, perturbation_type, cell_context)
            blocked = self.blocked_by_benchmark(g) if a.status != STATUS_ABSTAIN and not allow_unvalidated else None
            if a.status == STATUS_ABSTAIN or blocked:
                rows.append({"rank": None, "candidate": "+".join(g), "support_status": "abstain", "warnings": blocked or "; ".join(a.warnings), "rationale": "not ranked: no defensible prediction"})
            else:
                supported.append(g)
        ranked = pd.DataFrame()
        if supported:
            X = self.scaler.transform(self.featurizer.transform(supported))
            n = len(supported)
            mags, uncerts, member_mags = np.zeros(n), np.zeros(n), []
            contexts = [self.context_for(g) for g in supported]
            mask = np.zeros((n, len(self.data.genes)), dtype=bool)
            for i, g in enumerate(supported):
                for t in g:
                    if t in self._gene_pos:
                        mask[i, self._gene_pos[t]] = True
            n_m = min(len(m.members) for m in self.bundle["models"].values())
            for ctx in set(contexts):
                idx = [i for i, c in enumerate(contexts) if c == ctx]
                M = self.bundle["models"][ctx].predict_members([supported[i] for i in idx])[:n_m]
                uncerts[idx] = disagreement(M)
                mags[idx] = response_magnitude(M.mean(axis=0), mask[idx])
                member_mags.append((idx, np.stack([response_magnitude(M[k], mask[idx]) for k in range(M.shape[0])])))
            mm = np.zeros((n_m, n))
            for idx, v in member_mags:
                mm[:, idx] = v
            names = ["+".join(self.reverse_aliases.get(t, t) for t in g) for g in supported]
            c = CandidateSet(names=names, X=X, uncertainty=uncerts, magnitude=mags, member_magnitude=mm)
            best = float(response_magnitude(self.data.Y_all, self.data.self_mask).max())
            idx, reasons, scores = select(c, budget, strategy, X_measured=self.X_measured, best_magnitude=best, seed=seed)
            extra_rows = []
            for i in range(n):
                a = self.support.assess(supported[i], X[i], perturbation_type, cell_context)
                ev = self.evidence.get(contexts[i], {})
                extra_rows.append({"benchmark_verdict": ev.get("verdict"), "support_status": a.status, "support_level": a.support_level, "nearest_measured": a.nearest_condition, "distance_to_nearest": a.nn_distance, "warnings": "; ".join(a.warnings)})
            ranked = ranking_table(c, idx, reasons, scores, pd.DataFrame(extra_rows))
        table = pd.concat([ranked, pd.DataFrame(rows)], ignore_index=True) if rows else ranked
        table["model_version"] = self.model_version
        table["strategy"] = strategy
        if record:
            self._append_log("recommendations.jsonl", {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model_version": self.model_version, "strategy": strategy, "budget": budget, "seed": seed,
                                                         "n_candidates": len(queries), "selected": ranked["candidate"].tolist() if len(ranked) else []})
        return table

    def _append_log(self, name: str, record: dict[str, Any]) -> None:
        with open(self.run_dir / name, "a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def add_observations(self, table: pd.DataFrame, source: str) -> dict[str, Any]:
        """Register observed responses from completed experiments.

        Format: one row per condition with a `condition` column and one column per gene holding the
        response on the same scale as the training target. Observations are stored beside the run and
        used only by an explicit refit, so benchmark results are never changed retroactively.
        """
        if "condition" not in table.columns:
            raise ValueError("observation table needs a 'condition' column")
        overlap = [g for g in self.data.genes if g in table.columns]
        if len(overlap) < 0.5 * len(self.data.genes):
            raise ValueError(f"observation table has {len(overlap)} of {len(self.data.genes)} model genes; at least half are required")
        bad = [c for c in table["condition"] if self.support.assess(self.resolve(str(c)), None, self.perturbation_type, self.cell_context).status == STATUS_ABSTAIN]
        if bad:
            raise ValueError(f"unsupported conditions in observations: {bad}")
        obs_dir = self.run_dir / "observations"
        obs_dir.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = obs_dir / f"observations-{stamp}.csv"
        table.to_csv(path, index=False)
        rec = {"created_at": stamp, "file": str(path), "source": source, "n_conditions": int(len(table)), "n_genes_matched": len(overlap), "model_version": self.model_version}
        self._append_log("observations.jsonl", rec)
        return rec

    def load_observations(self) -> tuple[list[tuple[str, ...]], np.ndarray, np.ndarray]:
        genes, rows, masks = [], [], []
        for f in sorted((self.run_dir / "observations").glob("observations-*.csv")) if (self.run_dir / "observations").exists() else []:
            t = pd.read_csv(f)
            for _, r in t.iterrows():
                g = self.resolve(str(r["condition"]))
                y = r.reindex(self.data.genes).to_numpy(dtype=np.float32)
                m = ~np.isfinite(y) | np.isin(np.array(self.data.genes), list(g))
                genes.append(g)
                rows.append(np.nan_to_num(y))
                masks.append(m)
        n = len(self.data.genes)
        return genes, (np.stack(rows) if rows else np.zeros((0, n), dtype=np.float32)), (np.stack(masks) if masks else np.zeros((0, n), dtype=bool))

    def refit_with_observations(self, seed: int = 0) -> "PredictionService":
        """Refit serving models on measured data plus observations. Calibrators are carried over and
        should be re-estimated by a new benchmark before the new model's intervals are relied on."""
        from biophysical.evaluation.benchmark import model_factories

        genes, Y_new, M_new = self.load_observations()
        if not genes:
            raise ValueError("no observations registered")
        measured = set(map(frozenset, self.cond_genes))
        all_genes = self.cond_genes + genes
        Y = np.concatenate([self.data.Y_all, Y_new])
        M = np.concatenate([self.data.self_mask, M_new])
        fit = FitData(all_genes, Y, M, self.featurizer, self._gene_pos)
        models = {}
        for context, (task, model_name, _) in self.bundle["serving"].items():
            models[context] = model_factories(self.run["config"], task, seed)[model_name]().fit(fit)
        new_bundle = {**self.bundle, "models": models, "model_version": f"{self.model_version}+obs{len(genes)}"}
        self._append_log("refits.jsonl", {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "base_model": self.model_version, "new_model": new_bundle["model_version"], "n_observations": len(genes),
                                          "n_new_conditions": sum(frozenset(g) not in measured for g in genes)})
        return PredictionService(self.run_dir, bundle=new_bundle)
