"""End-to-end workflow on the SYNTHETIC fixture. Verifies software behaviour, not biology."""
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from biophysical import cli
from biophysical.config import dump_yaml
from biophysical.provenance import read_json
from biophysical.uncertainty.reliability import STATUS_ABSTAIN
from tests.conftest import PREPROCESS_CFG

CFG = {
    "dataset_id": "synthetic", "seed": 0, "preprocess": PREPROCESS_CFG,
    "context": {"cell_context": "K562", "perturbation_type": "CRISPRa"},
    "splits": {"n_folds": 3, "calibration_fraction": 0.2},
    "models": {"knn_k": [1, 3], "ridge_alpha": [1.0, 100.0], "mlp": {"n_members": 2, "hidden": [16], "n_components": 4, "max_epochs": 5, "patience": 3, "batch_size": 8}},
    "evaluation": {"top_k": 5, "n_bootstrap": 50},
    "active_learning": {"n_repeats": 1, "initial": 3, "batch": 1, "rounds": 1, "strategies": ["random", "hybrid"]},
}


@pytest.fixture(scope="module")
def run_dir(processed, tmp_path_factory, monkeypatch_module):
    from biophysical.evaluation.benchmark import run_benchmark

    root = tmp_path_factory.mktemp("home")
    monkeypatch_module.setenv("BIOPHYSICAL_HOME", str(root))
    cfg_path = root / "cfg.yaml"
    dump_yaml({**CFG, "_processed_path": str(processed)}, cfg_path)
    return run_benchmark(cfg_path, out_dir=root / "results" / "synthetic" / "run1")


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def service(run_dir):
    from biophysical.api.service import PredictionService

    return PredictionService(run_dir)


def test_benchmark_outputs_exist(run_dir):
    for f in ["summary.csv", "per_perturbation_metrics.csv", "paired_comparisons.csv", "split_assignments.csv", "noise_ceiling.csv", "uncertainty.csv", "runtime.csv", "run.json", "config.yaml", "serving_bundle.pkl", "acquisition_curves.csv", "perturbation_magnitude_ranking.csv"]:
        assert (run_dir / f).exists(), f


def test_all_models_scored_on_same_perturbations(run_dir):
    f = pd.read_csv(run_dir / "per_perturbation_metrics.csv")
    for (task, fold), g in f.groupby(["task", "fold"]):
        sets = g.groupby("model")["condition"].apply(frozenset)
        assert sets.nunique() == 1


def test_split_table_matches_task_definitions(run_dir):
    s = pd.read_csv(run_dir / "split_assignments.csv", keep_default_na=False)
    for (task, fold), g in s[s.task == "unseen_gene"].groupby(["task", "fold"]):
        roles = g.groupby("role")["condition"].apply(set)
        assert not roles["train"] & roles["test"] and not roles["calib"] & roles["test"]


def test_provenance_record(run_dir):
    r = read_json(run_dir / "run.json")
    assert r["source"]["md5"] and r["processed_version"] and r["config_digest"] and r["seed"] == 0
    assert r["environment"]["packages"]["torch"] and r["target_digests"]["Y_eval"] and r["peak_rss_mb"] > 0


def test_benchmark_is_reproducible(run_dir, processed, tmp_path):
    from biophysical.evaluation.benchmark import run_benchmark

    cfg_path = tmp_path / "cfg.yaml"
    dump_yaml({**CFG, "_processed_path": str(processed)}, cfg_path)
    again = run_benchmark(cfg_path, out_dir=tmp_path / "r2")
    a, b = [pd.read_csv(p / "per_perturbation_metrics.csv") for p in (run_dir, again)]
    pd.testing.assert_frame_equal(a[["condition", "model", "mse", "pearson"]], b[["condition", "model", "mse", "pearson"]])


def test_predict_returns_full_record(service):
    r = service.predict("SYN0", top_k=5)
    assert r["status"] in ("ok", "warning") and r["support_level"] == "measured" and any("training data" in w for w in r["warnings"])
    for key in ["model_version", "preprocessing_version", "input", "uncertainty", "known_limitations", "dataset"]:
        assert r[key]
    assert len(r["top_genes"]) == 5 and "interval_low" in r["top_genes"][0] and "observed_response" in r["top_genes"][0]
    assert "SYN0" not in [g["gene"] for g in r["top_genes"]]
    t = r["predicted_response"]
    assert (t["interval_low"] <= t["predicted_response"]).all() and (t["predicted_response"] <= t["interval_high"]).all()


@pytest.mark.parametrize("kw,fragment", [({"perturbation": "NOTAGENE"}, "no representation"), ({"perturbation": "SYN0", "perturbation_type": "CRISPRi"}, "perturbation type"), ({"perturbation": "SYN0", "cell_context": "RPE1"}, "cell context"), ({"perturbation": "SYN0", "cell_context": None}, "required"), ({"perturbation": "SYN0_SYN1_SYN2"}, "exceeds")])
def test_predict_abstains(service, kw, fragment):
    r = service.predict(**kw)
    assert r["status"] == STATUS_ABSTAIN and r["predicted_response"] is None and fragment in " ".join(r["warnings"]) and r["model_version"]


def test_unmeasured_gene_uses_single_model(service):
    measured = service.support.measured_singles
    other = next(g for g in service.data.genes if g not in measured)
    r = service.predict(other, allow_unvalidated=True)
    assert r["status"] != STATUS_ABSTAIN and r["support_level"] == "unmeasured_single" and r["serving_context"] == "unseen_single"


def test_rank_records_and_flags_unsupported(service, run_dir):
    measured = sorted(service.support.measured_singles)
    cands = [f"{measured[0]}_{g}" for g in measured[1:6]] + ["BADGENE"]
    t = service.rank(cands, budget=3, strategy="hybrid", record=True)
    ranked = t[t["rank"].notna()]
    assert len(ranked) == 3 and set(ranked["model_version"]) == {service.model_version}
    assert t[t["candidate"] == "BADGENE"]["support_status"].iloc[0] == "abstain"
    log = [json.loads(l) for l in open(run_dir / "recommendations.jsonl")]
    assert log[-1]["model_version"] == service.model_version and len(log[-1]["selected"]) == 3


def test_default_candidates_exclude_measured(service):
    measured = set(service.support.measured)
    c = service.default_candidates()
    assert c and not any(frozenset(p) in measured for p in c)


@pytest.mark.parametrize("strategy", ["uncertainty", "magnitude", "expected_improvement", "diversity", "random"])
def test_rank_all_strategies(service, strategy):
    assert len(service.rank(None, budget=2, strategy=strategy)) == 2


def test_observation_cycle(service, run_dir):
    genes = service.data.genes
    pair = service.default_candidates()[0]
    row = {"condition": "_".join(pair), **{g: 0.5 for g in genes}}
    rec = service.add_observations(pd.DataFrame([row]), source="synthetic test")
    assert rec["n_genes_matched"] == len(genes) and (run_dir / "observations.jsonl").exists()
    new = service.refit_with_observations()
    assert new.model_version != service.model_version and new.model_version.endswith("+obs1")
    assert not np.allclose(new.bundle["models"]["unseen_single"].predict([("SYN0",)]), service.bundle["models"]["unseen_single"].predict([("SYN0",)]))
    # the benchmark record stays unchanged by observations
    assert pd.read_csv(run_dir / "summary.csv").shape[0] > 0


def test_observations_validated(service):
    with pytest.raises(ValueError, match="condition"):
        service.add_observations(pd.DataFrame({"x": [1]}), "t")
    with pytest.raises(ValueError, match="model genes"):
        service.add_observations(pd.DataFrame({"condition": ["SYN0"], "SYN1": [1.0]}), "t")
    with pytest.raises(ValueError, match="unsupported"):
        service.add_observations(pd.DataFrame({"condition": ["BADGENE"], **{g: [0.0] for g in service.data.genes}}), "t")


def test_serving_bundle_loads(run_dir):
    with open(run_dir / "serving_bundle.pkl", "rb") as fh:
        b = pickle.load(fh)
    assert set(b["models"]) == {"unseen_single", "pair_with_measured_singles", "pair_with_unmeasured_single"}


def test_cli_lists_datasets(capsys):
    assert cli.main(["datasets", "list"]) == 0
    assert "norman2019" in capsys.readouterr().out


def test_cli_rejects_missing_local_file(tmp_path, capsys):
    assert cli.main(["datasets", "set-local", "norman2019", str(tmp_path / "missing.h5ad")]) == 2


def test_serving_evidence_recorded_and_used(run_dir, service):
    ev = read_json(run_dir / "serving_evidence.json")
    assert set(ev) == {"unseen_single", "pair_with_measured_singles", "pair_with_unmeasured_single"}
    assert all(v["verdict"] in ("better_than_baseline", "worse_than_baseline", "not_distinguishable_from_baseline") for v in ev.values())
    other = next(g for g in service.data.genes if g not in service.support.measured_singles)
    r = service.predict(other, allow_unvalidated=True)
    assert r["benchmark_evidence"]["model"] and "intervals covered" in " ".join(r["warnings"])


def _with_verdict(service, context, verdict):
    original = service.evidence
    service.evidence = {**original, context: {**original[context], "verdict": verdict}}
    return original


def test_abstains_where_benchmark_is_worse_than_baseline(service):
    other = next(g for g in service.data.genes if g not in service.support.measured_singles)
    original = _with_verdict(service, "unseen_single", "worse_than_baseline")
    try:
        r = service.predict(other)
        assert r["status"] == STATUS_ABSTAIN and r["predicted_response"] is None and "worse than" in r["warnings"][0]
        forced = service.predict(other, allow_unvalidated=True)
        assert forced["status"] == "warning" and forced["predicted_response"] is not None and "explicit request" in " ".join(forced["warnings"])
        table = service.rank([other], budget=1)
        assert table["support_status"].iloc[0] == "abstain"
        measured = next(iter(service.support.measured_singles))
        assert service.predict(measured)["status"] != STATUS_ABSTAIN  # measured conditions stay available
    finally:
        service.evidence = original


def test_not_distinguishable_verdict_warns(service):
    pair = service.default_candidates()[0]
    original = _with_verdict(service, "pair_with_measured_singles", "not_distinguishable_from_baseline")
    try:
        assert "not distinguishable" in " ".join(service.predict("+".join(pair))["warnings"])
    finally:
        service.evidence = original
