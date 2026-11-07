import numpy as np
import pandas as pd
import pytest

from biophysical.evaluation.study import confirmatory_tests, holm, one_sided_wilcoxon


def test_holm_matches_definition():
    adj = holm([0.01, 0.04, 0.03, float("nan")])
    assert adj[0] == pytest.approx(0.03) and adj[2] == pytest.approx(0.06) and adj[1] == pytest.approx(0.06) and np.isnan(adj[3])


def test_holm_is_monotone_and_capped():
    p = [0.5, 0.001, 0.2, 0.9]
    adj = holm(p)
    order = np.argsort(p)
    assert all(np.diff(np.array(adj)[order]) >= 0) and max(adj) <= 1


def test_one_sided_wilcoxon_direction():
    d = -np.arange(1, 21, dtype=float)
    assert one_sided_wilcoxon(d, "less") < 0.001 and one_sided_wilcoxon(d, "greater") > 0.99
    assert np.isnan(one_sided_wilcoxon(np.zeros(10), "less"))


def _frame(effect: float, n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        base = rng.uniform(1, 2)
        for variant, model, mse in [("coexpression", "mean", base), ("coexpression", "ridge", base + effect + rng.normal(0, 0.01)),
                                    ("coexpression", "gene_loading_ridge", base), ("coexpression", "mlp_ensemble", base + 0.1),
                                    ("shuffled_coexpression", "ridge", base + 0.2), ("shuffled_coexpression", "mlp_ensemble", base + 0.1),
                                    ("gene2vec", "ridge", base), ("gene2vec", "mlp_ensemble", base + 0.1)]:
            rows.append({"dataset_id": "d", "variant": variant, "model": model, "condition": f"g{i}", "stratum": "single_unseen", "split_seed": 0,
                         "mse": mse, "skill_vs_mean": 1 - mse / base, "pearson_specific": 0.0})
    return pd.DataFrame(rows)


CFG = {"seed": 0, "confirmatory_models": ["ridge", "gene_loading_ridge", "mlp_ensemble"], "representation_models": ["ridge", "mlp_ensemble"]}


def test_confirmatory_c1_detects_a_model_that_beats_mean():
    t = confirmatory_tests(_frame(-0.3), CFG, n_boot=200)
    c1 = t[t.test == "C1"].set_index("model")
    assert c1.loc["ridge", "beats_mean"] == 1 and c1.loc["mlp_ensemble", "beats_mean"] == 0 and c1.loc["gene_loading_ridge", "beats_mean"] == 0


def test_confirmatory_c2_c3_signs():
    t = confirmatory_tests(_frame(0.05), CFG, n_boot=200)  # co-expression ridge 0.05 worse than mean; gene2vec ridge equals mean
    c2 = t[t.test == "C2"].set_index("model")
    assert c2.loc["ridge", "coexpression_informative"] == 1   # shuffled is worse here by construction
    c3 = t[t.test == "C3"].set_index("model")
    assert c3.loc["ridge", "gene2vec_better"] == 1 and c3.loc["mlp_ensemble", "gene2vec_better"] == 0


def test_inclusion_rules_on_fixture(synthetic_h5ad, tmp_path):
    from biophysical.evaluation.study import inclusion_check

    class E:
        dataset_id = "syn"
        md5 = None

        def resolved_path(self, paths):
            return synthetic_h5ad

        def validation_record(self, paths):
            return {"status": "valid"}

    plan = {"min_control_cells": 100, "min_cells_per_condition": 50, "fallback_min_cells_per_condition": 25, "min_single_conditions": 10, "min_target_resolution": 0.8}
    r = inclusion_check(E(), None, plan)
    assert r["included"] and r["min_cells"] == 25 and r["replicate_col"] == "gemgroup" and r["cell_filters"] == {"good_coverage": True}
    assert not inclusion_check(E(), None, {**plan, "min_control_cells": 10_000})["included"]
    assert not inclusion_check(E(), None, {**plan, "min_single_conditions": 99})["included"]


def test_metric_calibration_duplicate_beats_mean_on_clean_data(data):
    from biophysical.evaluation.study import metric_calibration

    cal = metric_calibration(data)
    assert len(cal) == len(data.conditions)
    # planted effects are strong in the fixture, so a technical duplicate should beat the mean on MSE
    assert cal["mse_duplicate_wins"].mean() > 0.8 and "spearman_specific_duplicate" in cal
