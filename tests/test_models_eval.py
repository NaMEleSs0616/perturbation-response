import numpy as np
import pandas as pd
import pytest
import torch

from biophysical.active_learning.ranking import CandidateSet, expected_improvement, ranking_table, select
from biophysical.active_learning.simulate import run_simulation, summarise_simulation
from biophysical.evaluation.metrics import bootstrap_ci, metrics_frame, noise_ceiling, paired_comparison, perturbation_metrics, perturbation_ranking, response_magnitude
from biophysical.models.baselines import AdditiveModel, BootstrapRidgeEnsemble, FitData, GeneLoadingRidge, KNNModel, MeanModel, RidgeModel, ZeroModel
from biophysical.models.features import Featurizer, default_featurizer
from biophysical.models.mlp import MLPConfig, MLPEnsemble
from biophysical.training.trainer import TrainConfig, train_regressor
from biophysical.uncertainty.calibration import GeneIntervalCalibrator, disagreement, error_uncertainty_relation
from biophysical.uncertainty.reliability import STATUS_ABSTAIN, STATUS_OK, STATUS_WARN, SupportModel


def _fit(data):
    return FitData(data.condition_genes(), data.Y_fit, data.self_mask, default_featurizer(data), {g: i for i, g in enumerate(data.genes)})


ALL_MODELS = [ZeroModel, MeanModel, AdditiveModel, lambda: RidgeModel((1.0, 10.0)), lambda: KNNModel((1, 3)), lambda: GeneLoadingRidge(alphas=(1.0, 10.0)), lambda: BootstrapRidgeEnsemble(3, (1.0,)),
              lambda: MLPEnsemble(MLPConfig(n_members=2, max_epochs=5, patience=3, n_components=4)), lambda: RidgeModel((1.0,), base_kind="additive", name="ridge_add")]


@pytest.mark.parametrize("make", ALL_MODELS)
def test_model_output_shape(make, data):
    f = _fit(data)
    idx = np.arange(len(f.cond_genes))
    train, calib = f.subset(idx[:-4]), f.subset(idx[-4:-2])
    m = make().fit(train, calib)
    P = m.predict(f.cond_genes[-2:])
    assert P.shape == (2, len(data.genes)) and np.isfinite(P).all() and P.dtype == np.float32


def test_mean_additive_zero_definitions():
    gi = {"g0": 0, "g1": 1, "A": 2, "B": 3}
    feat = Featurizer(list(gi), {"e": np.eye(4, dtype=np.float32)}, np.zeros((4, 2), dtype=np.float32))
    Y = np.array([[1.0, 2, 9, 0], [3, 4, 0, 9], [10, 20, 0, 0]], dtype=np.float32)
    mask = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 1]], dtype=bool)
    f = FitData([("A",), ("B",), ("A", "B")], Y, mask, feat, gi)
    singles = f.subset(np.array([0, 1]))
    assert ZeroModel().fit(f).predict([("A",)]).sum() == 0
    mean = MeanModel().fit(singles).predict([("Z",)])[0]
    assert mean[:2].tolist() == [2.0, 3.0] and mean[2] == 0.0  # the masked self entry is excluded from the mean of A
    add = AdditiveModel().fit(singles).predict([("A", "B")])[0]
    assert add.tolist() == [4.0, 6.0, 0.0, 0.0]


def test_additive_uses_only_training_singles():
    gi = {"g": 0}
    feat = Featurizer(["g"], {"e": np.ones((1, 1), dtype=np.float32)}, np.zeros((1, 2), dtype=np.float32))
    f = FitData([("A",)], np.array([[1.0]], dtype=np.float32), np.zeros((1, 1), dtype=bool), feat, gi)
    m = AdditiveModel().fit(f)
    assert "B" not in m.singles


def test_ridge_learns_linear_signal():
    rng = np.random.default_rng(0)
    n, d, G = 80, 6, 30
    genes = [f"g{i}" for i in range(n)]
    E = rng.normal(size=(n, d)).astype(np.float32)
    W = rng.normal(size=(d, G))
    Y = (E @ W + 0.1 * rng.normal(size=(n, G))).astype(np.float32)
    feat = Featurizer(genes, {"e": E}, np.zeros((n, 2), dtype=np.float32), use_baseline=False)
    f = FitData([(g,) for g in genes], Y, np.zeros((n, G), dtype=bool), feat, {f"x{i}": i for i in range(G)})
    tr, te = np.arange(60), np.arange(60, 80)
    r = RidgeModel((0.1, 1.0)).fit(f.subset(tr), f.subset(te[:5]))
    P, B = r.predict(f.subset(te).cond_genes), MeanModel().fit(f.subset(tr)).predict(f.subset(te).cond_genes)
    assert ((Y[te] - P) ** 2).mean() < 0.2 * ((Y[te] - B) ** 2).mean()


def test_featurizer_sum_and_unknown_gene():
    feat = Featurizer(["A", "B"], {"e": np.array([[1.0], [2.0]], dtype=np.float32)}, np.array([[0.5, 0.1], [0.5, 0.1]], dtype=np.float32))
    X = feat.transform([("A",), ("A", "B")])
    assert X[1, 0] == 3.0 and X[0, -1] == 0.0 and X[1, -1] == 1.0
    with pytest.raises(KeyError):
        feat.transform([("Z",)])
    assert feat.transform([("A",)], covariates=np.array([[7.0]])).shape[1] == X.shape[1] + 1


def test_training_reduces_loss_and_early_stopping_restores_best():
    torch.manual_seed(0)
    X = np.random.default_rng(0).normal(size=(64, 3)).astype(np.float32)
    Y = X[:, [0]] * 2
    net = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
    before = float(((net(torch.from_numpy(X)).detach().numpy() - Y) ** 2).mean())
    h = train_regressor(net, X, Y, (X[:10], Y[:10]), TrainConfig(max_epochs=100, patience=20, loss="mse", lr=0.01), seed=0)
    assert float(((net(torch.from_numpy(X)).detach().numpy() - Y) ** 2).mean()) < before and h["best_val"] is not None


def test_unknown_loss_rejected():
    with pytest.raises(ValueError):
        train_regressor(torch.nn.Linear(1, 1), np.zeros((2, 1), np.float32), np.zeros((2, 1), np.float32), None, TrainConfig(loss="nope"))


def test_checkpoint_roundtrip_identical_predictions(data, tmp_path):
    f = _fit(data)
    m = MLPEnsemble(MLPConfig(n_members=2, max_epochs=5, patience=3, n_components=4)).fit(f.subset(np.arange(len(f.cond_genes) - 2)))
    p = tmp_path / "ckpt.pt"
    m.save(p, {"data_version": "x"})
    r = MLPEnsemble.load(p, f.featurizer)
    assert np.allclose(m.predict_members(f.cond_genes[-2:]), r.predict_members(f.cond_genes[-2:]))
    assert r.metadata["data_version"] == "x"


def test_additive_base_checkpoint_roundtrip(data, tmp_path):
    f = _fit(data)
    m = MLPEnsemble(MLPConfig(n_members=1, max_epochs=3, patience=2, n_components=3), base_kind="additive").fit(f)
    m.save(tmp_path / "c.pt")
    r = MLPEnsemble.load(tmp_path / "c.pt", f.featurizer)
    assert np.allclose(m.predict(f.cond_genes[:3]), r.predict(f.cond_genes[:3]))


def test_seeded_training_is_reproducible(data):
    f = _fit(data)
    mk = lambda: MLPEnsemble(MLPConfig(n_members=2, max_epochs=5, patience=3, n_components=3, seed=3)).fit(f)
    assert np.allclose(mk().predict(f.cond_genes[:4]), mk().predict(f.cond_genes[:4]))


def test_metrics_values():
    y = np.array([1.0, -2.0, 3.0, 0.0])
    p = np.array([1.0, -1.0, 2.0, 1.0])
    m = perturbation_metrics(y, p, top_k=2)
    assert m["mae"] == pytest.approx(0.75)
    assert m["mse"] == pytest.approx(0.75)
    assert m["pearson"] == pytest.approx(np.corrcoef(y, p)[0, 1])
    assert m["sign_acc_top"] == 1.0
    assert m["r2_vs_zero"] == pytest.approx(1 - 3 / 14)
    assert m["topk_recall"] == pytest.approx(0.5)


def test_mask_excludes_target():
    y, p = np.array([100.0, 1, 2, 3]), np.array([-100.0, 1, 2, 3])
    assert perturbation_metrics(y, p, np.array([True, False, False, False]))["mse"] == 0


def test_constant_prediction_has_undefined_correlation():
    m = perturbation_metrics(np.array([1.0, 2, 3]), np.zeros(3))
    assert np.isnan(m["pearson"]) and np.isnan(m["sign_acc_top"]) and np.isnan(m["topk_recall"])


def test_specific_metrics_and_mean_baseline_get_no_credit():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=50)
    specific = rng.normal(size=50)
    y = ref + specific
    mean_m = perturbation_metrics(y, ref, reference=ref)
    assert mean_m["pearson"] > 0.5  # looks good on raw response correlation
    assert np.isnan(mean_m["pearson_specific"]) and mean_m["skill_vs_mean"] == 0
    good = perturbation_metrics(y, ref + 0.9 * specific, reference=ref)
    assert good["pearson_specific"] == pytest.approx(1.0) and good["skill_vs_mean"] > 0.9


def test_metrics_frame_identical_conditions_across_models():
    Y, M = np.ones((3, 4)), np.zeros((3, 4), dtype=bool)
    f = metrics_frame(["a", "b", "c"], Y, Y * 0.5, M, top_k=2, reference=Y * 0, model="m")
    assert f["condition"].tolist() == ["a", "b", "c"] and set(f["model"]) == {"m"}


def test_ranking_and_ceiling():
    Y = np.array([[1.0, 1], [2, 2], [3, 3], [4, 4]])
    M = np.zeros_like(Y, dtype=bool)
    assert perturbation_ranking(Y, Y * 3, M) == pytest.approx(1.0)
    assert np.isnan(perturbation_ranking(Y, np.ones_like(Y), M))
    assert response_magnitude(Y, M)[1] == pytest.approx(2.0)
    rng = np.random.default_rng(0)
    a = rng.normal(size=(1, 200))
    t = noise_ceiling(a, a + 0.1 * rng.normal(size=(1, 200)), np.zeros((1, 200), dtype=bool))
    assert t.loc[0, "half_r"] > 0.9 and t.loc[0, "max_r"] >= t.loc[0, "half_r"]


def test_bootstrap_and_paired():
    mean, lo, hi = bootstrap_ci(np.arange(100.0))
    assert lo < mean < hi
    assert np.isnan(bootstrap_ci(np.array([np.nan]))[0])
    c = paired_comparison(np.arange(20.0) + 1, np.arange(20.0))
    assert c["mean_diff"] == pytest.approx(1) and c["frac_a_gt_b"] == 1 and c["wilcoxon_p"] < 0.01
    assert paired_comparison(np.array([np.nan]), np.array([1.0]))["n"] == 0


def test_interval_calibration_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    sd = rng.uniform(0.1, 2, 50)
    Y_cal, Y_te = rng.normal(0, sd, (200, 50)), rng.normal(0, sd, (500, 50))
    M_cal, M_te = np.zeros_like(Y_cal, bool), np.zeros_like(Y_te, bool)
    cal = GeneIntervalCalibrator(alpha=0.1).fit(Y_cal, np.zeros_like(Y_cal), M_cal)
    cov = cal.coverage(Y_te, np.zeros_like(Y_te), M_te).mean()
    assert 0.87 < cov < 0.93
    assert (cal.half_width()[np.argsort(sd)[-5:]].mean()) > (cal.half_width()[np.argsort(sd)[:5]].mean())
    lo, hi = cal.interval(np.zeros((1, 50)))
    assert (lo < 0).all() and (hi > 0).all()


def test_calibrator_excludes_masked_and_survives_state_roundtrip():
    Y = np.zeros((5, 3))
    Y[:, 0] = 100
    M = np.zeros((5, 3), bool)
    M[:, 0] = True
    cal = GeneIntervalCalibrator().fit(Y, np.zeros_like(Y), M)
    assert np.isfinite(cal.sigma).all() and cal.q < 10
    again = GeneIntervalCalibrator.from_state(cal.state())
    assert again.q == cal.q and np.array_equal(again.sigma, cal.sigma)


def test_disagreement_and_error_relation():
    assert disagreement(np.zeros((3, 2, 4)))[0] == 0
    M = np.stack([np.ones((2, 4)), -np.ones((2, 4))])
    assert disagreement(M)[0] > 0
    e = np.arange(20.0)
    assert error_uncertainty_relation(e, e * 2)["spearman"] == pytest.approx(1)
    assert np.isnan(error_uncertainty_relation(e, np.ones(20))["spearman"])


def _support():
    X = np.array([[0.0, 0], [1, 0], [0, 1]])
    return SupportModel("K562", "CRISPRa", 2, {"A", "B", "C", "LOW", "D"}, {frozenset(["A"]): "A", frozenset(["B"]): "B", frozenset(["A", "B"]): "A_B"}, {"A", "B"},
                        {"A": 0.9, "B": 0.9, "C": 0.9, "D": 0.9, "LOW": 0.001}, X, ["A", "B", "A_B"], nn_threshold=1.5)


@pytest.mark.parametrize("genes,ptype,ctx,status,fragment", [
    (("C",), "CRISPRa", "K562", STATUS_OK, ""),
    (("C",), "CRISPRi", "K562", STATUS_ABSTAIN, "perturbation type"),
    (("C",), None, "K562", STATUS_ABSTAIN, "required"),
    (("C",), "CRISPRa", "RPE1", STATUS_ABSTAIN, "cell context"),
    (("C",), "CRISPRa", None, STATUS_ABSTAIN, "required"),
    (("ZZZ",), "CRISPRa", "K562", STATUS_ABSTAIN, "no representation"),
    (("A", "B", "C"), "CRISPRa", "K562", STATUS_ABSTAIN, "exceeds"),
    ((), "CRISPRa", "K562", STATUS_ABSTAIN, "no target"),
    (("A",), "CRISPRa", "K562", STATUS_WARN, "training data"),
    (("LOW",), "CRISPRa", "K562", STATUS_WARN, "control cells"),
    (("A", "C"), "CRISPRa", "K562", STATUS_WARN, "no measured single"),
])
def test_support_checks(genes, ptype, ctx, status, fragment):
    a = _support().assess(genes, None, ptype, ctx)
    assert a.status == status
    assert fragment in " ".join(a.warnings)


def test_ood_warning_uses_distance():
    s = _support()
    assert s.assess(("C",), np.array([0.1, 0.1]), "CRISPRa", "K562").status == STATUS_OK
    far = s.assess(("C",), np.array([9.0, 9.0]), "CRISPRa", "K562")
    assert far.status == STATUS_WARN and far.nearest_condition and "far from" in far.warnings[0]


def _cands():
    return CandidateSet(["a", "b", "c", "d"], np.array([[0.0, 0], [0.1, 0], [5, 5], [10, 0]]), np.array([0.1, 0.9, 0.2, 0.3]), np.array([3.0, 1, 2, 0]), np.array([[3.0, 1, 2, 0], [3.2, 1.5, 2.1, 0.5]]))


@pytest.mark.parametrize("strategy,first", [("uncertainty", "b"), ("magnitude", "a")])
def test_ranking_order(strategy, first):
    c = _cands()
    idx, reasons, _ = select(c, 2, strategy)
    assert c.names[idx[0]] == first and len(reasons) == 2


def test_diversity_avoids_redundant_picks():
    c = _cands()
    idx, _, _ = select(c, 2, "diversity", X_measured=np.array([[0.0, 0]]))
    assert c.names[idx[0]] == "d" and "a" not in [c.names[i] for i in idx]


def test_hybrid_budget_unique_and_capped():
    idx, reasons, scores = select(_cands(), 99, "hybrid")
    assert sorted(idx.tolist()) == [0, 1, 2, 3] and len(reasons) == 4 and scores.size == 4


def test_random_strategy_seeded_and_unknown_rejected():
    c = _cands()
    assert select(c, 3, "random", seed=1)[0].tolist() == select(c, 3, "random", seed=1)[0].tolist()
    with pytest.raises(ValueError):
        select(c, 1, "nope")


def test_expected_improvement_ordering():
    M = np.array([[1.0, 2.0, 1.5], [1.2, 2.2, 3.0]])
    ei = expected_improvement(M, best=1.5)
    assert ei.min() >= 0 and ei[1] > ei[0]
    c = _cands()
    idx, reasons, _ = select(c, 1, "expected_improvement", best_magnitude=2.5)
    assert "Gaussian approximation" in reasons[0]
    with pytest.raises(ValueError):
        select(CandidateSet(["a"], np.zeros((1, 1)), np.zeros(1), np.zeros(1)), 1, "expected_improvement")


def test_ranking_table_columns():
    c = _cands()
    idx, r, s = select(c, 2, "uncertainty")
    t = ranking_table(c, idx, r, s, pd.DataFrame({"support_status": ["ok"] * 4}))
    assert {"rank", "candidate", "predicted_magnitude", "uncertainty", "rationale", "support_status"} <= set(t.columns) and t["rank"].tolist() == [1, 2]


def test_simulation_shares_starting_set_and_records_rounds(data):
    f = _fit(data)
    singles = np.flatnonzero(data.conditions["n_perts"].to_numpy() == 1)
    sim = run_simulation(f, singles, ["random", "hybrid"], n_repeats=2, initial=4, batch=2, rounds=2, test_fraction=0.3, seed=0)
    first = sim[sim["round"] == 0].pivot(index="repeat", columns="strategy", values="pearson_specific")
    assert np.allclose(first["random"], first["hybrid"])  # identical before any acquisition
    assert sim["n_acquired"].max() == 8
    s = summarise_simulation(sim, n_boot=50)
    assert s.set_index("strategy").loc["random", "auc_diff_vs_baseline"] == 0


def _pair_fit():
    gi = {"g0": 0, "g1": 1}
    feat = Featurizer(["A", "B", "C"], {"e": np.eye(3, dtype=np.float32)}, np.zeros((3, 2), dtype=np.float32))
    Y = np.array([[1.0, 2], [3, 4], [2, 3], [2, 3]], dtype=np.float32)
    return FitData([("A",), ("B",), ("C",), ("A", "B")], Y, np.zeros_like(Y, dtype=bool), feat, gi)


def test_scaled_additive_scale():
    from biophysical.models.baselines import ScaledAdditive

    f = _pair_fit()
    m = ScaledAdditive().fit(f)
    add = np.array([4.0, 6.0])
    expect = (add * f.Y[3]).sum() / (add * add).sum()
    assert m.scale == pytest.approx(expect)
    assert m.predict([("A", "C")])[0] == pytest.approx(np.array([3.0, 5.0]) * expect)
    assert m.predict([("A",)])[0].tolist() == [1.0, 2.0]


def test_additive_plus_mean_residual():
    from biophysical.models.baselines import AdditivePlusMeanResidual

    f = _pair_fit()
    m = AdditivePlusMeanResidual().fit(f)
    assert m.offset.tolist() == [-2.0, -3.0]
    assert m.predict([("A", "C")])[0].tolist() == [1.0, 2.0]
    assert m.predict([("A",)])[0].tolist() == [1.0, 2.0]


def test_singles_only_uses_no_pairs():
    from biophysical.models.baselines import SinglesOnly

    f = _pair_fit()
    m = SinglesOnly(MeanModel(), "s").fit(f)
    assert m.inner.overall.tolist() == [2.0, 3.0] and 2 not in m.inner.by_size


def test_interaction_metrics_and_top_coverage():
    from biophysical.evaluation.benchmark import _add_interaction_metrics, _coverage_top

    rng = np.random.default_rng(0)
    Y, A = rng.normal(size=(2, 30)), rng.normal(size=(2, 30))
    M = np.zeros_like(Y, dtype=bool)
    fr = pd.DataFrame({"x": [0, 1]})
    _add_interaction_metrics(fr, Y, Y.copy(), M, A, 5)  # perfect prediction
    assert fr["pearson_interaction"].iloc[0] == pytest.approx(1) and fr["skill_vs_additive"].iloc[0] == pytest.approx(1)
    fr2 = pd.DataFrame({"x": [0, 1]})
    _add_interaction_metrics(fr2, Y, A.copy(), M, A, 5)  # the additive prediction itself
    assert np.isnan(fr2["pearson_interaction"].iloc[0]) and fr2["skill_vs_additive"].iloc[0] == pytest.approx(0)
    cal = GeneIntervalCalibrator().fit(np.zeros((4, 30)), np.ones((4, 30)), np.zeros((4, 30), dtype=bool))
    assert _coverage_top(cal, Y, Y, M, np.zeros_like(Y), 5).tolist() == [1.0, 1.0]


def test_prioritisation_report_finds_known_top():
    from biophysical.active_learning.simulate import prioritisation_report

    y = np.arange(50, dtype=float)
    r = prioritisation_report([str(i) for i in range(50)], {"magnitude::perfect": y, "magnitude::flat": np.ones(50), "magnitude::reversed": -y}, {"magnitude": y}, n_boot=50).set_index("ranking")
    assert r.loc["perfect", "precision_at_k"] == 1 and r.loc["perfect", "enrichment_vs_random"] == 5 and r.loc["reversed", "precision_at_k"] == 0 and np.isnan(r.loc["flat", "precision_at_k"])


def test_pair_pool_simulation_keeps_singles_in_training(data):
    f = _fit(data)
    sizes = data.conditions["n_perts"].to_numpy()
    sim = run_simulation(f, np.flatnonzero(sizes == 2), ["random", "hybrid"], n_repeats=1, initial=3, batch=1, rounds=1, seed=0, fixed_idx=np.flatnonzero(sizes == 1), base_kind="additive")
    assert sim["n_acquired"].max() == 4 and len(sim) == 4


def test_spectral_embedding_clips_localised_eigenvectors():
    from biophysical.networks.features import spectral_embedding

    n = 200
    A = np.zeros((n, n))
    for i in range(n - 3):
        A[i, (i + 1) % (n - 3)] = A[(i + 1) % (n - 3), i] = 1
    A[n - 1, n - 2] = A[n - 2, n - 1] = 1  # a small separate component
    E = spectral_embedding(A, dim=8)
    assert np.abs(E).max() <= 5.0 and np.isfinite(E).all()
