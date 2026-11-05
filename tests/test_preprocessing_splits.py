import numpy as np
import pandas as pd
import pytest

from biophysical.data.inspect import read_obs_var
from biophysical.preprocessing.pipeline import apply_cell_filters, coexpression_embedding, load_response_data, preprocess_h5ad, resolve_target_genes, select_conditions
from biophysical.preprocessing.pseudobulk import build_groups, compute_group_stats, extract_dense_cells, select_genes
from biophysical.preprocessing.splits import Fold, SplitLeakageError, assert_no_leakage, unseen_combo_folds, unseen_gene_folds
from biophysical.preprocessing.targets import parse_condition
from tests.conftest import PREPROCESS_CFG, make_synthetic_adata


def _manual(path):
    import anndata as ad

    a = ad.read_h5ad(path)
    X = a.X.toarray()
    return a, np.log1p(X / X.sum(1, keepdims=True) * 1e4)


@pytest.mark.parametrize("fixture", ["synthetic_h5ad", "synthetic_h5ad_csr"])
def test_group_stats_match_direct_calculation(fixture, request):
    path = request.getfixturevalue(fixture)
    a, L = _manual(path)
    groups, M = build_groups(a.obs, "perturbation", "gemgroup", "control", seed=1)
    stats = compute_group_stats(path, groups, M, chunk=13)
    for i, r in groups.iterrows():
        members = M[:, i].toarray().ravel() > 0
        assert np.allclose(stats.sum_log[i], L[members].sum(0), atol=1e-2)
    cond = groups[groups.kind == "condition"].reset_index(drop=True)
    j = cond.index[cond.condition == "control"][0]
    assert np.allclose(stats.sum_sq_log[j], (L[(a.obs.perturbation == "control").to_numpy()] ** 2).sum(0), atol=1e-1)


def test_response_target_definition(synthetic_h5ad, data):
    """The training target equals the cell-weighted per-batch difference, computed independently."""
    a, L = _manual(synthetic_h5ad)
    obs = a.obs
    halves = np.random.default_rng(0).integers(0, 2, size=len(obs))
    genes = list(a.var_names)
    g_idx = [genes.index(g) for g in data.genes]
    cond = data.conditions["condition"].iloc[0]
    p_mask = (obs.perturbation == cond).to_numpy()
    c_mask = (obs.perturbation == "control").to_numpy()
    for version, half in [(data.Y_fit, 0), (data.Y_eval, 1)]:
        total, n = 0, 0
        for r in np.unique(obs.gemgroup):
            rm = (obs.gemgroup == r).to_numpy()
            k = (p_mask & rm).sum()
            if k == 0:
                continue
            total = total + k * (L[p_mask & rm].mean(0) - L[c_mask & rm & (halves == half)].mean(0))
            n += k
        assert np.allclose(version[0], (total / n)[g_idx], atol=1e-3)


def test_fit_and_eval_targets_use_disjoint_controls(synthetic_h5ad):
    import anndata as ad

    obs = ad.read_h5ad(synthetic_h5ad, backed="r").obs
    groups, M = build_groups(obs, "perturbation", "gemgroup", "control", seed=0)
    splits = groups[groups.kind == "control_split"]
    A, B = [M[:, splits.index[splits.level.str.endswith(f"|{h}")]].sum(1).A1 > 0 for h in (0, 1)]
    assert not (A & B).any()
    assert (A | B).sum() == (obs.perturbation == "control").sum()


def test_self_mask_marks_only_targets(data):
    idx = data.index()
    for cond, genes in zip(data.conditions["condition"], data.condition_genes()):
        expect = {g for g in genes}
        assert {data.genes[j] for j in np.flatnonzero(data.self_mask[idx[cond]])} == expect


def test_planted_effect_recovered(data):
    own = data.Y_all[data.self_mask]
    assert (own > 0).mean() > 0.9  # targets were up-regulated 4-fold in the fixture


def test_gene_selection_uses_controls_only(synthetic_h5ad, tmp_path):
    """Changing perturbed-cell expression must not change the selected genes or the embedding."""
    a = make_synthetic_adata()
    p1 = tmp_path / "a.h5ad"
    a.write_h5ad(p1)
    b = a.copy()
    X = b.X.toarray()
    perturbed = (b.obs.perturbation != "control").to_numpy()
    X[perturbed] = X[perturbed][:, ::-1]  # scramble perturbed cells only
    import scipy.sparse as sp

    b.X = sp.csc_matrix(X)
    p2 = tmp_path / "b.h5ad"
    b.write_h5ad(p2)
    d1 = load_response_data(preprocess_h5ad(p1, PREPROCESS_CFG, tmp_path / "o1", "s"))
    d2 = load_response_data(preprocess_h5ad(p2, PREPROCESS_CFG, tmp_path / "o2", "s"))
    assert d1.genes == d2.genes
    assert np.allclose(d1.gene_embedding, d2.gene_embedding)
    assert not np.allclose(d1.Y_eval, d2.Y_eval)


def test_select_genes_modes():
    mean = np.array([1, 2, 3, 4, 5.0, 0.0])
    var = np.array([1, 5, 1, 1, 1, 0.0])
    det = np.array([1, 1, 1, 1, 1, 0.0])
    assert 1 in select_genes(mean, var, det, "hvg", 1, n_bins=1)
    assert select_genes(mean, var, det, "top_expressed", 2).tolist() == [3, 4]
    with pytest.raises(ValueError):
        select_genes(mean, var, det, "nope", 2)
    with pytest.raises(ValueError):
        select_genes(mean, var, np.zeros(6), "hvg", 2)


def test_extract_dense_cells_order(synthetic_h5ad):
    a, L = _manual(synthetic_h5ad)
    cells, genes = np.array([5, 300, 2]), np.array([40, 3])
    got = extract_dense_cells(synthetic_h5ad, np.sort(cells), genes, chunk=11)
    assert np.allclose(got, L[np.sort(cells)][:, genes], atol=1e-4)


def test_cell_filters_and_unknown_column():
    obs = pd.DataFrame({"good": [True, True, False], "n": [1, 2, 1]})
    keep, rec = apply_cell_filters(obs, {"good": True, "n": 1})
    assert keep.tolist() == [True, False, False]
    assert rec["removed_by_filter"] == {"good": 1, "n": 1}
    with pytest.raises(ValueError, match="not in obs"):
        apply_cell_filters(obs, {"zzz": 1})


def test_condition_filtering_records_exclusions():
    labels = ["control"] * 5 + ["A"] * 5 + ["B"] * 2 + ["A_B_C"] * 5 + [None] * 2
    from biophysical.preprocessing.pipeline import DEFAULTS

    t, r = select_conditions(pd.DataFrame({"perturbation": labels}), {**DEFAULTS, "min_cells_per_condition": 3})
    assert t["condition"].tolist() == ["A"]
    reasons = {e["condition"]: e["reason"] for e in r["excluded_conditions"]}
    assert "fewer than" in reasons["B"] and "more than" in reasons["A_B_C"]
    assert r["n_cells_missing_label"] == 2


def test_duplicate_gene_sets_rejected():
    from biophysical.preprocessing.pipeline import DEFAULTS

    labels = ["A_B"] * 5 + ["B_A"] * 5 + ["control"] * 5
    with pytest.raises(ValueError, match="identical gene sets"):
        select_conditions(pd.DataFrame({"perturbation": labels}), {**DEFAULTS, "min_cells_per_condition": 3})


def test_parse_condition_drops_control_tokens():
    assert parse_condition("B_A") == ("A", "B")
    assert parse_condition("A_ctrl") == ("A",)
    assert parse_condition("control") == ()


def test_missing_control_label_is_not_guessed(synthetic_h5ad, tmp_path):
    with pytest.raises(ValueError, match="not inferred"):
        preprocess_h5ad(synthetic_h5ad, {**PREPROCESS_CFG, "control_label": "guide_62"}, tmp_path, "s")


def test_unresolvable_targets_reported():
    var = pd.DataFrame({"ensembl_id": ["E1"]}, index=["A"])
    resolved, rec = resolve_target_genes(["A", "B"], var, use_hgnc=False)
    assert resolved == {"A": "A"} and rec["unresolved"][0]["target"] == "B"


def test_raw_file_not_modified(synthetic_h5ad, tmp_path):
    from biophysical.provenance import file_md5

    before = file_md5(synthetic_h5ad)
    preprocess_h5ad(synthetic_h5ad, PREPROCESS_CFG, tmp_path, "s")
    assert file_md5(synthetic_h5ad) == before


def test_preprocess_provenance_and_cache(synthetic_h5ad, tmp_path):
    out1 = preprocess_h5ad(synthetic_h5ad, PREPROCESS_CFG, tmp_path, "s")
    out2 = preprocess_h5ad(synthetic_h5ad, PREPROCESS_CFG, tmp_path, "s")
    out3 = preprocess_h5ad(synthetic_h5ad, {**PREPROCESS_CFG, "n_genes": 50}, tmp_path, "s")
    assert out1 == out2 and out1 != out3
    m = load_response_data(out1).meta
    assert m["source"]["md5"] and m["selection"]["n_conditions_kept"] == 24 and m["normalisation"]


def test_coexpression_embedding_shape():
    X = np.random.default_rng(0).normal(size=(50, 12)).astype(np.float32)
    assert coexpression_embedding(X, 4).shape == (12, 4)


GENES = [("A",), ("B",), ("C",), ("D",), ("E",), ("F",), ("G",), ("H",), ("I",), ("J",), ("A", "B"), ("A", "C"), ("B", "D"), ("E", "F"), ("G", "H"), ("I", "J"), ("A", "J"), ("C", "D")]


def test_unseen_gene_folds_have_no_held_out_gene_in_training():
    folds = unseen_gene_folds(GENES, n_folds=3, seed=1)
    tested = set()
    for f in folds:
        assert_no_leakage(f, GENES)
        H = set(f.heldout_genes)
        assert not H & {g for i in f.train for g in GENES[i]}
        assert not H & {g for i in f.calib for g in GENES[i]}
        for i in f.test:
            n = sum(g in H for g in GENES[i])
            assert f.strata[int(i)] == ("single_unseen" if len(GENES[i]) == 1 else ("combo_two_unseen" if n == 2 else "combo_one_unseen"))
            tested |= {GENES[i]} if len(GENES[i]) == 1 else set()
    assert len(tested) == 10  # every single is tested exactly once across folds


def test_unseen_combo_folds_keep_singles_and_cover_all_pairs():
    folds = unseen_combo_folds(GENES, n_folds=3, seed=1)
    covered = []
    for f in folds:
        assert_no_leakage(f, GENES)
        assert {GENES[i] for i in range(10)} <= {GENES[i] for i in f.train}
        covered += f.test.tolist()
    assert sorted(covered) == list(range(10, 18))


def test_combo_without_measured_single_is_not_tested():
    genes = GENES + [("X", "Y")]
    for f in unseen_combo_folds(genes, 3, seed=0):
        assert 18 not in f.test


def test_split_is_seed_deterministic():
    a, b = unseen_gene_folds(GENES, 3, seed=5), unseen_gene_folds(GENES, 3, seed=5)
    assert all(np.array_equal(x.train, y.train) and x.heldout_genes == y.heldout_genes for x, y in zip(a, b))
    assert unseen_gene_folds(GENES, 3, seed=6)[0].heldout_genes != a[0].heldout_genes


def test_leakage_detector_fires():
    gs = [("A",), ("B",), ("C",), ("A", "B")]
    with pytest.raises(SplitLeakageError, match="share"):
        assert_no_leakage(Fold("unseen_gene", 0, np.array([0, 1]), np.array([]), np.array([1])), gs)
    with pytest.raises(SplitLeakageError, match="held-out genes"):
        assert_no_leakage(Fold("unseen_gene", 0, np.array([3]), np.array([], dtype=int), np.array([0]), heldout_genes=["A"]), gs)
    with pytest.raises(SplitLeakageError, match="calibration genes"):
        assert_no_leakage(Fold("unseen_gene", 0, np.array([3]), np.array([1]), np.array([2]), heldout_genes=["C"], calib_genes=["B"]), gs)
    with pytest.raises(SplitLeakageError, match="lacks a measured single"):
        assert_no_leakage(Fold("unseen_combo", 0, np.array([0]), np.array([], dtype=int), np.array([3])), gs)
    with pytest.raises(SplitLeakageError, match="duplicated"):
        assert_no_leakage(Fold("unseen_combo", 0, np.array([0, 1, 3]), np.array([], dtype=int), np.array([4])), gs + [("B", "A")])
