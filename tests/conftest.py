"""SYNTHETIC fixtures for software tests. Nothing here is biological evidence."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def make_synthetic_adata(n_singles=14, n_pairs=10, cells_per=40, n_ctrl=300, n_genes=120, sparse_format="csc", seed=0, control_label="control"):
    """A small Perturb-seq-like AnnData with planted effects. Labelled synthetic by gene names."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    genes = [f"SYN{i}" for i in range(n_genes)]
    base = rng.gamma(2.0, 2.0, size=n_genes) + 0.2
    targets = genes[:n_singles]
    effects = {t: rng.normal(0, 0.8, size=n_genes) * (rng.random(n_genes) < 0.2) for t in targets}
    conds, rates = [control_label] * n_ctrl, [base] * n_ctrl

    def rate_for(ts):
        r = base * np.exp(sum(effects[t] for t in ts))
        r = r.copy()
        for t in ts:
            r[genes.index(t)] *= 4
        return r

    labels = [(t,) for t in targets]
    pairs = []
    for i in range(n_singles):
        for j in range(i + 1, n_singles):
            pairs.append((targets[i], targets[j]))
    order = rng.permutation(len(pairs))[:n_pairs]
    labels += [pairs[k] for k in order]
    for ts in labels:
        name = "_".join(ts)
        conds += [name] * cells_per
        rates += [rate_for(ts)] * cells_per
    counts = rng.poisson(np.stack(rates)).astype(np.float32)
    X = sp.csc_matrix(counts) if sparse_format == "csc" else sp.csr_matrix(counts)
    n = len(conds)
    obs = pd.DataFrame({
        "perturbation": pd.Categorical(conds),
        "perturbation_type": pd.Categorical(["CRISPR"] * n),
        "nperts": [0 if c == control_label else c.count("_") + 1 for c in conds],
        "ncounts": np.asarray(counts.sum(1)).ravel(),
        "ngenes": (counts > 0).sum(1),
        "gemgroup": rng.integers(1, 5, size=n),
        "good_coverage": np.ones(n, dtype=bool),
    }, index=[f"cell{i}" for i in range(n)])
    var = pd.DataFrame({"ensembl_id": [f"ENSG{i:011d}" for i in range(n_genes)]}, index=genes)
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture(scope="session")
def synthetic_h5ad(tmp_path_factory):
    path = tmp_path_factory.mktemp("synthetic") / "synthetic_perturbseq.h5ad"
    make_synthetic_adata().write_h5ad(path)
    return path


@pytest.fixture(scope="session")
def synthetic_h5ad_csr(tmp_path_factory):
    path = tmp_path_factory.mktemp("synthetic_csr") / "synthetic_perturbseq_csr.h5ad"
    make_synthetic_adata(sparse_format="csr").write_h5ad(path)
    return path


PREPROCESS_CFG = {"replicate_col": "gemgroup", "cell_filters": {"good_coverage": True}, "min_cells_per_condition": 10, "n_genes": 60, "min_control_detection": 0.01, "embedding_dim": 8, "resolve_targets_with_hgnc": False}


@pytest.fixture(scope="session")
def processed(synthetic_h5ad, tmp_path_factory):
    from biophysical.preprocessing.pipeline import preprocess_h5ad

    return preprocess_h5ad(synthetic_h5ad, PREPROCESS_CFG, tmp_path_factory.mktemp("processed"), dataset_id="synthetic")


@pytest.fixture(scope="session")
def data(processed):
    from biophysical.preprocessing.pipeline import load_response_data

    return load_response_data(processed)
