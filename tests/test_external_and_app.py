import numpy as np
import pandas as pd
import pytest

from biophysical.models.external import align_embeddings, read_text_embeddings
from biophysical.networks.build import string_mapping
from biophysical.visualization import plots


def test_text_embeddings_parse_and_align(tmp_path):
    p = tmp_path / "emb.txt"
    p.write_text("2 3\nAAA 1 2 3\nBBB 4 5 6\nbad line x\n")
    t = read_text_embeddings(p)
    assert set(t) == {"AAA", "BBB"}
    M, rep = align_embeddings(t, ["AAA", "ZZZ"])
    assert M[0].tolist() == [1, 2, 3] and not M[1].any() and rep["missing"] == 1


def test_text_embeddings_reject_empty(tmp_path):
    p = tmp_path / "e.txt"
    p.write_text("not an embedding\n")
    with pytest.raises(ValueError):
        read_text_embeddings(p)


class _FakeString:
    def map_identifiers(self, genes):
        return pd.DataFrame({"queryItem": ["A", "B", "C"], "preferredName": ["A", "DUP", "DUP"]})


def test_string_mapping_drops_ambiguous():
    m, rep = string_mapping(["A", "B", "C", "D"], _FakeString())
    assert m == {"A": "A"} and rep["n_ambiguous_dropped"] == 2 and rep["n_unmapped"] == 1


def test_charts_build():
    s = pd.DataFrame({"metric": ["mse", "mse"], "model": ["a", "b"], "mean": [1.0, np.nan], "ci_low": [0.9, np.nan], "ci_high": [1.1, np.nan], "median": [1.0, np.nan], "n_perturbations": [3, 0]})
    assert plots.model_comparison(s, "mse", 0.5).to_dict()
    top = pd.DataFrame({"gene": ["g"], "predicted_response": [0.1], "interval_low": [0.0], "interval_high": [0.2], "ensemble_sd": [0.01]})
    assert plots.predicted_response(top).to_dict()
    c = pd.DataFrame({"strategy": ["random", "hybrid"], "n_acquired": [5, 5], "mse": [1.0, 2.0]})
    assert plots.acquisition_curves(c, "mse").to_dict()
    f = pd.DataFrame({"condition": ["x"], "model": ["a"], "mse": [1.0], "n_cells": [10], "disagreement": [0.1], "rmse": [1.0], "coverage": [0.9]})
    assert plots.per_perturbation_strip(f, "mse").to_dict() and plots.uncertainty_scatter(f).to_dict()


@pytest.mark.skipif("not __import__('importlib').util.find_spec('streamlit')")
def test_app_renders_without_exception():
    """Runs the Streamlit script headlessly against whatever runs exist under the current home."""
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
