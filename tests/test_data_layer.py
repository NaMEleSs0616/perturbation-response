import json

import numpy as np
import pandas as pd
import pytest
import requests

from biophysical.annotations.gene_ids import GeneMapper
from biophysical.config import Paths, get_paths
from biophysical.data.download import DownloadError, download_file
from biophysical.data.geo import GEOError, series_ftp_dir
from biophysical.data.h5sparse import cell_library_sizes, iter_major_chunks, sample_block, x_layout
from biophysical.data.inspect import inspect_h5ad, read_obs_var
from biophysical.data.registry import DatasetRegistry, RegistryError
from biophysical.data.validate import validate_tables
from biophysical.networks.features import neighbour_average, spectral_embedding
from biophysical.networks.string_db import edges_to_adjacency
from biophysical.provenance import file_md5


def test_registry_entries_complete():
    reg = DatasetRegistry()
    assert "norman2019" in reg
    for e in reg.list():
        assert e.md5 and e.download_url and e.license and e.accession
    with pytest.raises(RegistryError):
        reg.get("not_a_dataset")


def test_registry_rejects_incomplete_entry(tmp_path):
    p = tmp_path / "d.yaml"
    p.write_text("datasets:\n  x:\n    name: only a name\n")
    with pytest.raises(RegistryError, match="missing required fields"):
        DatasetRegistry(p)


def test_local_path_override(tmp_path, synthetic_h5ad):
    root = tmp_path
    (root / "configs").mkdir()
    paths = get_paths()

    class P(Paths):
        @property
        def local_datasets(self):
            return root / "configs" / "local_paths.yaml"

    (root / "configs" / "local_paths.yaml").write_text(f"norman2019: {synthetic_h5ad}\n")
    reg = DatasetRegistry(paths=P(paths.root))
    assert reg.get("norman2019").resolved_path(reg.paths) == synthetic_h5ad.resolve()


class _FakeResponse:
    def __init__(self, body, status=200, fail_after=None):
        self.body, self.status_code, self.fail_after = body, status, fail_after
        self.headers = {"Content-Length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self.body), 4):
            if self.fail_after is not None and i >= self.fail_after:
                raise requests.ConnectionError("dropped")
            yield self.body[i:i + 4]


class _FakeSession:
    """Serves bytes, honours Range requests, and can drop the first connection."""
    def __init__(self, body, drop_first_after=None):
        self.body, self.drop, self.ranges = body, drop_first_after, []

    def get(self, url, headers=None, stream=True, timeout=None):
        start = int(headers["Range"].split("=")[1].rstrip("-")) if headers else 0
        self.ranges.append(start)
        drop, self.drop = self.drop, None
        return _FakeResponse(self.body[start:], status=206 if start else 200, fail_after=drop)


def test_download_resumes_and_verifies(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    body = bytes(range(200))
    import hashlib

    sess = _FakeSession(body, drop_first_after=80)
    dest = download_file("http://x/f", tmp_path / "f.bin", md5=hashlib.md5(body).hexdigest(), session=sess)
    assert dest.read_bytes() == body
    assert sess.ranges[0] == 0 and sess.ranges[1] >= 80  # second request resumed
    assert not (tmp_path / "f.bin.part").exists()


def test_download_checksum_mismatch_keeps_no_final_file(tmp_path):
    with pytest.raises(DownloadError, match="MD5 mismatch"):
        download_file("http://x/f", tmp_path / "f.bin", md5="0" * 32, session=_FakeSession(b"data" * 10))
    assert not (tmp_path / "f.bin").exists()


def test_download_refuses_to_overwrite_mismatched_raw(tmp_path):
    (tmp_path / "f.bin").write_bytes(b"existing")
    with pytest.raises(DownloadError, match="refusing to overwrite"):
        download_file("http://x/f", tmp_path / "f.bin", md5="0" * 32, session=_FakeSession(b""))
    assert (tmp_path / "f.bin").read_bytes() == b"existing"


def test_large_download_requires_flag():
    from biophysical.data.download import download_dataset

    entry = DatasetRegistry().get("replogle2022_rpe1")
    with pytest.raises(DownloadError, match="--allow-large"):
        download_dataset(entry)


@pytest.mark.parametrize("fixture", ["synthetic_h5ad", "synthetic_h5ad_csr"])
def test_chunked_reader_matches_dense(fixture, request):
    import anndata as ad

    path = request.getfixturevalue(fixture)
    dense = ad.read_h5ad(path).X.toarray()
    layout = x_layout(path)
    rebuilt = np.zeros(dense.shape, dtype=np.float32)
    for s, e, block in iter_major_chunks(path, chunk=17):
        if layout.major_axis == 1:
            rebuilt[:, s:e] = block.toarray()
        else:
            rebuilt[s:e] = block.toarray()
    assert np.array_equal(rebuilt, dense)
    assert np.allclose(cell_library_sizes(path), dense.sum(1))
    assert sample_block(path).shape[0] > 0


def test_inspect_and_validate(synthetic_h5ad):
    rep = inspect_h5ad(synthetic_h5ad)
    assert rep["perturbation_summary"]["n_control_cells"] == 300
    assert rep["X_sample_integer_valued"] and "gemgroup" in rep["replicate_like_columns"]
    obs, var = read_obs_var(synthetic_h5ad)
    checks = {c["check"]: c["passed"] for c in validate_tables(obs, var, sample_block(synthetic_h5ad), ["perturbation"], "control")}
    assert all(checks.values())
    bad = {c["check"]: c["passed"] for c in validate_tables(obs, var, sample_block(synthetic_h5ad), ["missing_col"], "not_the_control_label")}
    assert not bad["expected_obs_fields"] and not bad["control_cells"]


def _mapper():
    return GeneMapper(pd.DataFrame([
        {"hgnc_id": "HGNC:1", "symbol": "AAA1", "status": "Approved", "alias_symbol": "SHARED|OLDA", "prev_symbol": "RETIRED1", "ensembl_gene_id": "ENSG00000000001"},
        {"hgnc_id": "HGNC:2", "symbol": "BBB2", "status": "Approved", "alias_symbol": "SHARED", "prev_symbol": "AMBIGPREV", "ensembl_gene_id": "ENSG00000000002"},
        {"hgnc_id": "HGNC:3", "symbol": "CCC3", "status": "Approved", "alias_symbol": "AAA1", "prev_symbol": "AMBIGPREV", "ensembl_gene_id": "ENSG00000000003"},
        {"hgnc_id": "HGNC:4", "symbol": "WDRAWN", "status": "Entry Withdrawn", "alias_symbol": "", "prev_symbol": "", "ensembl_gene_id": "ENSG00000000004"},
    ]))


@pytest.mark.parametrize("query,status,symbol", [
    ("AAA1", "approved", "AAA1"),             # approved beats another gene's alias
    ("aaa1", "approved", "AAA1"),
    ("RETIRED1", "previous_symbol", "AAA1"),
    ("OLDA", "alias", "AAA1"),
    ("SHARED", "ambiguous", None),            # never picks arbitrarily
    ("AMBIGPREV", "ambiguous", None),
    ("ENSG00000000002.7", "ensembl", "BBB2"), # version stripped
    ("ENSG99999999999", "unmapped", None),
    ("WDRAWN", "unmapped", None),             # withdrawn entries do not resolve
    ("NOPE", "unmapped", None),
])
def test_gene_mapping(query, status, symbol):
    m = _mapper().map(query)
    assert (m.status, m.symbol) == (status, symbol)


def test_gene_mapping_lists_ambiguous_candidates():
    assert set(_mapper().map("SHARED").candidates) == {"AAA1", "BBB2"}


def test_geo_paths():
    assert series_ftp_dir("GSE133344").endswith("/GSE133nnn/GSE133344/")
    assert series_ftp_dir("GSE90").endswith("/GSEnnn/GSE90/")
    with pytest.raises(GEOError):
        series_ftp_dir("GSM1")


class _HtmlSession:
    def __init__(self, pages):
        self.pages = list(pages)

    def get(self, url, params=None, timeout=None):
        body = self.pages.pop(0) if len(self.pages) > 1 else self.pages[0]
        r = requests.Response()
        r.status_code = 200
        r._content = body.encode()
        return r


def test_geo_listing_ignores_unrelated_pages(monkeypatch):
    from biophysical.data.geo import GEOClient

    monkeypatch.setattr("time.sleep", lambda s: None)
    junk = '<a href="https://www.hhs.gov/vulnerability-disclosure-policy/index.html">x</a>'
    listing = '<a href="/geo/">up</a><a href="GSE1_matrix.mtx.gz">m</a><a href="filelist.txt">f</a><a href="https://x.org/y">z</a>'
    assert GEOClient(session=_HtmlSession([junk, listing])).supplementary_files("GSE1") == ["GSE1_matrix.mtx.gz", "filelist.txt"]
    with pytest.raises(GEOError, match="non-listing page"):
        GEOClient(session=_HtmlSession([junk])).supplementary_files("GSE1")


def test_string_adjacency_preserves_evidence_channels():
    edges = pd.DataFrame({"preferredName_A": ["A", "A"], "preferredName_B": ["B", "C"], "score": [0.9, 0.7], "escore": [0.8, 0.0], "dscore": [0.0, 0.0], "tscore": [0.5, 0.7], "nscore": [0, 0], "fscore": [0, 0], "pscore": [0, 0], "ascore": [0, 0]})
    genes = ["A", "B", "C", "D"]
    assert edges_to_adjacency(edges, genes)[0, 2] == pytest.approx(0.7)
    exp = edges_to_adjacency(edges, genes, channels=["experimental"])
    assert exp[0, 1] == pytest.approx(0.8) and exp[0, 2] == 0 and exp[1, 0] == exp[0, 1]


def test_network_features_shapes():
    A = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=float)
    E = np.arange(8, dtype=float).reshape(4, 2)
    N = neighbour_average(A, E)
    assert np.allclose(N[1], (E[0] + E[2]) / 2) and np.allclose(N[3], 0)
    assert spectral_embedding(A, dim=2).shape == (4, 2)


def test_file_md5(tmp_path):
    p = tmp_path / "a"
    p.write_bytes(b"abc")
    assert file_md5(p) == "900150983cd24fb0d6963f7d28e17f72"
