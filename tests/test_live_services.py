"""Live-service tests. Skipped unless BIOPHYSICAL_NETWORK_TESTS=1, so the default suite stays offline."""
import os

import pytest

pytestmark = [pytest.mark.network, pytest.mark.skipif(os.environ.get("BIOPHYSICAL_NETWORK_TESTS") != "1", reason="set BIOPHYSICAL_NETWORK_TESTS=1 to run")]


def test_geo_summary_and_supplementary_files():
    from biophysical.data.geo import GEOClient

    c = GEOClient()
    s = c.summary("GSE133344")
    assert s["accession"] == "GSE133344" and "31395745" in s["pubmed_ids"]
    assert "GSE133344_filtered_matrix.mtx.gz" in c.supplementary_files("GSE133344")


def test_string_network_has_evidence_channels(tmp_path):
    from biophysical.networks.string_db import StringClient

    c = StringClient(cache_dir=tmp_path)
    df = c.network(["SPI1", "GATA1", "CEBPE"])
    assert {"score", "escore", "dscore", "tscore", "ascore"} <= set(df.columns) and len(df) > 0
    assert c.version()["string_version"]


def test_open_targets_summary_carries_caveat():
    from biophysical.annotations.opentargets import OpenTargetsClient

    r = OpenTargetsClient().target_summary("KLF1")
    assert r["status"] == "ok" and "not causal" in r["caveat"].lower()


def test_zenodo_record_lists_expected_checksum():
    import requests

    files = requests.get("https://zenodo.org/api/records/13350497", timeout=30).json()["files"]
    assert {f["key"]: f["checksum"] for f in files}["NormanWeissman2019_filtered.h5ad"] == "md5:c870e6967d91c017d9da827bab183cd6"
