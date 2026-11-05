"""NCBI GEO metadata adapter (E-utilities + GEO FTP/HTTPS listing).

Uses the documented E-utilities endpoints (esearch/esummary on db=gds) rather than
scraping HTML. Supplementary files are listed from the GEO HTTPS mirror of the FTP tree.
This adapter retrieves metadata only; it does not parse raw sequencing data.
"""
from __future__ import annotations

import re
import time
from typing import Any

import requests

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
GEO_FTP_HTTPS = "https://ftp.ncbi.nlm.nih.gov/geo/series"


class GEOError(RuntimeError):
    pass


def series_ftp_dir(accession: str) -> str:
    """GSE133344 -> https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/"""
    m = re.fullmatch(r"GSE(\d+)", accession.strip().upper())
    if not m:
        raise GEOError(f"not a GEO series accession: {accession}")
    num = m.group(1)
    stub = "GSE" + (num[:-3] if len(num) > 3 else "") + "nnn"
    return f"{GEO_FTP_HTTPS}/{stub}/GSE{num}/"


class GEOClient:
    def __init__(self, session: requests.Session | None = None, timeout: float = 30.0, email: str | None = None):
        self.session = session or requests.Session()
        self.timeout = timeout
        self.email = email  # NCBI asks callers to identify themselves for heavy use

    def _get(self, url: str, params: dict[str, Any] | None = None) -> requests.Response:
        params = dict(params or {})
        if self.email:
            params["email"] = self.email
        for attempt in range(3):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 429:
                    time.sleep(1 + attempt)
                    continue
                r.raise_for_status()
                return r
            except requests.RequestException as exc:
                if attempt == 2:
                    raise GEOError(f"GEO request failed: {exc}") from exc
                time.sleep(1 + attempt)
        raise GEOError("GEO request failed (rate limited)")

    def summary(self, accession: str) -> dict[str, Any]:
        ids = self._get(f"{EUTILS}/esearch.fcgi", {"db": "gds", "term": f"{accession}[ACCN] AND gse[ETYP]", "retmode": "json"}).json()
        idlist = ids.get("esearchresult", {}).get("idlist", [])
        if not idlist:
            raise GEOError(f"no GDS record found for {accession}")
        res = self._get(f"{EUTILS}/esummary.fcgi", {"db": "gds", "id": idlist[0], "retmode": "json"}).json()["result"][idlist[0]]
        return {
            "accession": res.get("accession"),
            "title": res.get("title"),
            "summary": res.get("summary"),
            "taxon": res.get("taxon"),
            "gpl": res.get("gpl"),
            "n_samples": res.get("n_samples"),
            "pubmed_ids": res.get("pubmedids"),
            "supplementary_types": res.get("suppfile"),
            "ftp_link": res.get("ftplink"),
            "release_date": res.get("pdat"),
            "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

    def supplementary_files(self, accession: str, retries: int = 3) -> list[str]:
        """List the series' supplementary files. Only names in the series' own namespace are accepted,
        because the server can answer with an unrelated page (e.g. when throttling) and a 200 status."""
        url = series_ftp_dir(accession) + "suppl/"
        prefix = accession.strip().upper()
        for attempt in range(retries):
            links = re.findall(r'href="([^"?/][^"]*)"', self._get(url).text)
            files = sorted({link for link in links if not re.match(r"[a-z]+:", link) and (link.startswith(prefix) or link in ("filelist.txt",))})
            if files:
                return files
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
        raise GEOError(f"no files for {accession} could be read from {url}; the server may have returned a non-listing page")
