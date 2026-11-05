"""Gene identifier validation and mapping against the HGNC complete set.

Mapping rules (never silently pick among ambiguous candidates):
  1. Ensembl gene ID (version stripped) -> HGNC record; >1 record => ambiguous.
  2. Exact approved symbol                -> status "approved".
  3. Previous (retired) symbol           -> status "previous_symbol" if it resolves to exactly one
                                             approved symbol, else "ambiguous".
  4. Alias symbol                        -> status "alias" if unique, else "ambiguous".
  5. Otherwise                           -> "unmapped".
Approved symbols always win over previous/alias usage of the same string.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from biophysical.config import get_paths
from biophysical.data.download import download_file
from biophysical.logging_utils import get_logger
from biophysical.provenance import file_md5, write_json

log = get_logger(__name__)

HGNC_URL = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
ENSG_RE = re.compile(r"^ENSG\d{11}(\.\d+)?$")


@dataclass
class GeneMapping:
    query: str
    status: str  # approved | previous_symbol | alias | ensembl | ambiguous | unmapped
    symbol: str | None = None
    hgnc_id: str | None = None
    ensembl_gene_id: str | None = None
    candidates: tuple[str, ...] = ()


class GeneMapper:
    def __init__(self, table: pd.DataFrame, source: dict | None = None):
        t = table.fillna("")
        t = t[t["status"].str.lower().isin(["approved", ""])] if "status" in t else t
        self.source = source or {}
        self._by_symbol = {r.symbol: r for r in t.itertuples(index=False)}
        self._by_symbol_upper: dict[str, list[str]] = {}
        for s in self._by_symbol:
            self._by_symbol_upper.setdefault(s.upper(), []).append(s)
        self._prev: dict[str, set[str]] = {}
        self._alias: dict[str, set[str]] = {}
        self._ensembl: dict[str, set[str]] = {}
        for r in t.itertuples(index=False):
            for p in filter(None, str(getattr(r, "prev_symbol", "")).split("|")):
                self._prev.setdefault(p.strip().upper(), set()).add(r.symbol)
            for a in filter(None, str(getattr(r, "alias_symbol", "")).split("|")):
                self._alias.setdefault(a.strip().upper(), set()).add(r.symbol)
            e = str(getattr(r, "ensembl_gene_id", "")).strip()
            if e:
                self._ensembl.setdefault(e, set()).add(r.symbol)

    def _record(self, query: str, status: str, symbol: str) -> GeneMapping:
        r = self._by_symbol[symbol]
        return GeneMapping(query, status, symbol, r.hgnc_id, getattr(r, "ensembl_gene_id", "") or None)

    def map(self, query: str) -> GeneMapping:
        q = str(query).strip()
        if not q:
            return GeneMapping(query, "unmapped")
        if ENSG_RE.match(q):
            hits = self._ensembl.get(q.split(".")[0], set())
            if len(hits) == 1:
                return self._record(query, "ensembl", next(iter(hits)))
            return GeneMapping(query, "ambiguous" if hits else "unmapped", candidates=tuple(sorted(hits)))
        if q in self._by_symbol:
            return self._record(query, "approved", q)
        up = self._by_symbol_upper.get(q.upper(), [])
        if len(up) == 1:
            return self._record(query, "approved", up[0])
        for status, index in (("previous_symbol", self._prev), ("alias", self._alias)):
            hits = index.get(q.upper(), set())
            if len(hits) == 1:
                return self._record(query, status, next(iter(hits)))
            if len(hits) > 1:
                return GeneMapping(query, "ambiguous", candidates=tuple(sorted(hits)))
        return GeneMapping(query, "unmapped")

    def map_many(self, queries) -> pd.DataFrame:
        rows = [asdict(self.map(q)) for q in queries]
        df = pd.DataFrame(rows)
        df["candidates"] = df["candidates"].apply(lambda c: "|".join(c))
        return df


def fetch_hgnc(dest_dir: str | Path | None = None, url: str = HGNC_URL) -> Path:
    dest_dir = Path(dest_dir or get_paths().external / "hgnc")
    dest = dest_dir / "hgnc_complete_set.txt"
    if not dest.exists():
        download_file(url, dest)
        write_json({"url": url, "md5": file_md5(dest), "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, dest_dir / "provenance.json")
    return dest


def load_mapper(path: str | Path | None = None, download: bool = True) -> GeneMapper:
    path = Path(path) if path else get_paths().external / "hgnc" / "hgnc_complete_set.txt"
    if not path.exists():
        if not download:
            raise FileNotFoundError(path)
        path = fetch_hgnc(path.parent)
    cols = ["hgnc_id", "symbol", "status", "alias_symbol", "prev_symbol", "ensembl_gene_id"]
    table = pd.read_csv(path, sep="\t", usecols=lambda c: c in cols, dtype=str, low_memory=False)
    prov_path = path.parent / "provenance.json"
    source = {"file": str(path), "md5": file_md5(path)}
    if prov_path.exists():
        import json

        source.update(json.loads(prov_path.read_text()))
    return GeneMapper(table, source=source)
