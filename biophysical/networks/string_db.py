"""STRING interaction database adapter (https://string-db.org/help/api/).

Edges keep STRING's per-channel evidence scores so downstream code can distinguish:
  experimental  : "escore"  (experimentally determined interactions)
  curated       : "dscore"  (curated databases)
  text mining   : "tscore"
  predicted     : "nscore" (neighbourhood), "fscore" (gene fusion), "pscore" (phylogenetic
                  co-occurrence), "ascore" (co-expression)
STRING scores are *functional association* confidences, not evidence of direct physical
binding or causal regulation.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from biophysical.config import get_paths
from biophysical.logging_utils import get_logger
from biophysical.provenance import json_digest, write_json

log = get_logger(__name__)
API = "https://string-db.org/api"
SPECIES_HUMAN = 9606
CHANNELS = {
    "experimental": ["escore"],
    "curated": ["dscore"],
    "textmining": ["tscore"],
    "predicted": ["nscore", "fscore", "pscore", "ascore"],
}


class StringClient:
    def __init__(self, species: int = SPECIES_HUMAN, cache_dir: str | Path | None = None, caller: str = "biophysical_mvp",
                 session: requests.Session | None = None, timeout: float = 60.0):
        self.species = species
        self.cache_dir = Path(cache_dir or get_paths().external / "string")
        self.caller = caller
        self.session = session or requests.Session()
        self.timeout = timeout

    def _post(self, endpoint: str, params: dict) -> requests.Response:
        params = {**params, "species": self.species, "caller_identity": self.caller}
        for attempt in range(3):
            try:
                r = self.session.post(f"{API}/{endpoint}", data=params, timeout=self.timeout)
                r.raise_for_status()
                return r
            except requests.RequestException as exc:
                if attempt == 2:
                    raise RuntimeError(f"STRING request failed: {exc}") from exc
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    def version(self) -> dict:
        p = self.cache_dir / "version.json"
        if p.exists():
            return json.loads(p.read_text())
        r = self.session.get(f"{API}/json/version", timeout=self.timeout)
        r.raise_for_status()
        v = r.json()[0] if isinstance(r.json(), list) else r.json()
        write_json(v, p)
        return v

    def map_identifiers(self, genes: list[str]) -> pd.DataFrame:
        """Map symbols to STRING ids. Only the top hit (limit=1) is kept, and its preferredName is
        returned so callers can detect symbol changes; genes with no hit are absent."""
        r = self._post("tsv/get_string_ids", {"identifiers": "\r".join(genes), "limit": 1, "echo_query": 1})
        return pd.read_csv(io.StringIO(r.text), sep="\t")

    def network(self, genes: list[str], required_score: int = 400, use_cache: bool = True) -> pd.DataFrame:
        """Edges among `genes` (plus none added), with per-channel evidence scores in [0, 1]."""
        key = json_digest({"genes": sorted(genes), "score": required_score, "species": self.species})
        cache = self.cache_dir / f"network_{key}.tsv"
        if use_cache and cache.exists():
            return pd.read_csv(cache, sep="\t")
        # edges among a set require all identifiers in one request; POST handles a few thousand ids
        r = self._post("tsv/network", {"identifiers": "\r".join(genes), "required_score": required_score})
        df = pd.read_csv(io.StringIO(r.text), sep="\t").drop_duplicates()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, sep="\t", index=False)
        write_json({"genes": len(genes), "required_score": required_score, "species": self.species, "version": self.version(),
                    "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_edges": len(df)}, cache.with_suffix(".json"))
        return df


def edges_to_adjacency(edges: pd.DataFrame, genes: list[str], channels: list[str] | None = None) -> np.ndarray:
    """Symmetric weighted adjacency over `genes` from STRING edges.

    channels=None uses the combined score; otherwise the max of the selected channel groups
    (e.g. ["experimental", "curated"] for evidence that excludes text mining and prediction).
    """
    idx = {g: i for i, g in enumerate(genes)}
    A = np.zeros((len(genes), len(genes)), dtype=np.float32)
    if edges.empty:
        return A
    if channels is None:
        w = edges["score"].to_numpy(dtype=float)
    else:
        cols = [c for ch in channels for c in CHANNELS[ch] if c in edges.columns]
        w = edges[cols].to_numpy(dtype=float).max(axis=1)
    for a, b, s in zip(edges["preferredName_A"], edges["preferredName_B"], w):
        if a in idx and b in idx and s > 0:
            i, j = idx[a], idx[b]
            A[i, j] = A[j, i] = max(A[i, j], s)
    return A
