"""Open Targets Platform annotation adapter (GraphQL API).

Downstream annotation only. Target-disease association scores summarise heterogeneous public
evidence; they are NOT perturbation-response labels and do not establish causality, efficacy
or safety. They are never used as model training targets in this project.
"""
from __future__ import annotations

import time
from typing import Any

import requests

API = "https://api.platform.opentargets.org/api/v4/graphql"

_SEARCH = """
query search($q: String!) {
  search(queryString: $q, entityNames: ["target"], page: {index: 0, size: 5}) {
    hits { id name entity }
  }
}"""

_TARGET = """
query target($id: String!) {
  target(ensemblId: $id) {
    id approvedSymbol approvedName biotype
    associatedDiseases(page: {index: 0, size: 5}) {
      count
      rows { score disease { id name } }
    }
  }
}"""

_META = "query { meta { name apiVersion { x y z } dataVersion { year month } } }"


class OpenTargetsClient:
    def __init__(self, session: requests.Session | None = None, timeout: float = 30.0):
        self.session = session or requests.Session()
        self.timeout = timeout

    def _query(self, query: str, variables: dict | None = None) -> dict[str, Any]:
        r = self.session.post(API, json={"query": query, "variables": variables or {}}, timeout=self.timeout)
        r.raise_for_status()
        payload = r.json()
        if "errors" in payload:
            raise RuntimeError(f"Open Targets error: {payload['errors']}")
        return payload["data"]

    def version(self) -> dict[str, Any]:
        return self._query(_META)["meta"]

    def target_summary(self, symbol: str) -> dict[str, Any]:
        hits = [h for h in self._query(_SEARCH, {"q": symbol})["search"]["hits"] if h["entity"] == "target"]
        exact = [h for h in hits if h["name"].upper() == symbol.upper()]
        if len(exact) != 1:
            return {"query": symbol, "status": "ambiguous" if len(exact) > 1 or hits else "not_found",
                    "candidates": [h["id"] for h in hits]}
        t = self._query(_TARGET, {"id": exact[0]["id"]})["target"]
        return {
            "query": symbol,
            "status": "ok",
            "ensembl_id": t["id"],
            "approved_symbol": t["approvedSymbol"],
            "approved_name": t["approvedName"],
            "biotype": t["biotype"],
            "n_associated_diseases": t["associatedDiseases"]["count"],
            "top_associations": [{"disease": r["disease"]["name"], "disease_id": r["disease"]["id"], "overall_score": r["score"]}
                                 for r in t["associatedDiseases"]["rows"]],
            "caveat": "Association scores aggregate public evidence; not causal, not efficacy or safety evidence.",
            "source": API,
            "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
