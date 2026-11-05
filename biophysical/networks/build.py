"""Build a STRING adjacency matrix over a dataset's gene names, recording identifier mapping."""
from __future__ import annotations

import numpy as np
import pandas as pd

from biophysical.logging_utils import get_logger
from biophysical.networks.string_db import StringClient, edges_to_adjacency

log = get_logger(__name__)


def string_mapping(genes: list[str], client: StringClient) -> tuple[dict[str, str], dict]:
    """var gene name -> STRING preferred name. Two var genes sharing one STRING entry are both dropped."""
    table = client.map_identifiers(genes)
    pairs = dict(zip(table["queryItem"].astype(str), table["preferredName"].astype(str)))
    counts = pd.Series(list(pairs.values())).value_counts()
    ambiguous = {g for g, p in pairs.items() if counts[p] > 1}
    mapping = {g: p for g, p in pairs.items() if g not in ambiguous}
    return mapping, {"n_query": len(genes), "n_mapped": len(mapping), "n_unmapped": len(genes) - len(pairs), "n_ambiguous_dropped": len(ambiguous),
                     "n_renamed": sum(g != p for g, p in mapping.items())}


def string_adjacency(genes: list[str], client: StringClient, required_score: int = 400, channel_sets: dict[str, list[str] | None] | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """Adjacency matrices in `genes` order for each named evidence-channel selection."""
    mapping, report = string_mapping(genes, client)
    names = list(mapping.values())
    edges = client.network(names, required_score=required_score)
    report.update({"n_edges_returned": int(len(edges)), "required_score": required_score, "string_version": client.version()})
    channel_sets = channel_sets or {"combined": None}
    inverse = {p: g for g, p in mapping.items()}
    mapped_edges = edges.assign(preferredName_A=edges["preferredName_A"].map(inverse), preferredName_B=edges["preferredName_B"].map(inverse)).dropna(subset=["preferredName_A", "preferredName_B"])
    return {name: edges_to_adjacency(mapped_edges, genes, channels=ch) for name, ch in channel_sets.items()}, report
