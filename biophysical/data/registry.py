"""Dataset registry: configured public datasets plus user-supplied local files."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from biophysical.config import Paths, get_paths, load_yaml
from biophysical.provenance import read_json

REQUIRED_FIELDS = [
    "name", "publication", "source_url", "accession", "file", "license", "organism",
    "assay", "cell_context", "perturbation_types", "expected_obs_fields", "control_label",
]


class RegistryError(ValueError):
    pass


@dataclass
class DatasetEntry:
    dataset_id: str
    name: str
    publication: str
    source_url: str
    accession: str
    file: str
    license: str
    organism: str
    assay: str
    cell_context: str
    perturbation_types: list[str]
    expected_obs_fields: list[str]
    control_label: str
    doi: str | None = None
    download_url: str | None = None
    md5: str | None = None
    size_mb: float | None = None
    download_instructions: str | None = None
    role: str | None = None
    preprocessing_config: str | None = None
    validation_status: str = "unknown"
    local_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def raw_dir(self, paths: Paths | None = None) -> Path:
        return (paths or get_paths()).raw / self.dataset_id

    def resolved_path(self, paths: Paths | None = None) -> Path:
        """Local override path if configured, otherwise the managed raw-data location."""
        if self.local_path:
            return Path(self.local_path).expanduser().resolve()
        return self.raw_dir(paths) / self.file

    def validation_record(self, paths: Paths | None = None) -> dict[str, Any] | None:
        p = self.raw_dir(paths) / "validation.json"
        return read_json(p) if p.exists() else None

    def current_status(self, paths: Paths | None = None) -> str:
        rec = self.validation_record(paths)
        if rec:
            return rec.get("status", "unknown")
        return "downloaded_unvalidated" if self.resolved_path(paths).exists() else self.validation_status


class DatasetRegistry:
    def __init__(self, registry_path: str | Path | None = None, paths: Paths | None = None):
        self.paths = paths or get_paths()
        self.registry_path = Path(registry_path) if registry_path else self.paths.configs / "datasets.yaml"
        cfg = load_yaml(self.registry_path)
        self.defaults: dict[str, Any] = cfg.get("defaults", {})
        local = load_yaml(self.paths.local_datasets) if self.paths.local_datasets.exists() else {}
        self.entries: dict[str, DatasetEntry] = {}
        for ds_id, raw in (cfg.get("datasets") or {}).items():
            missing = [f for f in REQUIRED_FIELDS if f not in raw]
            if missing:
                raise RegistryError(f"dataset '{ds_id}' missing required fields: {missing}")
            known = {k: v for k, v in raw.items() if k in DatasetEntry.__dataclass_fields__}
            extra = {k: v for k, v in raw.items() if k not in DatasetEntry.__dataclass_fields__}
            entry = DatasetEntry(dataset_id=ds_id, extra=extra, **known)
            if ds_id in (local or {}):
                entry.local_path = str(local[ds_id])
            self.entries[ds_id] = entry

    def __contains__(self, ds_id: str) -> bool:
        return ds_id in self.entries

    def get(self, ds_id: str) -> DatasetEntry:
        if ds_id not in self.entries:
            raise RegistryError(f"unknown dataset '{ds_id}'. Known: {sorted(self.entries)}")
        return self.entries[ds_id]

    def list(self) -> list[DatasetEntry]:
        return list(self.entries.values())
