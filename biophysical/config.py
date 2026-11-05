"""Project paths and configuration loading.

All paths can be overridden with the ``BIOPHYSICAL_HOME`` environment variable so that
raw data can live on a larger disk without editing source code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def configs(self) -> Path:
        return REPO_ROOT / "configs"

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def external(self) -> Path:
        return self.root / "data" / "external"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def local_datasets(self) -> Path:
        """User-editable YAML mapping dataset_id -> local file path."""
        return self.configs / "local_paths.yaml"


def get_paths() -> Paths:
    return Paths(Path(os.environ.get("BIOPHYSICAL_HOME", REPO_ROOT)).resolve())


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def dump_yaml(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(obj, fh, sort_keys=False)
