"""Provenance helpers: hashing, environment capture and run records."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
import uuid
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

TRACKED_PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "anndata", "h5py", "torch", "scanpy", "requests"]


def file_md5(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def array_digest(arr: np.ndarray) -> str:
    """Stable short digest of an array's contents (used to version processed data)."""
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.tobytes() + str(a.shape).encode() + str(a.dtype).encode()).hexdigest()[:16]


def json_digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def environment_info() -> dict[str, Any]:
    pkgs = {}
    for p in TRACKED_PACKAGES:
        try:
            pkgs[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pkgs[p] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        commit = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": pkgs,
        "git_commit": commit,
    }


def new_run_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def write_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def read_json(path: str | Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)
