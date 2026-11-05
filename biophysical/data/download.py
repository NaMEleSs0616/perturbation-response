"""Resumable, checksum-verified downloads.

Downloads go to ``<file>.part`` and are renamed only after the MD5 matches, so an
interrupted or corrupted transfer never masquerades as a valid raw file.
"""
from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path

import requests

from biophysical.logging_utils import get_logger
from biophysical.provenance import file_md5, write_json

log = get_logger(__name__)


class DownloadError(RuntimeError):
    pass


def download_file(
    url: str,
    dest: str | Path,
    md5: str | None = None,
    expected_size_mb: float | None = None,
    retries: int = 8,
    timeout: float = 60.0,
    chunk: int = 1 << 20,
    session: requests.Session | None = None,
) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if md5 is None or file_md5(dest) == md5:
            log.info("already present and verified: %s", dest)
            return dest
        raise DownloadError(f"{dest} exists but MD5 does not match {md5}; refusing to overwrite raw data")

    if expected_size_mb is not None:
        free_mb = shutil.disk_usage(dest.parent).free / 1e6
        if free_mb < expected_size_mb * 1.2:
            raise DownloadError(f"insufficient disk space: {free_mb:.0f} MB free, need ~{expected_size_mb * 1.2:.0f} MB")

    part = dest.with_suffix(dest.suffix + ".part")
    sess = session or requests.Session()
    attempt = 0
    while True:
        attempt += 1
        offset = part.stat().st_size if part.exists() else 0
        start_offset = offset
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with sess.get(url, headers=headers, stream=True, timeout=timeout) as r:
                if offset and r.status_code == 200:
                    log.warning("server ignored Range request; restarting download")
                    offset = 0
                    part.unlink(missing_ok=True)
                elif r.status_code == 416:
                    break  # requested range beyond end -> file complete
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0)) + offset
                mode = "ab" if offset else "wb"
                done = offset
                last = time.time()
                with open(part, mode) as fh:
                    for block in r.iter_content(chunk_size=chunk):
                        fh.write(block)
                        done += len(block)
                        if time.time() - last > 15:
                            log.info("downloading %s: %.0f / %.0f MB", dest.name, done / 1e6, total / 1e6)
                            last = time.time()
            break
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as exc:
            if part.exists() and part.stat().st_size > start_offset:
                attempt = 1  # progress was made: count only consecutive failures
            if attempt >= retries:
                raise DownloadError(f"download failed after {attempt} attempts: {exc}") from exc
            wait = min(60, 2 ** attempt)
            log.warning("network error (%s); retrying in %ss (attempt %d/%d)", exc, wait, attempt, retries)
            time.sleep(wait)
        except requests.HTTPError as exc:
            raise DownloadError(f"HTTP error for {url}: {exc}") from exc

    if md5 is not None:
        got = file_md5(part)
        if got != md5:
            raise DownloadError(f"MD5 mismatch for {part}: expected {md5}, got {got}. Partial file kept for inspection.")
    part.rename(dest)
    log.info("downloaded and verified %s", dest)
    return dest


def download_dataset(entry, allow_large: bool = False, max_default_mb: float = 800, paths=None) -> Path:
    """Download a registry entry into data/raw/<dataset_id>/ and write a provenance record."""
    if entry.local_path:
        raise DownloadError(f"{entry.dataset_id} is configured with a local path ({entry.local_path}); nothing to download")
    if not entry.download_url:
        raise DownloadError(f"{entry.dataset_id} has no download_url; follow: {entry.download_instructions}")
    if entry.size_mb and entry.size_mb > max_default_mb and not allow_large:
        raise DownloadError(f"{entry.dataset_id} is {entry.size_mb} MB (> {max_default_mb} MB default limit); pass --allow-large")
    dest = entry.raw_dir(paths) / entry.file
    t0 = time.time()
    download_file(entry.download_url, dest, md5=entry.md5, expected_size_mb=entry.size_mb)
    write_json(
        {
            "dataset_id": entry.dataset_id,
            "url": entry.download_url,
            "file": str(dest.name),
            "md5": entry.md5 or file_md5(dest),
            "size_bytes": dest.stat().st_size,
            "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "seconds": round(time.time() - t0, 1),
        },
        entry.raw_dir(paths) / "download.json",
    )
    return dest


def md5_of_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()
