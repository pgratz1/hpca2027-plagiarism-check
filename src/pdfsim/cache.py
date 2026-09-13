"""Per-PDF extraction cache.

Extraction is deterministic given (PDF bytes, extractor schema, options), so a
result is keyed by sha256(pdf) + SCHEMA_VERSION + a hash of the options.  The
JSON document and its figure PNGs live under `<cache_dir>/<key>/`.  The cache
is gitignored and safe to delete at any time (`make clean-cache`).

Bump SCHEMA_VERSION whenever extract.py or textnorm.py changes what they emit.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2


def pdf_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_key(sha256: str, options: dict[str, Any]) -> str:
    opt = hashlib.sha1(json.dumps(options, sort_keys=True).encode()).hexdigest()[:8]
    return f"{sha256[:16]}-v{SCHEMA_VERSION}-{opt}"


def entry_dir(cache_dir: str | Path, key: str) -> Path:
    return Path(cache_dir) / key


def load(cache_dir: str | Path, key: str) -> dict[str, Any] | None:
    path = entry_dir(cache_dir, key) / "document.json"
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if data.get("schema_version") != SCHEMA_VERSION:
        return None
    return data


def store(cache_dir: str | Path, key: str, data: dict[str, Any]) -> Path:
    d = entry_dir(cache_dir, key)
    d.mkdir(parents=True, exist_ok=True)
    data = dict(data, schema_version=SCHEMA_VERSION)
    tmp = d / "document.json.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    tmp.replace(d / "document.json")
    return d


def clear(cache_dir: str | Path) -> None:
    p = Path(cache_dir)
    if p.is_dir():
        shutil.rmtree(p)
