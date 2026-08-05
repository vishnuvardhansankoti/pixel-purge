"""Resolve and merge Google Takeout sidecar JSON metadata.

Google Takeout uses inconsistent sidecar naming. We resolve in the PRD's
priority order and merge the useful fields onto the MediaRecord.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..models import MediaRecord

# Truncated-name variants Google emits for long filenames, plus edited copies.
_SUPPLEMENTAL_SUFFIXES = (
    ".supplemental-metadata.json",
    ".supplemental-meta.json",
    ".suppl.json",
)


def _load_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def resolve_sidecar(media_file: Path) -> Path | None:
    """Find the sidecar JSON for a media file, trying naming conventions in order.

    Priority (PRD §2.3):
      1. {filename}.json                         e.g. IMG_1234.jpg.json
      2. {filename}.supplemental-metadata.json
      3. {stem}-edited.json                      (edited copies)
      4. {stem}({n}).json / {name}({n}).json     (duplicate-named exports)
      5. photoTakenTime match within dir (handled by caller as last resort)
    """
    directory = media_file.parent
    name = media_file.name          # IMG_1234.jpg
    stem = media_file.stem          # IMG_1234

    # 1. {filename}.json
    p = directory / f"{name}.json"
    if p.exists():
        return p

    # 2. supplemental-metadata variants
    for suf in _SUPPLEMENTAL_SUFFIXES:
        p = directory / f"{name}{suf}"
        if p.exists():
            return p
        # Google sometimes attaches the suffix to the stem, not the full name.
        p = directory / f"{stem}{suf}"
        if p.exists():
            return p

    # 3. {stem}-edited.json  (and the "-edited" media points back to base json)
    m = re.match(r"^(?P<base>.+)-edited$", stem)
    if m:
        p = directory / f"{m.group('base')}.json"
        if p.exists():
            return p
    p = directory / f"{stem}-edited.json"
    if p.exists():
        return p

    # 4. duplicate-named exports: IMG_1234(1).jpg -> IMG_1234.jpg(1).json
    m = re.match(r"^(?P<base>.+)\((?P<n>\d+)\)$", stem)
    if m:
        base, n = m.group("base"), m.group("n")
        for cand in (
            directory / f"{base}{media_file.suffix}({n}).json",
            directory / f"{base}({n}){media_file.suffix}.json",
            directory / f"{base}.json",
        ):
            if cand.exists():
                return cand

    return None


def merge_sidecar_metadata(record: MediaRecord, sidecar: Path) -> bool:
    """Merge fields from a sidecar JSON onto the record. Returns True on success."""
    data = _load_json(sidecar)
    if not data:
        return False

    def _ts(obj: dict, key: str) -> int | None:
        node = obj.get(key)
        if isinstance(node, dict) and node.get("timestamp") not in (None, ""):
            try:
                return int(node["timestamp"])
            except (TypeError, ValueError):
                return None
        return None

    record.taken_timestamp = _ts(data, "photoTakenTime") or record.taken_timestamp
    record.creation_timestamp = _ts(data, "creationTime") or record.creation_timestamp

    # Geo: prefer geoData, fall back to geoDataExif. (0,0) is Takeout's "unknown".
    for key in ("geoData", "geoDataExif"):
        geo = data.get(key) or {}
        lat, lon = geo.get("latitude"), geo.get("longitude")
        if lat not in (None, 0, 0.0) or lon not in (None, 0, 0.0):
            record.latitude = float(lat)
            record.longitude = float(lon)
            break

    desc = data.get("description")
    if desc:
        record.user_description = desc
    title = data.get("title")
    if title:
        record.original_title = title

    views = data.get("imageViews")
    if views not in (None, ""):
        try:
            record.view_count = int(views)
        except (TypeError, ValueError):
            pass

    return True
