"""CSV / deletion-manifest export.

The deletion manifest is the **safe primary output** of Module D: a plain,
inspectable list of what the pipeline proposes to remove. The user can review it,
delete manually, or feed it to the (experimental) browser driver. Nothing here
touches Google Photos.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from ..database import Database
from ..models import MediaRecord

# PRD §3.3 CSV schema.
CSV_COLUMNS = [
    "id", "filename", "local_path", "media_type", "taken_timestamp",
    "latitude", "longitude", "file_size", "sha256_hash", "phash",
    "phash_cluster_id", "is_duplicate", "duplicate_of", "dedup_tier",
    "ai_caption", "ai_label", "blur_score", "ocr_text_ratio", "face_count",
    "person_cluster_ids", "classification_bucket", "keeper_status",
]


def _iso(ts: int | None) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _row(rec: MediaRecord) -> dict:
    d = {c: getattr(rec, c, None) for c in CSV_COLUMNS}
    d["taken_timestamp"] = _iso(rec.taken_timestamp)
    d["is_duplicate"] = bool(rec.is_duplicate)
    return d


def export_csv(db: Database, output: Path, records: list[MediaRecord] | None = None) -> int:
    """Write the full manifest (or a given subset) to CSV. Returns row count."""
    records = records if records is not None else db.get_all_records()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for rec in records:
            writer.writerow(_row(rec))
    return len(records)


def export_deletion_manifest(db: Database, output: Path) -> int:
    """Write only the items marked for deletion. Returns count."""
    return export_csv(db, output, records=db.get_deletions())
