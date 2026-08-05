"""Data models for the manifest (plain dataclasses, no ORM)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class MediaRecord:
    """A single media item as tracked in the ``media_items`` table.

    Only the columns Phase 1 reads/writes are represented as first-class
    fields; the row round-trips through the DB by column name.
    """

    filename: str
    local_path: str
    file_size: int
    media_type: str  # 'PHOTO' | 'VIDEO'

    id: int | None = None

    # Timestamps (Unix epoch, seconds)
    taken_timestamp: int | None = None
    creation_timestamp: int | None = None

    # Geolocation
    latitude: float | None = None
    longitude: float | None = None

    # Sidecar metadata
    user_description: str | None = None
    original_title: str | None = None
    view_count: int | None = None

    # Dedup
    sha256_hash: str | None = None
    phash: str | None = None
    phash_cluster_id: int | None = None
    is_duplicate: int = 0
    duplicate_of: int | None = None
    dedup_tier: str | None = None
    hamming_distance: int | None = None

    # Module C: vision
    ai_caption: str | None = None
    ai_label: str | None = None
    blur_score: float | None = None
    ocr_text_ratio: float | None = None
    face_count: int = 0
    person_cluster_ids: str | None = None  # JSON array of cluster ids

    # Module C/E: classification
    classification_bucket: str | None = None
    classification_confidence: float | None = None
    classification_reasoning: str | None = None

    # Video
    keyframe_path: str | None = None
    keyframe_paths: str | None = None  # JSON array of multi-frame keyframes [M9]
    duration_seconds: float | None = None

    # Module D: cleanup / upload
    upload_status: str | None = None
    cloud_media_id: str | None = None

    # Status
    ingestion_status: str = "PENDING"
    vision_status: str = "PENDING"
    face_status: str = "PENDING"
    keeper_status: str = "PENDING"
    error_message: str | None = None

    @property
    def visual_path(self) -> str:
        """Path used for single-image analysis (video keyframe if present)."""
        return self.keyframe_path or self.local_path

    @property
    def frame_paths(self) -> list[str]:
        """All frames to hash for dedup — multiple keyframes for videos [M9]."""
        if self.keyframe_paths:
            import json

            try:
                paths = json.loads(self.keyframe_paths)
                if paths:
                    return paths
            except (ValueError, TypeError):
                pass
        return [self.visual_path]

    @property
    def metadata_richness(self) -> int:
        """Count of non-null 'valuable' metadata fields — used for keeper choice [M4]."""
        candidates = (
            self.taken_timestamp,
            self.latitude,
            self.longitude,
            self.user_description,
            self.original_title,
            self.view_count,
        )
        return sum(1 for c in candidates if c not in (None, ""))

    @classmethod
    def from_row(cls, row: Any) -> "MediaRecord":
        """Build a MediaRecord from a sqlite3.Row (extra columns ignored)."""
        known = {f.name for f in fields(cls)}
        data = {k: row[k] for k in row.keys() if k in known}
        return cls(**data)
