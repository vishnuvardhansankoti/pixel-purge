"""SQLite data-access layer for the manifest DB (WAL mode)."""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path
from typing import Iterable, Iterator

from .models import MediaRecord

# Columns Phase 1 writes when inserting a freshly ingested record.
_INSERT_COLUMNS = (
    "filename",
    "local_path",
    "file_size",
    "media_type",
    "taken_timestamp",
    "creation_timestamp",
    "latitude",
    "longitude",
    "user_description",
    "original_title",
    "view_count",
    "keyframe_path",
    "duration_seconds",
    "error_message",
    "ingestion_status",
)


class Database:
    """Thin wrapper over sqlite3 with the queries the pipeline needs.

    Usable as a context manager; commits on clean exit, rolls back on error.
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=30000")  # 30s, per PRD error handling

    # ---- lifecycle -------------------------------------------------------
    def init_schema(self) -> None:
        schema = resources.files("core_engine").joinpath("schema.sql").read_text()
        self.conn.executescript(schema)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        self.close()

    def commit(self) -> None:
        self.conn.commit()

    # ---- ingestion -------------------------------------------------------
    def insert_media_record(self, record: MediaRecord) -> int:
        placeholders = ", ".join("?" for _ in _INSERT_COLUMNS)
        cols = ", ".join(_INSERT_COLUMNS)
        values = [getattr(record, c) for c in _INSERT_COLUMNS]
        cur = self.conn.execute(
            f"INSERT INTO media_items ({cols}) VALUES ({placeholders})", values
        )
        record.id = cur.lastrowid
        return record.id

    def get_ingested_paths(self) -> set[str]:
        """Return the set of local_paths already in the manifest.

        Resume keys on the FULL PATH, not the basename [M1]: Takeout reuses
        filenames across albums, and local_path is the UNIQUE column.
        """
        rows = self.conn.execute("SELECT local_path FROM media_items").fetchall()
        return {r["local_path"] for r in rows}

    def count_media(self) -> int:
        return self.conn.execute("SELECT COUNT(*) AS c FROM media_items").fetchone()["c"]

    # ---- generic helpers -------------------------------------------------
    def _query_records(self, sql: str, params: Iterable = ()) -> list[MediaRecord]:
        rows = self.conn.execute(sql, tuple(params)).fetchall()
        return [MediaRecord.from_row(r) for r in rows]

    def get_all_records(self) -> list[MediaRecord]:
        return self._query_records("SELECT * FROM media_items ORDER BY id")

    def get_item(self, item_id: int) -> MediaRecord | None:
        row = self.conn.execute(
            "SELECT * FROM media_items WHERE id = ?", (item_id,)
        ).fetchone()
        return MediaRecord.from_row(row) if row else None

    # ---- Tier 1: exact hash ---------------------------------------------
    def get_items_without_hash(self) -> list[MediaRecord]:
        return self._query_records(
            "SELECT * FROM media_items WHERE sha256_hash IS NULL AND is_duplicate = 0"
        )

    def update_hash(self, item_id: int, sha256_hash: str) -> None:
        self.conn.execute(
            "UPDATE media_items SET sha256_hash = ?, updated_at = datetime('now') WHERE id = ?",
            (sha256_hash, item_id),
        )

    def get_exact_hash_groups(self) -> list[list[int]]:
        rows = self.conn.execute(
            """
            SELECT GROUP_CONCAT(id) AS ids
            FROM media_items
            WHERE sha256_hash IS NOT NULL AND is_duplicate = 0
            GROUP BY sha256_hash
            HAVING COUNT(*) > 1
            """
        ).fetchall()
        return [[int(x) for x in r["ids"].split(",")] for r in rows]

    # ---- Tier 2/3: non-duplicate working set ----------------------------
    def get_non_duplicate_items(self) -> list[MediaRecord]:
        return self._query_records(
            "SELECT * FROM media_items WHERE is_duplicate = 0 ORDER BY taken_timestamp"
        )

    def update_phash(self, item_id: int, phash: str) -> None:
        self.conn.execute(
            "UPDATE media_items SET phash = ?, updated_at = datetime('now') WHERE id = ?",
            (phash, item_id),
        )

    def set_phash_cluster(self, item_ids: Iterable[int], cluster_id: int) -> None:
        self.conn.executemany(
            "UPDATE media_items SET phash_cluster_id = ? WHERE id = ?",
            [(cluster_id, i) for i in item_ids],
        )

    # ---- duplicate flagging (flag only; never deletes) [H1] -------------
    def flag_duplicate(
        self,
        dupe_id: int,
        duplicate_of: int,
        dedup_tier: str,
        hamming_distance: int | None = None,
    ) -> None:
        """Mark an item as a duplicate. Sets keeper_status to REVIEW — deletion
        remains a separate, human-gated step (Module D). No bytes are removed."""
        self.conn.execute(
            """
            UPDATE media_items
            SET is_duplicate = 1,
                duplicate_of = ?,
                dedup_tier = ?,
                hamming_distance = ?,
                keeper_status = 'REVIEW',
                updated_at = datetime('now')
            WHERE id = ?
            """,
            (duplicate_of, dedup_tier, hamming_distance, dupe_id),
        )

    # ---- Module C: vision ------------------------------------------------
    def get_items_for_vision(self) -> list[MediaRecord]:
        """Non-duplicate items still needing vision classification (resume-safe)."""
        return self._query_records(
            "SELECT * FROM media_items "
            "WHERE is_duplicate = 0 AND vision_status = 'PENDING' "
            "AND ingestion_status = 'COMPLETE'"
        )

    def update_vision(
        self,
        item_id: int,
        *,
        ai_caption: str | None,
        ai_label: str | None,
        blur_score: float | None,
        ocr_text_ratio: float | None,
        bucket: str | None,
        confidence: float | None,
        reasoning: str | None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE media_items
            SET ai_caption = ?, ai_label = ?, blur_score = ?, ocr_text_ratio = ?,
                classification_bucket = ?, classification_confidence = ?,
                classification_reasoning = ?, vision_status = 'COMPLETE',
                updated_at = datetime('now')
            WHERE id = ?
            """,
            (ai_caption, ai_label, blur_score, ocr_text_ratio,
             bucket, confidence, reasoning, item_id),
        )

    def mark_vision_error(self, item_id: int, message: str) -> None:
        self.conn.execute(
            "UPDATE media_items SET vision_status = 'ERROR', error_message = ?, "
            "updated_at = datetime('now') WHERE id = ?",
            (message[:500], item_id),
        )

    # ---- Module C: faces -------------------------------------------------
    def get_items_for_faces(self) -> list[MediaRecord]:
        return self._query_records(
            "SELECT * FROM media_items "
            "WHERE is_duplicate = 0 AND face_status = 'PENDING' "
            "AND ingestion_status = 'COMPLETE'"
        )

    def mark_faces_done(
        self, item_id: int, face_count: int, error: str | None = None
    ) -> None:
        status = "ERROR" if error else "COMPLETE"
        self.conn.execute(
            "UPDATE media_items SET face_count = ?, face_status = ?, "
            "error_message = COALESCE(?, error_message), updated_at = datetime('now') "
            "WHERE id = ?",
            (face_count, status, error[:500] if error else None, item_id),
        )

    def add_face_embedding(
        self,
        media_item_id: int,
        face_index: int,
        embedding: bytes,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> int:
        top, right, bottom, left = bbox or (None, None, None, None)
        cur = self.conn.execute(
            "INSERT OR REPLACE INTO face_embeddings "
            "(media_item_id, face_index, embedding, bbox_top, bbox_right, bbox_bottom, bbox_left) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (media_item_id, face_index, embedding, top, right, bottom, left),
        )
        return cur.lastrowid

    def get_all_face_embeddings(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, media_item_id, face_index, person_cluster_id, embedding "
            "FROM face_embeddings ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def set_face_cluster(self, embedding_id: int, person_cluster_id: str | None) -> None:
        self.conn.execute(
            "UPDATE face_embeddings SET person_cluster_id = ? WHERE id = ?",
            (person_cluster_id, embedding_id),
        )

    def set_person_cluster_ids(self, media_item_id: int, cluster_ids_json: str) -> None:
        self.conn.execute(
            "UPDATE media_items SET person_cluster_ids = ?, updated_at = datetime('now') "
            "WHERE id = ?",
            (cluster_ids_json, media_item_id),
        )

    # ---- review ----------------------------------------------------------
    def get_items_for_review(self) -> list[MediaRecord]:
        """Items needing a human decision: dedup duplicates flagged REVIEW, plus
        vision purge candidates — excluding anything already resolved KEEP/DELETE."""
        return self._query_records(
            "SELECT * FROM media_items "
            "WHERE keeper_status NOT IN ('KEEP', 'DELETE') "
            "AND (keeper_status = 'REVIEW' OR classification_bucket = 'ADHOC_PURGE') "
            "ORDER BY id"
        )

    def set_keeper_status(self, item_id: int, status: str) -> None:
        self.conn.execute(
            "UPDATE media_items SET keeper_status = ?, updated_at = datetime('now') WHERE id = ?",
            (status, item_id),
        )

    # ---- Module D: cleanup ----------------------------------------------
    def get_deletions(self) -> list[MediaRecord]:
        """Items the human review marked for deletion."""
        return self._query_records(
            "SELECT * FROM media_items WHERE keeper_status = 'DELETE' ORDER BY id"
        )

    def get_keepers(self) -> list[MediaRecord]:
        """Everything to retain: not marked DELETE (used to stage the curated set)."""
        return self._query_records(
            "SELECT * FROM media_items WHERE keeper_status != 'DELETE' "
            "AND ingestion_status = 'COMPLETE' ORDER BY id"
        )

    def set_upload_status(
        self, item_id: int, status: str, cloud_media_id: str | None = None
    ) -> None:
        self.conn.execute(
            "UPDATE media_items SET upload_status = ?, "
            "cloud_media_id = COALESCE(?, cloud_media_id), updated_at = datetime('now') "
            "WHERE id = ?",
            (status, cloud_media_id, item_id),
        )

    def get_uploaded_item_ids(self) -> set[int]:
        rows = self.conn.execute(
            "SELECT id FROM media_items WHERE upload_status = 'UPLOADED'"
        ).fetchall()
        return {r["id"] for r in rows}

    def cleanup_summary(self) -> dict:
        row = self.conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN keeper_status = 'DELETE' THEN 1 ELSE 0 END) AS to_delete,
                SUM(CASE WHEN keeper_status != 'DELETE' THEN 1 ELSE 0 END) AS to_keep,
                COALESCE(SUM(CASE WHEN keeper_status = 'DELETE' THEN file_size ELSE 0 END), 0)
                    AS delete_bytes
            FROM media_items WHERE ingestion_status = 'COMPLETE'
            """
        ).fetchone()
        return dict(row)

    # ---- stats -----------------------------------------------------------
    def vision_stats(self) -> dict:
        rows = self.conn.execute(
            "SELECT classification_bucket AS bucket, COUNT(*) AS c FROM media_items "
            "WHERE classification_bucket IS NOT NULL GROUP BY classification_bucket"
        ).fetchall()
        return {r["bucket"]: r["c"] for r in rows}

    def face_cluster_stats(self) -> dict:
        total = self.conn.execute(
            "SELECT COUNT(*) AS c FROM face_embeddings"
        ).fetchone()["c"]
        clusters = self.conn.execute(
            "SELECT COUNT(DISTINCT person_cluster_id) AS c FROM face_embeddings "
            "WHERE person_cluster_id IS NOT NULL"
        ).fetchone()["c"]
        noise = self.conn.execute(
            "SELECT COUNT(*) AS c FROM face_embeddings WHERE person_cluster_id IS NULL"
        ).fetchone()["c"]
        return {"faces": total, "clusters": clusters, "unclustered": noise}

    def dedup_stats(self) -> dict:
        row = self.conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(is_duplicate) AS duplicates,
                COALESCE(SUM(CASE WHEN is_duplicate = 1 THEN file_size ELSE 0 END), 0) AS reclaimable_bytes,
                SUM(CASE WHEN dedup_tier = 'EXACT_HASH' THEN 1 ELSE 0 END) AS exact_dupes,
                SUM(CASE WHEN dedup_tier = 'VISUAL_PHASH' THEN 1 ELSE 0 END) AS visual_dupes
            FROM media_items
            """
        ).fetchone()
        return dict(row)

    # ---- app_state -------------------------------------------------------
    def get_state(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM app_state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_state(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO app_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
