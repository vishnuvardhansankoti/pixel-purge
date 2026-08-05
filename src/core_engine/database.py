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

    # ---- stats -----------------------------------------------------------
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
