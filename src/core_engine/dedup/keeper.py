"""Keeper selection among a group of duplicates [M4].

For exact-hash duplicates the bytes are identical, so resolution/pixels are moot;
the tiebreak is metadata/provenance quality. The same ordering is reused for
visual (pHash) groups, where file_size does meaningfully differ.

Ordering (most-preferred first):
  1. Richest metadata (most non-null sidecar fields)
  2. Earliest taken_timestamp (the original capture)
  3. Largest file size (highest-quality encode; ties for exact dupes)
  4. Longest filename (often retains the original camera name)
  5. Lowest id (stable deterministic tiebreak)
"""

from __future__ import annotations

from ..models import MediaRecord


def _sort_key(item: MediaRecord):
    # Higher tuple sorts first, so negate "smaller-is-better" fields.
    return (
        item.metadata_richness,
        -(item.taken_timestamp if item.taken_timestamp is not None else float("inf")),
        item.file_size,
        len(item.filename),
        -(item.id if item.id is not None else 0),
    )


def select_keeper(items: list[MediaRecord]) -> MediaRecord:
    """Return the item to keep from a duplicate group (len >= 1)."""
    if not items:
        raise ValueError("select_keeper requires at least one item")
    return max(items, key=_sort_key)
