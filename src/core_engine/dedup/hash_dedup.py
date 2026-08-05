"""Tier 1: exact binary (SHA-256) deduplication — O(N)."""

from __future__ import annotations

import hashlib

from rich.console import Console
from rich.progress import Progress

from ..database import Database
from .keeper import select_keeper

console = Console()


def sha256_file(filepath: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def run_tier1(db: Database) -> int:
    """Hash every un-hashed item, then flag exact-duplicate groups.

    Returns the number of items newly flagged as duplicates.
    """
    unprocessed = db.get_items_without_hash()
    with Progress(console=console) as progress:
        task = progress.add_task("Tier 1: hashing files...", total=len(unprocessed))
        for item in unprocessed:
            try:
                db.update_hash(item.id, sha256_file(item.local_path))
            except OSError as e:
                console.print(f"[yellow]hash skip[/yellow] {item.filename}: {e}")
            progress.advance(task)
    db.commit()

    flagged = 0
    for id_group in db.get_exact_hash_groups():
        items = [db.get_item(i) for i in id_group]
        keeper = select_keeper(items)
        for item in items:
            if item.id != keeper.id:
                db.flag_duplicate(item.id, duplicate_of=keeper.id, dedup_tier="EXACT_HASH")
                flagged += 1
    db.commit()

    console.print(f"[green]Tier 1:[/green] flagged {flagged} exact duplicate(s).")
    return flagged
