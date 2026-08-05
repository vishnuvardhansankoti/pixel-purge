"""Tier 3: perceptual-hash comparison within spatiotemporal buckets — O(B·K²).

Duplicates are only ever *flagged* (is_duplicate=1, keeper_status=REVIEW); no
bytes are deleted here. Actual deletion is a later, human-gated step (Module D),
so a false pHash merge cannot silently destroy an original [H1].
"""

from __future__ import annotations

import imagehash
from rich.console import Console
from rich.progress import Progress

from ..database import Database
from ..models import MediaRecord
from ..ingestion.decode import UnsupportedImageError, open_image
from .keeper import select_keeper
from .spatial_bucket import Bucket

console = Console()


def compute_phash(path: str) -> imagehash.ImageHash:
    return imagehash.phash(open_image(path))


def _ensure_phashes(db: Database, bucket: Bucket) -> dict[int, imagehash.ImageHash]:
    """Compute + persist pHash for each item in the bucket. Returns id -> hash."""
    hashes: dict[int, imagehash.ImageHash] = {}
    for item in bucket:
        try:
            if item.phash:
                h = imagehash.hex_to_hash(item.phash)
            else:
                h = compute_phash(item.visual_path)
                db.update_phash(item.id, str(h))
            hashes[item.id] = h
        except (UnsupportedImageError, OSError, ValueError) as e:
            console.print(f"[yellow]phash skip[/yellow] {item.filename}: {e}")
    return hashes


def run_tier3(
    db: Database,
    buckets: list[Bucket],
    hamming_threshold: int = 8,
    next_cluster_id: int = 1,
) -> int:
    """pHash-compare within each bucket; flag near-duplicates. Returns flagged count."""
    flagged = 0
    cluster_id = next_cluster_id

    with Progress(console=console) as progress:
        task = progress.add_task("Tier 3: visual dedup...", total=len(buckets))
        for bucket in buckets:
            progress.advance(task)
            if len(bucket) < 2:
                continue

            hashes = _ensure_phashes(db, bucket)
            usable = [it for it in bucket if it.id in hashes]

            # Union-find: connect any pair within the Hamming threshold, so a
            # chain of near-identical frames becomes a single visual cluster.
            parent = {it.id: it.id for it in usable}

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for i, a in enumerate(usable):
                for b in usable[i + 1:]:
                    dist = hashes[a.id] - hashes[b.id]  # Hamming distance
                    if dist <= hamming_threshold:
                        parent[find(b.id)] = find(a.id)

            groups: dict[int, list[MediaRecord]] = {}
            for it in usable:
                groups.setdefault(find(it.id), []).append(it)

            for members in groups.values():
                if len(members) < 2:
                    continue
                keeper = select_keeper(members)
                db.set_phash_cluster([m.id for m in members], cluster_id)
                cluster_id += 1
                for m in members:
                    if m.id != keeper.id:
                        d = hashes[m.id] - hashes[keeper.id]
                        db.flag_duplicate(
                            m.id, duplicate_of=keeper.id,
                            dedup_tier="VISUAL_PHASH", hamming_distance=int(d),
                        )
                        flagged += 1

    db.commit()
    console.print(f"[green]Tier 3:[/green] flagged {flagged} visual duplicate(s).")
    return flagged
