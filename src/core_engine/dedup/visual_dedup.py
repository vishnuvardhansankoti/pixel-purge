"""Tier 3: perceptual-hash comparison within spatiotemporal buckets — O(B·K²).

Duplicates are only ever *flagged* (is_duplicate=1, keeper_status=REVIEW); no
bytes are deleted here. Actual deletion is a later, human-gated step (Module D),
so a false pHash merge cannot silently destroy an original [H1].

Screenshot/document guard [H1]: pHash discards the high-frequency detail that is
all that distinguishes two different text-dense images, so distinct receipts or
chat screenshots often sit within the normal Hamming threshold. We compute text
density inline (from the same image open) and require a **near-exact** threshold
before merging when either candidate is text-heavy.

Videos are compared by multi-frame pHash sets with a duration gate [M9] (see
``video.py``), so re-encodes/trims still match while clips of very different
length do not.
"""

from __future__ import annotations

from dataclasses import dataclass

import imagehash
from rich.console import Console
from rich.progress import Progress

from ..database import Database
from ..models import MediaRecord
from ..ingestion.decode import UnsupportedImageError, open_image
from ..vision import quality
from .keeper import select_keeper
from .spatial_bucket import Bucket
from .video import duration_gate, frame_set_distance

console = Console()


@dataclass
class _Analyzed:
    frames: list[imagehash.ImageHash]  # 1 for photos, N for videos [M9]
    text_heavy: bool
    is_video: bool
    duration: float | None


def compute_phash(path: str) -> imagehash.ImageHash:
    return imagehash.phash(open_image(path))


def _analyze_bucket(db: Database, bucket: Bucket) -> dict[int, _Analyzed]:
    """Compute + persist frame pHashes and a text-heavy flag for each item.

    Photos hash their single image; videos hash every extracted keyframe [M9].
    """
    out: dict[int, _Analyzed] = {}
    for item in bucket:
        try:
            primary = open_image(item.visual_path)
            text_heavy = quality.is_text_heavy(quality.text_density(primary))

            frames: list[imagehash.ImageHash] = []
            for fp in item.frame_paths:
                try:
                    frames.append(imagehash.phash(open_image(fp)))
                except (UnsupportedImageError, OSError, ValueError):
                    continue
            if not frames:
                frames = [imagehash.phash(primary)]

            if not item.phash:
                db.update_phash(item.id, str(frames[0]))
            out[item.id] = _Analyzed(
                frames, text_heavy, item.media_type == "VIDEO", item.duration_seconds
            )
        except (UnsupportedImageError, OSError, ValueError) as e:
            console.print(f"[yellow]phash skip[/yellow] {item.filename}: {e}")
    return out


def run_tier3(
    db: Database,
    buckets: list[Bucket],
    hamming_threshold: int = 8,
    text_hamming_threshold: int = 2,
    next_cluster_id: int = 1,
) -> int:
    """pHash-compare within each bucket; flag near-duplicates. Returns flagged count.

    A pair is merged when its Hamming distance is within the threshold — but if
    either item is text-heavy, the stricter `text_hamming_threshold` (near-exact)
    applies, so distinct screenshots/documents are not false-merged [H1].
    """
    flagged = 0
    cluster_id = next_cluster_id

    with Progress(console=console) as progress:
        task = progress.add_task("Tier 3: visual dedup...", total=len(buckets))
        for bucket in buckets:
            progress.advance(task)
            if len(bucket) < 2:
                continue

            analyzed = _analyze_bucket(db, bucket)
            usable = [it for it in bucket if it.id in analyzed]

            def pair_distance(x: int, y: int) -> int:
                ax, ay = analyzed[x], analyzed[y]
                # Two videos of very different length are not the same clip [M9].
                if ax.is_video and ay.is_video and not duration_gate(ax.duration, ay.duration):
                    return 1 << 30
                return frame_set_distance(ax.frames, ay.frames)

            # Union-find: connect any pair within its (guarded) threshold, so a
            # chain of near-identical frames becomes a single visual cluster.
            parent = {it.id: it.id for it in usable}

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for i, a in enumerate(usable):
                for b in usable[i + 1:]:
                    dist = pair_distance(a.id, b.id)
                    text_heavy = analyzed[a.id].text_heavy or analyzed[b.id].text_heavy
                    threshold = text_hamming_threshold if text_heavy else hamming_threshold
                    if dist <= threshold:
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
                        d = pair_distance(m.id, keeper.id)
                        db.flag_duplicate(
                            m.id, duplicate_of=keeper.id,
                            dedup_tier="VISUAL_PHASH", hamming_distance=int(d),
                        )
                        flagged += 1

    db.commit()
    console.print(f"[green]Tier 3:[/green] flagged {flagged} visual duplicate(s).")
    return flagged
