"""Module B orchestrator: run the three dedup tiers in order."""

from __future__ import annotations

from rich.console import Console

from ..database import Database
from .hash_dedup import run_tier1
from .spatial_bucket import partition
from .visual_dedup import run_tier3

console = Console()


def run_tier2(db: Database, gps_radius_m: float, time_window_min: int):
    """Build spatiotemporal buckets from the current non-duplicate working set."""
    items = db.get_non_duplicate_items()
    buckets = partition(items, gps_radius_m=gps_radius_m, time_window_min=time_window_min)
    multi = sum(1 for b in buckets if len(b) > 1)
    console.print(
        f"[green]Tier 2:[/green] {len(buckets)} bucket(s), {multi} with >1 item "
        f"(from {len(items)} non-duplicate items)."
    )
    return buckets


def run_dedup(
    db: Database,
    gps_radius_m: float = 100.0,
    time_window_min: int = 30,
    hamming_threshold: int = 8,
    text_hamming_threshold: int = 2,
    tiers: tuple[int, ...] = (1, 2, 3),
) -> dict[str, int]:
    """Run the requested dedup tiers. Tier 3 depends on Tier 2 buckets."""
    flagged_t1 = flagged_t3 = 0

    if 1 in tiers:
        flagged_t1 = run_tier1(db)

    buckets = None
    if 2 in tiers or 3 in tiers:
        buckets = run_tier2(db, gps_radius_m, time_window_min)

    if 3 in tiers:
        flagged_t3 = run_tier3(
            db, buckets, hamming_threshold=hamming_threshold,
            text_hamming_threshold=text_hamming_threshold,
        )

    return {"exact_flagged": flagged_t1, "visual_flagged": flagged_t3}
