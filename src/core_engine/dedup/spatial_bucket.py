"""Tier 2: spatiotemporal partitioning — O(N log N).

Constrains the expensive Tier-3 visual comparison to small local groups. This
implementation fixes the v1.0 correctness bugs [H3]:

  * Temporal grouping uses **gap-based sessionization** (a new session starts
    only when the inter-photo gap exceeds the window), so a burst that straddles
    a fixed clock boundary is no longer split in half.
  * Spatial grouping uses **union-find over pairwise haversine distance** within
    a session, instead of a lexicographic (lat, lon) sort + single-anchor greedy
    walk (which mis-split chains and mis-merged unrelated points).

Items without a timestamp are grouped into one bucket; items without GPS inside a
session are grouped by time only (the correct fallback for indoor bursts and
sequential downloads).
"""

from __future__ import annotations

from ..models import MediaRecord
from .geo import haversine

Bucket = list[MediaRecord]


def _sessionize_by_time(items: list[MediaRecord], window_min: int) -> list[list[MediaRecord]]:
    """Split time-sorted items into sessions; new session when gap > window."""
    if not items:
        return []
    window_s = window_min * 60
    ordered = sorted(items, key=lambda x: x.taken_timestamp)
    sessions: list[list[MediaRecord]] = [[ordered[0]]]
    for item in ordered[1:]:
        if item.taken_timestamp - sessions[-1][-1].taken_timestamp <= window_s:
            sessions[-1].append(item)
        else:
            sessions.append([item])
    return sessions


def _cluster_by_gps(items: list[MediaRecord], radius_m: float) -> list[list[MediaRecord]]:
    """Union-find clustering: two items merge if within radius_m of each other."""
    n = len(items)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if haversine(
                items[i].latitude, items[i].longitude,
                items[j].latitude, items[j].longitude,
            ) <= radius_m:
                union(i, j)

    groups: dict[int, list[MediaRecord]] = {}
    for idx, item in enumerate(items):
        groups.setdefault(find(idx), []).append(item)
    return list(groups.values())


def partition(
    items: list[MediaRecord],
    gps_radius_m: float = 100.0,
    time_window_min: int = 30,
) -> list[Bucket]:
    """Partition non-duplicate items into spatiotemporal buckets."""
    with_time = [i for i in items if i.taken_timestamp is not None]
    without_time = [i for i in items if i.taken_timestamp is None]

    buckets: list[Bucket] = []

    for session in _sessionize_by_time(with_time, time_window_min):
        gps_items = [i for i in session if i.latitude is not None]
        no_gps_items = [i for i in session if i.latitude is None]

        if gps_items:
            buckets.extend(_cluster_by_gps(gps_items, gps_radius_m))
        if no_gps_items:
            buckets.append(no_gps_items)

    # Items with no timestamp at all: one catch-all bucket for Tier 3 to compare.
    if without_time:
        buckets.append(without_time)

    return buckets
