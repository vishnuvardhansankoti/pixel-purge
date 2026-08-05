"""GPS-based TRIP override — pure, testable.

If an item was captured far from home, it is almost certainly travel, so we
override the visual classifier to TRIP. We do NOT rescue ADHOC_PURGE items: a
screenshot or receipt taken on a trip is still junk. Items without coordinates
are returned unchanged.
"""

from __future__ import annotations

from dataclasses import replace

from ..dedup.geo import haversine_miles
from ..vision.clip_tagger import ClassificationResult
from ..vision.taxonomy import ADHOC_PURGE, TRIP


def apply_gps_override(
    result: ClassificationResult,
    latitude: float | None,
    longitude: float | None,
    home_lat: float,
    home_lon: float,
    trip_distance_miles: float = 50.0,
) -> ClassificationResult:
    """Return the classification, overridden to TRIP when far from home."""
    if latitude is None or longitude is None:
        return result
    if result.bucket == ADHOC_PURGE:  # don't rescue junk shot while travelling
        return result

    distance = haversine_miles(home_lat, home_lon, latitude, longitude)
    if distance > trip_distance_miles:
        return replace(
            result,
            bucket=TRIP,
            reasoning=f"{result.reasoning} [GPS override: {distance:.0f}mi from home]",
        )
    return result
