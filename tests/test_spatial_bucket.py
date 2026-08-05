"""Tier 2 spatiotemporal partitioning [H3]."""

from core_engine.dedup.spatial_bucket import partition
from core_engine.models import MediaRecord


def _item(i, ts=None, lat=None, lon=None):
    r = MediaRecord(f"{i}.jpg", f"/{i}.jpg", 10, "PHOTO")
    r.id = i
    r.taken_timestamp = ts
    r.latitude, r.longitude = lat, lon
    return r


def test_gap_based_sessionization_does_not_split_bursts():
    # A burst at 20-min spacing should stay one session even across a clock hour.
    items = [_item(i, ts=1000 + i * 20 * 60) for i in range(6)]
    buckets = partition(items, time_window_min=30)
    assert len(buckets) == 1
    assert len(buckets[0]) == 6


def test_large_gap_creates_new_session():
    a = _item(1, ts=1000)
    b = _item(2, ts=1000 + 60 * 60)  # 1h later > 30m window
    buckets = partition([a, b], time_window_min=30)
    assert len(buckets) == 2


def test_gps_subclusters_within_a_session():
    # Same time window, two far-apart locations -> two spatial buckets.
    near1 = _item(1, ts=1000, lat=37.7749, lon=-122.4194)
    near2 = _item(2, ts=1100, lat=37.77495, lon=-122.41945)  # ~7m away
    far = _item(3, ts=1200, lat=40.7128, lon=-74.0060)       # NYC
    buckets = partition([near1, near2, far], gps_radius_m=100, time_window_min=30)
    sizes = sorted(len(b) for b in buckets)
    assert sizes == [1, 2]


def test_gps_less_items_bucket_by_time_only():
    a = _item(1, ts=1000)
    b = _item(2, ts=1200)
    buckets = partition([a, b], time_window_min=30)
    assert len(buckets) == 1 and len(buckets[0]) == 2


def test_items_without_timestamp_grouped_together():
    a = _item(1, ts=None)
    b = _item(2, ts=None)
    buckets = partition([a, b])
    assert any(len(b_) == 2 for b_ in buckets)
