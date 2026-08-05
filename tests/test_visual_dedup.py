from core_engine.dedup.pipeline import run_dedup
from core_engine.dedup.spatial_bucket import partition
from core_engine.dedup.visual_dedup import run_tier3
from core_engine.ingestion import ingest


def test_near_duplicate_flagged_visual(db, takeout_dir):
    ingest(takeout_dir, db)
    # Run tier1 first so IMG_002 (exact) is removed from the working set.
    run_dedup(db, tiers=(1,))
    buckets = partition(db.get_non_duplicate_items(), gps_radius_m=100, time_window_min=30)
    flagged = run_tier3(db, buckets, hamming_threshold=8)

    recs = {r.filename: r for r in db.get_all_records()}
    # IMG_003 is a re-compressed copy of IMG_001 -> visual duplicate.
    assert recs["IMG_003.jpg"].is_duplicate == 1
    assert recs["IMG_003.jpg"].dedup_tier == "VISUAL_PHASH"
    assert recs["IMG_003.jpg"].hamming_distance is not None
    # The visually distinct photo in the same bucket must NOT be flagged.
    assert recs["IMG_010.jpg"].is_duplicate == 0
    assert flagged >= 1


def test_visual_duplicates_share_cluster_id(db, takeout_dir):
    ingest(takeout_dir, db)
    run_dedup(db, tiers=(1,))
    buckets = partition(db.get_non_duplicate_items(), gps_radius_m=100, time_window_min=30)
    run_tier3(db, buckets, hamming_threshold=8)
    recs = {r.filename: r for r in db.get_all_records()}
    assert recs["IMG_001.jpg"].phash_cluster_id is not None
    assert recs["IMG_003.jpg"].phash_cluster_id == recs["IMG_001.jpg"].phash_cluster_id


def test_full_pipeline_smoke(db, takeout_dir):
    ingest(takeout_dir, db)
    result = run_dedup(db)
    assert result["exact_flagged"] == 1
    assert result["visual_flagged"] >= 1
    stats = db.dedup_stats()
    assert stats["duplicates"] >= 2
