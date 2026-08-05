from core_engine.dedup.hash_dedup import run_tier1
from core_engine.dedup.keeper import select_keeper
from core_engine.ingestion import ingest
from core_engine.models import MediaRecord


def test_exact_duplicate_detected(db, takeout_dir):
    ingest(takeout_dir, db)
    flagged = run_tier1(db)
    # IMG_002 is a byte-identical copy of IMG_001 -> exactly one exact dupe.
    assert flagged == 1
    recs = {r.filename: r for r in db.get_all_records()}
    assert recs["IMG_002.jpg"].is_duplicate == 1
    assert recs["IMG_002.jpg"].dedup_tier == "EXACT_HASH"
    # The richer-metadata original is the keeper.
    assert recs["IMG_001.jpg"].is_duplicate == 0
    assert recs["IMG_002.jpg"].duplicate_of == recs["IMG_001.jpg"].id


def test_keeper_prefers_richest_metadata():
    poor = MediaRecord("a.jpg", "/a.jpg", 10, "PHOTO")
    poor.id = 1
    rich = MediaRecord("b.jpg", "/b.jpg", 10, "PHOTO")
    rich.id = 2
    rich.taken_timestamp = 1700
    rich.latitude, rich.longitude = 1.0, 2.0
    rich.user_description = "x"
    assert select_keeper([poor, rich]).id == 2


def test_keeper_prefers_earliest_when_metadata_equal():
    a = MediaRecord("a.jpg", "/a.jpg", 10, "PHOTO"); a.id = 1; a.taken_timestamp = 200
    b = MediaRecord("b.jpg", "/b.jpg", 10, "PHOTO"); b.id = 2; b.taken_timestamp = 100
    assert select_keeper([a, b]).id == 2  # earliest capture wins
