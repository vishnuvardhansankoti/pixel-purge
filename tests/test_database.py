from core_engine.models import MediaRecord


def test_insert_and_count(db):
    rec = MediaRecord(filename="a.jpg", local_path="/x/a.jpg", file_size=10, media_type="PHOTO")
    rid = db.insert_media_record(rec)
    assert rid == 1
    assert db.count_media() == 1
    assert rec.id == 1


def test_ingested_paths_uses_full_path(db):
    db.insert_media_record(MediaRecord("a.jpg", "/album1/a.jpg", 1, "PHOTO"))
    db.insert_media_record(MediaRecord("a.jpg", "/album2/a.jpg", 1, "PHOTO"))
    paths = db.get_ingested_paths()
    assert paths == {"/album1/a.jpg", "/album2/a.jpg"}  # same basename, both tracked


def test_flag_duplicate_sets_review_not_delete(db):
    keep = db.insert_media_record(MediaRecord("a.jpg", "/a.jpg", 10, "PHOTO"))
    dupe = db.insert_media_record(MediaRecord("b.jpg", "/b.jpg", 10, "PHOTO"))
    db.flag_duplicate(dupe, duplicate_of=keep, dedup_tier="EXACT_HASH")
    row = db.get_item(dupe)
    assert row.is_duplicate == 1
    assert row.duplicate_of == keep
    # never auto-DELETE from dedup — human-gated [H1]
    assert row.keeper_status == "REVIEW"


def test_app_state_roundtrip(db):
    assert db.get_state("watermark") is None
    db.set_state("watermark", "123")
    assert db.get_state("watermark") == "123"
    db.set_state("watermark", "456")
    assert db.get_state("watermark") == "456"
