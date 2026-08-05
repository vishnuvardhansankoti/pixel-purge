from core_engine.ingestion import ingest


def test_ingest_discovers_and_merges(db, takeout_dir):
    result = ingest(takeout_dir, db, resume=True)
    # 5 media files: IMG_001/002/003/010 + Screenshot_01
    assert result.discovered == 5
    assert result.ingested == 5
    assert result.with_sidecar == 5
    assert db.count_media() == 5

    recs = {r.filename: r for r in db.get_all_records()}
    assert recs["IMG_001.jpg"].taken_timestamp == 1_700_000_000
    assert recs["IMG_001.jpg"].latitude == 37.7749
    assert recs["IMG_001.jpg"].user_description == "beach day"


def test_ingest_resume_by_path_is_idempotent(db, takeout_dir):
    first = ingest(takeout_dir, db, resume=True)
    assert first.ingested == 5
    second = ingest(takeout_dir, db, resume=True)
    assert second.ingested == 0
    assert second.skipped_existing == 5
    assert db.count_media() == 5  # no duplicate rows on re-run [M1]


def test_dry_run_writes_nothing(db, takeout_dir):
    result = ingest(takeout_dir, db, resume=True, dry_run=True)
    assert result.discovered == 5
    assert db.count_media() == 0
