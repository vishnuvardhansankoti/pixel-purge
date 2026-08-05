"""Phase 3 (Module D) — curation, metadata restore, export, planner gate, upload."""

import csv
from pathlib import Path

import piexif
import pytest
from PIL import Image

from core_engine.cleanup import curate, export, planner, uploader
from core_engine.cleanup.metadata_restore import restore_metadata
from core_engine.models import MediaRecord


# ---- selection ---------------------------------------------------------------
def _add(db, filename, path, **kw) -> int:
    rec = MediaRecord(filename, path, 1000, "PHOTO", ingestion_status="COMPLETE")
    rid = db.insert_media_record(rec)
    for k, v in kw.items():
        db.conn.execute(f"UPDATE media_items SET {k} = ? WHERE id = ?", (v, rid))
    db.commit()
    return rid


def test_keepers_and_deletions_split(db):
    _add(db, "keep.jpg", "/keep.jpg", keeper_status="KEEP")
    _add(db, "del.jpg", "/del.jpg", keeper_status="DELETE")
    _add(db, "pend.jpg", "/pend.jpg", keeper_status="PENDING")
    keepers = {r.filename for r in db.get_keepers()}
    dels = {r.filename for r in db.get_deletions()}
    assert keepers == {"keep.jpg", "pend.jpg"}   # DELETE excluded from keepers
    assert dels == {"del.jpg"}


# ---- metadata restore (real, JPEG via piexif) [M2] --------------------------
def test_restore_metadata_writes_exif_to_jpeg(tmp_path):
    p = tmp_path / "x.jpg"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(p, "JPEG")
    rec = MediaRecord("x.jpg", str(p), p.stat().st_size, "PHOTO")
    rec.taken_timestamp = 1_700_000_000
    rec.latitude, rec.longitude = 37.5, -122.1

    result = restore_metadata(p, rec)
    assert result.ok

    exif = piexif.load(str(p))
    assert exif["Exif"][piexif.ExifIFD.DateTimeOriginal].startswith(b"2023")
    # piexif round-trips the GPS ref as bytes.
    assert exif["GPS"][piexif.GPSIFD.GPSLatitudeRef] in ("N", b"N")
    assert exif["GPS"][piexif.GPSIFD.GPSLongitudeRef] in ("W", b"W")


def test_restore_metadata_unsupported_format_is_reported(tmp_path):
    p = tmp_path / "x.png"
    Image.new("RGB", (16, 16)).save(p)
    rec = MediaRecord("x.png", str(p), 10, "PHOTO")
    rec.taken_timestamp = 1_700_000_000
    result = restore_metadata(p, rec)
    # PNG routes to exiftool; without it installed we get a clear skip, not a crash.
    if not result.ok:
        assert "exiftool" in result.reason.lower()


# ---- staging (real file ops) -------------------------------------------------
def test_stage_keepers_copies_files(db, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    p = src / "a.jpg"
    Image.new("RGB", (16, 16)).save(p, "JPEG")
    _add(db, "a.jpg", str(p), keeper_status="KEEP", taken_timestamp=1_700_000_000)

    staging = tmp_path / "staging"
    result = curate.stage_keepers(db, staging, restore=True)
    assert result.staged == 1
    assert (staging / "a.jpg").exists()


def test_stage_dry_run_writes_nothing(db, tmp_path):
    _add(db, "a.jpg", "/nonexistent/a.jpg", keeper_status="KEEP")
    staging = tmp_path / "staging"
    result = curate.stage_keepers(db, staging, dry_run=True)
    assert result.staged == 1
    assert not staging.exists()


# ---- export ------------------------------------------------------------------
def test_export_deletion_manifest(db, tmp_path):
    _add(db, "keep.jpg", "/keep.jpg", keeper_status="KEEP")
    _add(db, "del.jpg", "/del.jpg", keeper_status="DELETE",
         classification_bucket="ADHOC_PURGE")
    out = tmp_path / "dels.csv"
    n = export.export_deletion_manifest(db, out)
    assert n == 1
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 1 and rows[0]["filename"] == "del.jpg"
    assert rows[0]["keeper_status"] == "DELETE"


# ---- planner gate [C2] -------------------------------------------------------
def test_clean_slate_requires_backup_and_phrase():
    assert not planner.clean_slate_allowed("", False)
    assert not planner.clean_slate_allowed(planner.CONFIRM_PHRASE, False)
    assert not planner.clean_slate_allowed("nope", True)
    assert planner.clean_slate_allowed(planner.CONFIRM_PHRASE, True)
    assert planner.clean_slate_allowed(f"  {planner.CONFIRM_PHRASE}  ", True)


# ---- uploader (mocked service) resume ---------------------------------------
class _FakeMediaItems:
    def __init__(self, parent):
        self.parent = parent

    def batchCreate(self, body):
        items = body["newMediaItems"]
        self.parent.batch_calls += 1

        class _Exec:
            def execute(_self):
                return {"newMediaItemResults": [
                    {"status": {"code": 0}, "mediaItem": {"id": f"cloud_{i}"}}
                    for i, _ in enumerate(items)
                ]}
        return _Exec()


class _FakeService:
    def __init__(self):
        self.batch_calls = 0

    def mediaItems(self):
        return _FakeMediaItems(self)


def test_upload_resumes_and_skips_uploaded(db, tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    for name in ("a.jpg", "b.jpg"):
        (staging / name).write_bytes(b"x")
    a = _add(db, "a.jpg", "/a.jpg", keeper_status="KEEP")
    _add(db, "b.jpg", "/b.jpg", keeper_status="KEEP")
    db.set_upload_status(a, "UPLOADED")  # already done -> must be skipped
    db.commit()

    service = _FakeService()
    result = uploader.upload_curated_set(
        db, service, staging,
        upload_bytes=lambda svc, path: f"token_{path.name}",
        rate_limit_delay=0,
    )
    assert result.uploaded == 1
    assert result.skipped == 1
    assert db.get_item(a).upload_status == "UPLOADED"
