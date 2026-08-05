"""Phase 4 (Module E) — GPS override, watermark idempotency, delta run, scheduling."""

from pathlib import Path

from core_engine.config import Config
from core_engine.delta import delta_run
from core_engine.delta.delta_run import classify_items
from core_engine.delta.gps_override import apply_gps_override
from core_engine.delta.scheduling import build_plist
from core_engine.vision.clip_tagger import ClassificationResult
from core_engine.vision.taxonomy import ADHOC_PURGE, FAMILY_KEEP, PROMPTS, TRIP


# ---- GPS override ------------------------------------------------------------
def _res(bucket):
    return ClassificationResult(bucket, 0.9, "x", "reason", 100.0, 0.05)


HOME = (37.7749, -122.4194)  # SF


def test_far_from_home_overrides_to_trip():
    r = apply_gps_override(_res(FAMILY_KEEP), 40.7128, -74.0060, *HOME, 50)  # NYC
    assert r.bucket == TRIP
    assert "GPS override" in r.reasoning


def test_near_home_unchanged():
    r = apply_gps_override(_res(FAMILY_KEEP), 37.7750, -122.4195, *HOME, 50)
    assert r.bucket == FAMILY_KEEP


def test_purge_not_rescued_by_gps():
    # A screenshot taken far from home is still junk.
    r = apply_gps_override(_res(ADHOC_PURGE), 40.7128, -74.0060, *HOME, 50)
    assert r.bucket == ADHOC_PURGE


def test_no_gps_unchanged():
    r = apply_gps_override(_res(FAMILY_KEEP), None, None, *HOME, 50)
    assert r.bucket == FAMILY_KEEP


# ---- fake classifier ---------------------------------------------------------
class _FakeClassifier:
    """Returns a fixed dominant prompt so no torch/open_clip is needed."""

    def __init__(self, dominant: str):
        self.dominant = dominant

    def classify(self, image) -> dict:
        return {p: (0.9 if p == self.dominant else 0.001) for p in PROMPTS}


PERSON = "a portrait photo of a person or a selfie"


# ---- delta run + watermark ---------------------------------------------------
def test_delta_classifies_and_sets_watermark(db, takeout_dir):
    cfg = Config()
    result = delta_run(takeout_dir, db, cfg, classifier=_FakeClassifier(PERSON))
    assert result.classified >= 1
    # watermark advanced to the newest capture time seen
    assert db.get_delta_watermark() > 0
    assert result.watermark == db.get_delta_watermark()


def test_delta_is_idempotent(db, takeout_dir):
    cfg = Config()
    first = delta_run(takeout_dir, db, cfg, classifier=_FakeClassifier(PERSON))
    assert first.classified >= 1
    wm = db.get_delta_watermark()
    # Re-running the same export classifies nothing new [H5].
    second = delta_run(takeout_dir, db, cfg, classifier=_FakeClassifier(PERSON))
    assert second.classified == 0
    assert db.get_delta_watermark() == wm


def test_classify_items_writes_buckets(db, takeout_dir):
    from core_engine.ingestion import ingest

    ingest(takeout_dir, db)
    items = db.get_items_for_delta()
    result = classify_items(db, items, _FakeClassifier(PERSON), *HOME, 50.0)
    assert result.classified == len(items)
    # every classified item now has a bucket
    assert all(db.get_item(i.id).classification_bucket for i in items)


# ---- scheduling --------------------------------------------------------------
def test_build_plist_shape():
    p = build_plist(Path("/data/takeout"), "/usr/local/bin/pixel-purge", day=1, hour=9)
    assert p["Label"] == "com.pixelpurge.delta"
    assert p["ProgramArguments"][:2] == ["/usr/local/bin/pixel-purge", "delta"]
    assert "/data/takeout" in p["ProgramArguments"]
    assert p["StartCalendarInterval"] == {"Day": 1, "Hour": 9, "Minute": 0}
