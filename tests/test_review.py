"""TUI review helpers + review working-set query."""

from core_engine.models import MediaRecord
from core_engine.tui import build_review_table, keep_item, mark_for_deletion


def _add(db, filename, **kw) -> int:
    rec = MediaRecord(filename, f"/{filename}", 1_000_000, "PHOTO")
    rid = db.insert_media_record(rec)
    for k, v in kw.items():
        db.conn.execute(f"UPDATE media_items SET {k} = ? WHERE id = ?", (v, rid))
    db.commit()
    return rid


def test_review_set_includes_dupes_and_purge_candidates(db):
    dupe = _add(db, "d.jpg", is_duplicate=1, keeper_status="REVIEW", duplicate_of=None)
    purge = _add(db, "s.jpg", classification_bucket="ADHOC_PURGE")
    _add(db, "keep.jpg", classification_bucket="FAMILY_KEEP")  # not in review set
    ids = {r.id for r in db.get_items_for_review()}
    assert ids == {dupe, purge}


def test_resolved_items_excluded(db):
    rid = _add(db, "d.jpg", is_duplicate=1, keeper_status="REVIEW")
    mark_for_deletion(db, rid)
    assert db.get_items_for_review() == []
    assert db.get_item(rid).keeper_status == "DELETE"


def test_keep_and_delete_transitions(db):
    a = _add(db, "a.jpg", classification_bucket="ADHOC_PURGE")
    b = _add(db, "b.jpg", classification_bucket="ADHOC_PURGE")
    keep_item(db, a)
    mark_for_deletion(db, b)
    assert db.get_item(a).keeper_status == "KEEP"
    assert db.get_item(b).keeper_status == "DELETE"


def test_build_review_table_renders(db):
    _add(db, "d.jpg", is_duplicate=1, keeper_status="REVIEW", dedup_tier="VISUAL_PHASH",
         hamming_distance=4, duplicate_of=1)
    table = build_review_table(db.get_items_for_review())
    assert table.row_count == 1
