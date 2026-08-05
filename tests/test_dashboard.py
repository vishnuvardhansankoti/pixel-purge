"""Phase 5 (Module F) — local dashboard API + thumbnails."""

import json

import pytest
from PIL import Image

from core_engine.models import MediaRecord

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from core_engine.dashboard import create_app  # noqa: E402
from core_engine.dashboard import thumbs  # noqa: E402


@pytest.fixture
def client(tmp_path):
    from core_engine.database import Database

    db_path = tmp_path / "manifest.db"
    db = Database(db_path)
    db.init_schema()
    db.close()
    thumbs.clear_cache()
    return TestClient(create_app(db_path)), db_path


def _db(db_path):
    from core_engine.database import Database

    return Database(db_path)


def _add(db, filename, path, **kw):
    rid = db.insert_media_record(
        MediaRecord(filename, path, 1_000_000, "PHOTO", ingestion_status="COMPLETE")
    )
    for k, v in kw.items():
        db.conn.execute(f"UPDATE media_items SET {k}=? WHERE id=?", (v, rid))
    db.commit()
    return rid


def test_summary_endpoint(client):
    tc, db_path = client
    with _db(db_path) as db:
        _add(db, "a.jpg", "/a.jpg", keeper_status="DELETE", classification_bucket="ADHOC_PURGE")
    r = tc.get("/api/summary")
    assert r.status_code == 200
    assert r.json()["cleanup"]["to_delete"] == 1


def test_review_and_status_update(client):
    tc, db_path = client
    with _db(db_path) as db:
        rid = _add(db, "s.jpg", "/s.jpg", classification_bucket="ADHOC_PURGE")
    r = tc.get("/api/review")
    assert r.json()["count"] == 1
    assert r.json()["items"][0]["thumb"] == f"/thumb/{rid}"

    r = tc.post(f"/api/items/{rid}/status", json={"status": "DELETE"})
    assert r.status_code == 200
    with _db(db_path) as db:
        assert db.get_item(rid).keeper_status == "DELETE"


def test_review_bucket_filter(client):
    tc, db_path = client
    with _db(db_path) as db:
        _add(db, "a.jpg", "/a.jpg", classification_bucket="ADHOC_PURGE")
        _add(db, "b.jpg", "/b.jpg", is_duplicate=1, keeper_status="REVIEW")
    assert tc.get("/api/review?bucket=ADHOC_PURGE").json()["count"] == 1


def test_status_validation(client):
    tc, db_path = client
    with _db(db_path) as db:
        rid = _add(db, "a.jpg", "/a.jpg", classification_bucket="ADHOC_PURGE")
    assert tc.post(f"/api/items/{rid}/status", json={"status": "BOGUS"}).status_code == 400
    assert tc.post("/api/items/9999/status", json={"status": "KEEP"}).status_code == 404


def test_dedup_endpoint(client):
    tc, db_path = client
    with _db(db_path) as db:
        keeper = _add(db, "k.jpg", "/k.jpg")
        _add(db, "d.jpg", "/d.jpg", is_duplicate=1, duplicate_of=keeper,
             dedup_tier="VISUAL_PHASH", hamming_distance=3)
    data = tc.get("/api/dedup").json()
    assert data["count"] == 1
    assert data["clusters"][0]["keeper"]["filename"] == "k.jpg"
    assert len(data["clusters"][0]["members"]) == 1


def test_faces_list_name_and_merge(client):
    tc, db_path = client
    with _db(db_path) as db:
        a = _add(db, "a.jpg", "/a.jpg")
        b = _add(db, "b.jpg", "/b.jpg")
        db.add_face_embedding(a, 0, b"\x00" * 8)
        db.conn.execute("UPDATE face_embeddings SET person_cluster_id='person_0000'")
        db.add_face_embedding(b, 0, b"\x00" * 8)
        db.conn.execute("UPDATE face_embeddings SET person_cluster_id='person_0001' WHERE media_item_id=?", (b,))
        db.commit()

    assert tc.get("/api/faces").json()["count"] == 2
    assert tc.post("/api/faces/person_0000/name", json={"name": "Mom"}).status_code == 200
    assert tc.get("/api/faces/person_0000").json()["name"] == "Mom"
    # merge person_0001 into person_0000
    r = tc.post("/api/faces/merge", json={"source": "person_0001", "target": "person_0000"})
    assert r.json()["moved"] == 1
    assert tc.get("/api/faces").json()["count"] == 1


def test_thumbnail_streams_from_disk(client, tmp_path):
    tc, db_path = client
    p = tmp_path / "real.jpg"
    Image.new("RGB", (300, 200), (10, 120, 200)).save(p, "JPEG")
    with _db(db_path) as db:
        rid = _add(db, "real.jpg", str(p))
    r = tc.get(f"/thumb/{rid}?size=128")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    out = Image.open(__import__("io").BytesIO(r.content))
    assert max(out.size) <= 128


def test_thumbnail_missing_file_404(client):
    tc, db_path = client
    with _db(db_path) as db:
        rid = _add(db, "gone.jpg", "/nonexistent/gone.jpg")
    assert tc.get(f"/thumb/{rid}").status_code == 404


def test_index_served(client):
    tc, _ = client
    r = tc.get("/")
    assert r.status_code == 200
    assert "Pixel Purge" in r.text
