"""Face clustering: pure assignment logic + (sklearn-gated) DBSCAN wrapper."""

import json

import numpy as np
import pytest

from core_engine.models import MediaRecord
from core_engine.vision import faces


def _seed_faces(db, specs):
    """specs: list of (media_item_id, n_faces). Returns embedding rows."""
    for i, (_mid, _n) in enumerate(specs, start=1):
        db.insert_media_record(MediaRecord(f"p{i}.jpg", f"/p{i}.jpg", 10, "PHOTO"))
    for mid, n in specs:
        for fi in range(n):
            db.add_face_embedding(mid, fi, faces.embedding_to_bytes(np.zeros(512)))
    return db.get_all_face_embeddings()


def test_embedding_roundtrip():
    vec = np.arange(512, dtype=np.float32)
    assert np.array_equal(faces.embedding_from_bytes(faces.embedding_to_bytes(vec)), vec)


def test_assign_person_clusters_writes_ids_and_aggregates(db):
    rows = _seed_faces(db, [(1, 1), (2, 1), (3, 1)])
    # items 1 & 2 are the same person (label 0); item 3 is noise (-1)
    labels = [0, 0, -1]
    stats = faces.assign_person_clusters(db, rows, labels)
    assert stats["clusters"] == 1

    embs = {r["media_item_id"]: r for r in db.get_all_face_embeddings()}
    assert embs[1]["person_cluster_id"] == "person_0000"
    assert embs[2]["person_cluster_id"] == "person_0000"
    assert embs[3]["person_cluster_id"] is None

    item1 = db.get_item(1)
    assert json.loads(item1.person_cluster_ids) == ["person_0000"]
    assert db.get_item(3).person_cluster_ids is None


def test_two_photo_person_is_not_dropped(db):
    # min_samples=2 means a person in exactly two photos still clusters [H4].
    rows = _seed_faces(db, [(1, 1), (2, 1)])
    same = np.array([1.0] + [0.0] * 511, dtype=np.float32)
    matrix = np.vstack([same, same.copy()])
    sklearn = pytest.importorskip("sklearn")  # noqa: F841
    labels = faces.cluster_faces(matrix, eps=0.45, min_samples=2)
    assert len(set(int(x) for x in labels) - {-1}) == 1  # one real cluster


def test_dbscan_separates_distinct_people(db):
    pytest.importorskip("sklearn")
    a = np.array([1.0, 0.0] + [0.0] * 510, dtype=np.float32)
    b = np.array([0.0, 1.0] + [0.0] * 510, dtype=np.float32)
    matrix = np.vstack([a, a.copy(), b, b.copy()])
    labels = faces.cluster_faces(matrix, eps=0.3, min_samples=2)
    assert len(set(int(x) for x in labels) - {-1}) == 2
