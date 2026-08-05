"""M9 — multi-frame video hashing + duration gate."""

import json

import imagehash
import pytest

from core_engine.dedup.video import duration_gate, frame_set_distance
from core_engine.dedup.visual_dedup import run_tier3
from core_engine.models import MediaRecord
from tests.conftest import make_image


# ---- pure logic --------------------------------------------------------------
def _h(seed):
    from pathlib import Path
    import tempfile
    p = Path(tempfile.mkdtemp()) / f"{seed}.jpg"
    make_image(p, seed=seed)
    from PIL import Image
    return imagehash.phash(Image.open(p))


def test_frame_set_distance_takes_best_match():
    a, b, c = _h(1), _h(2), _h(3)
    # Set {a,b} vs {b,c} shares b -> distance 0 (best matching frame).
    assert frame_set_distance([a, b], [b, c]) == 0
    # Disjoint distinct frames -> a positive distance.
    assert frame_set_distance([a], [c]) > 0


def test_frame_set_distance_empty():
    assert frame_set_distance([], [_h(1)]) > 1_000_000


def test_duration_gate():
    assert duration_gate(10.0, 11.0)            # 10% apart -> same clip
    assert not duration_gate(10.0, 20.0)        # 100% apart -> different
    assert duration_gate(None, 10.0)            # unknown duration passes
    assert duration_gate(0, 10.0)               # zero treated as unknown


# ---- Tier-3 integration (image files stand in for extracted frames) ----------
def _video(db, tmp_path, name, frame_seeds, duration):
    """Insert a VIDEO record whose keyframe_paths point to real images."""
    paths = []
    for i, s in enumerate(frame_seeds):
        p = tmp_path / f"{name}_kf{i}.jpg"
        make_image(p, seed=s)
        paths.append(str(p))
    rec = MediaRecord(name, str(tmp_path / name), 1000, "VIDEO",
                      ingestion_status="COMPLETE")
    rec.keyframe_path = paths[len(paths) // 2]
    rec.keyframe_paths = json.dumps(paths)
    rec.duration_seconds = duration
    return db.insert_media_record(rec)


def test_reencoded_video_matches_on_shared_frames(db, tmp_path):
    # Two clips sharing frames (seeds 1,2,3) but different ordering/trim -> merge.
    _video(db, tmp_path, "a.mp4", [1, 2, 3], duration=30.0)
    _video(db, tmp_path, "b.mp4", [2, 3, 4], duration=31.0)  # shares 2 & 3
    flagged = run_tier3(db, [db.get_non_duplicate_items()])
    assert flagged == 1


def test_duration_gate_blocks_unrelated_same_frame(db, tmp_path):
    # Same opening frame (seed 1) but very different lengths -> NOT merged [M9].
    _video(db, tmp_path, "a.mp4", [1, 5, 6], duration=5.0)
    _video(db, tmp_path, "b.mp4", [1, 7, 8], duration=120.0)
    flagged = run_tier3(db, [db.get_non_duplicate_items()])
    assert flagged == 0


def test_distinct_videos_not_merged(db, tmp_path):
    _video(db, tmp_path, "a.mp4", [10, 11, 12], duration=20.0)
    _video(db, tmp_path, "b.mp4", [80, 81, 82], duration=20.0)
    flagged = run_tier3(db, [db.get_non_duplicate_items()])
    assert flagged == 0
