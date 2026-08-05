"""H1 — screenshot/document guard: distinct text-heavy images must not false-merge."""

import io

import numpy as np
from PIL import Image

from core_engine.dedup.visual_dedup import run_tier3
from core_engine.models import MediaRecord
from core_engine.vision import quality


def _screenshot(path, flips=0, seed=1, size=128):
    """App-like layout with identical low-freq structure; `flips` changes fine text."""
    arr = np.full((size, size, 3), 245)
    arr[0:16, :] = 40  # shared header bar
    rng = np.random.default_rng(seed)
    cells = [(y, x) for y in range(20, size - 4, 6) for x in range(4, size - 4, 6)]
    keep = set(range(len(cells)))
    for i in (rng.choice(len(cells), size=flips, replace=False) if flips else []):
        keep.discard(int(i))
    for i in keep:
        y, x = cells[i]
        arr[y:y + 3, x:x + 4] = 20
    Image.fromarray(arr.astype("uint8"), "RGB").save(path, "JPEG", quality=95)
    return path


def _add(db, path):
    return db.insert_media_record(
        MediaRecord(path.name, str(path), path.stat().st_size, "PHOTO",
                    ingestion_status="COMPLETE")
    )


def _bucket(db):
    return db.get_non_duplicate_items()


def test_fixtures_are_text_heavy(tmp_path):
    img = Image.open(_screenshot(tmp_path / "s.jpg"))
    assert quality.is_text_heavy(quality.text_density(img))


def test_distinct_screenshots_not_merged(db, tmp_path):
    _add(db, _screenshot(tmp_path / "a.jpg", flips=0))
    _add(db, _screenshot(tmp_path / "b.jpg", flips=6, seed=6))  # ~6 bits apart, text-heavy
    # Default thresholds: 8 for photos, 2 (near-exact) for text-heavy [H1].
    flagged = run_tier3(db, [_bucket(db)], hamming_threshold=8, text_hamming_threshold=2)
    assert flagged == 0
    assert all(r.is_duplicate == 0 for r in db.get_all_records())


def test_without_guard_they_would_merge(db, tmp_path):
    # Same pair, but relax the text threshold to the normal one -> they merge.
    _add(db, _screenshot(tmp_path / "a.jpg", flips=0))
    _add(db, _screenshot(tmp_path / "b.jpg", flips=6, seed=6))
    flagged = run_tier3(db, [_bucket(db)], hamming_threshold=8, text_hamming_threshold=8)
    assert flagged == 1  # proves the guard (not some other factor) is what protects them


def test_recompressed_screenshot_still_merges(db, tmp_path):
    base = _screenshot(tmp_path / "a.jpg", flips=0)
    # Near-exact re-encode of the SAME screenshot (~2 bits) -> still a duplicate.
    img = Image.open(base)
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=30)
    (tmp_path / "a_re.jpg").write_bytes(buf.getvalue())
    _add(db, base)
    _add(db, tmp_path / "a_re.jpg")
    flagged = run_tier3(db, [_bucket(db)], hamming_threshold=8, text_hamming_threshold=2)
    assert flagged == 1
