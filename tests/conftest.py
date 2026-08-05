"""Shared pytest fixtures — synthetic Takeout structures built at runtime."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core_engine.database import Database


# ---- image / sidecar builders ------------------------------------------------
def make_image(path: Path, seed: int = 0, size=(128, 128), quality=95) -> Path:
    """Write a deterministic, seed-distinct image.

    Uses a small sum of low-frequency sinusoids so that (a) different seeds
    produce perceptually different images with well-separated pHashes, and
    (b) re-encoding the same content stays pHash-close (realistic near-dup).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    h, w = size[1], size[0]
    yy, xx = np.mgrid[0:h, 0:w] / float(max(h, w))
    arr = np.zeros((h, w, 3), dtype=np.float64)
    for c in range(3):
        acc = np.zeros((h, w))
        for _ in range(3):
            fx, fy = rng.uniform(1.0, 4.0, 2)
            phase = rng.uniform(0, 2 * np.pi)
            acc += np.sin(2 * np.pi * (fx * xx + fy * yy) + phase)
        acc = (acc - acc.min()) / (np.ptp(acc) + 1e-9)
        arr[..., c] = acc
    img = Image.fromarray((arr * 255).astype("uint8"), "RGB")
    if path.suffix.lower() in (".jpg", ".jpeg"):
        img.save(path, "JPEG", quality=quality)
    else:
        img.save(path)
    return path


def make_sidecar(
    media_path: Path,
    taken_ts: int | None = None,
    lat: float | None = None,
    lon: float | None = None,
    description: str | None = None,
    title: str | None = None,
    views: int | None = None,
) -> Path:
    data: dict = {}
    if title is not None:
        data["title"] = title
    if description is not None:
        data["description"] = description
    if views is not None:
        data["imageViews"] = str(views)
    if taken_ts is not None:
        data["photoTakenTime"] = {"timestamp": str(taken_ts)}
        data["creationTime"] = {"timestamp": str(taken_ts)}
    if lat is not None and lon is not None:
        data["geoData"] = {"latitude": lat, "longitude": lon}
    sidecar = media_path.with_name(media_path.name + ".json")
    sidecar.write_text(json.dumps(data))
    return sidecar


# ---- fixtures ----------------------------------------------------------------
@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(tmp_path / "manifest.db")
    database.init_schema()
    yield database
    database.close()


@pytest.fixture
def takeout_dir(tmp_path) -> Path:
    """A synthetic Takeout tree matching the PRD fixture layout.

    Album1:
      IMG_001.jpg  (+ sidecar, GPS+time)      -> keeper
      IMG_002.jpg  (byte-identical to 001)    -> EXACT duplicate
      IMG_003.jpg  (re-compressed 001)        -> VISUAL near-duplicate
      IMG_010.jpg  (distinct, same GPS/time)  -> not a duplicate
    Screenshots:
      Screenshot_01.png (+ sidecar, no GPS)   -> distinct
    """
    root = tmp_path / "Takeout" / "Google Photos"
    album = root / "Album1"
    shots = root / "Screenshots"

    base_ts = 1_700_000_000
    lat, lon = 37.7749, -122.4194  # SF

    import shutil

    img1 = make_image(album / "IMG_001.jpg", seed=1, quality=95)
    make_sidecar(img1, taken_ts=base_ts, lat=lat, lon=lon,
                 description="beach day", title="IMG_001.jpg", views=5)

    # Exact duplicate: identical bytes.
    img2 = album / "IMG_002.jpg"
    shutil.copy2(img1, img2)
    make_sidecar(img2, taken_ts=base_ts + 30, lat=lat, lon=lon, title="IMG_002.jpg")

    # Near-duplicate: same content re-encoded at lower quality (few pHash bits differ).
    Image.open(img1).convert("RGB").save(album / "IMG_003.jpg", "JPEG", quality=30)
    make_sidecar(album / "IMG_003.jpg", taken_ts=base_ts + 60, lat=lat, lon=lon,
                 title="IMG_003.jpg")

    # Distinct photo, same place/time window — must NOT be flagged.
    img10 = make_image(album / "IMG_010.jpg", seed=99, quality=95)
    make_sidecar(img10, taken_ts=base_ts + 90, lat=lat, lon=lon, title="IMG_010.jpg")

    shot = make_image(shots / "Screenshot_01.png", seed=7)
    make_sidecar(shot, taken_ts=base_ts + 100_000, title="Screenshot_01.png")

    return tmp_path / "Takeout"
