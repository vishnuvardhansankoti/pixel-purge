"""On-disk thumbnail rendering with a small in-memory LRU cache.

Streams a downscaled JPEG for a media item straight from the local file (video
keyframe when present), decoding HEIC/RAW via the shared decoder. This is the
piece that lets a local browser render the library — a hosted app could not.
"""

from __future__ import annotations

import io
from functools import lru_cache

from ..ingestion.decode import open_image


@lru_cache(maxsize=512)
def _render(path: str, size: int) -> bytes:
    img = open_image(path)
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=82)
    return buf.getvalue()


def render_thumbnail(path: str, size: int = 256) -> bytes:
    """Return JPEG bytes for a downscaled thumbnail (cached by path+size)."""
    return _render(path, size)


def clear_cache() -> None:
    _render.cache_clear()
