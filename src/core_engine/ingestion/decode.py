"""Image decoding with graceful HEIC/RAW support [M3].

PIL alone cannot open HEIC (iPhone default) or camera RAW. We register
``pillow-heif`` when available and fall back to ``rawpy`` for RAW. When a
decoder is missing, callers get a clear ``UnsupportedImageError`` rather than a
cryptic PIL failure, so ingestion can log-and-skip instead of crashing.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .media import HEIC_EXTENSIONS, RAW_EXTENSIONS

# Register HEIC/HEIF opener with Pillow if the plugin is installed.
_HEIC_OK = False
try:  # pragma: no cover - depends on optional dep
    import pillow_heif

    pillow_heif.register_heif_opener()
    _HEIC_OK = True
except Exception:  # noqa: BLE001
    _HEIC_OK = False

_RAW_OK = False
try:  # pragma: no cover - depends on optional dep
    import rawpy  # noqa: F401

    _RAW_OK = True
except Exception:  # noqa: BLE001
    _RAW_OK = False


class UnsupportedImageError(Exception):
    """Raised when a format needs an optional decoder that isn't installed."""


def heic_supported() -> bool:
    return _HEIC_OK


def raw_supported() -> bool:
    return _RAW_OK


def open_image(path: Path | str) -> Image.Image:
    """Open an image as RGB, handling HEIC/RAW when decoders are present.

    Raises UnsupportedImageError with an actionable message when the required
    optional decoder is missing.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in RAW_EXTENSIONS:
        if not _RAW_OK:
            raise UnsupportedImageError(
                f"{ext} is a RAW format; install the 'raw' extra "
                f"(pip install 'pixel-purge[raw]') to process {path.name}"
            )
        import rawpy

        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess()
        return Image.fromarray(rgb)

    if ext in HEIC_EXTENSIONS and not _HEIC_OK:
        raise UnsupportedImageError(
            f"{ext} needs the 'heic' extra (pip install 'pixel-purge[heic]') "
            f"to process {path.name}"
        )

    return Image.open(path).convert("RGB")
